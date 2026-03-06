import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import Config
from backend.query_translator import translate_query, get_translator
from backend.embeddings import get_loader
from backend.ingestion import IndexService
from backend.retriever import ImageSearchService
from backend.logger import GLOBAL_LOGGER as log
from backend.exception.custom_exception import SemanticImageSearchException


app = FastAPI(
    title="Semantic Image Search API",
    description="CLIP + Qdrant + LLM Query Translator",
    version="1.0",
)

# Dev CORS: allow local frontend (file:// or localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local images for frontend display
Config.IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(Config.IMAGES_ROOT)), name="images")

# Lazy singletons
search_service = None
index_service = None


@app.on_event("startup")
def init_services():
    global search_service, index_service
    search_service = ImageSearchService()
    index_service = IndexService()

    if Config.PRELOAD_TRANSLATOR_ON_STARTUP:
        log.info("Preloading query translator on startup")
        get_translator()
        log.info("Query translator preloaded")

    if Config.PRELOAD_CLIP_ON_STARTUP:
        log.info("Preloading CLIP embedder on startup")
        get_loader()
        log.info("CLIP embedder preloaded")

    log.info("Services initialized successfully")


def _image_url(path_value: str) -> Optional[str]:
    if not path_value:
        return None
    p = Path(path_value)
    image_root = Config.IMAGES_ROOT.resolve()

    candidates = []
    if p.is_absolute():
        candidates.append(p.resolve())
    else:
        candidates.append((Config.BASE_DIR / p).resolve())

    for cand in candidates:
        try:
            rel = cand.relative_to(image_root)
            return f"/images/{rel.as_posix()}"
        except Exception:
            pass

    return None


def _unique_result_items(results) -> list[dict]:
    seen_paths = set()
    items = []
    for p in results.points:
        path_value = p.payload.get("path")
        if path_value in seen_paths:
            continue
        seen_paths.add(path_value)
        items.append(
            {
                "filename": p.payload.get("filename"),
                "path": path_value,
                "category": p.payload.get("category"),
                "score": p.score,
                "image_url": _image_url(path_value),
            }
        )
    return items


# ---------------------------------------------------------
# INGEST ENDPOINT
# ---------------------------------------------------------
@app.post("/ingest")
def ingest_images(
    folder_path: Optional[str] = Query(None, description="Folder of images to index"),
):
    folder = folder_path or str(Config.IMAGES_ROOT)
    log.info("Ingest request received", folder=folder)

    try:
        index_service.index_folder(folder)
        log.info("Ingestion completed", folder=folder)
        return {"message": f"Indexed images from {folder}"}

    except Exception as e:
        log.error("Ingestion failed", folder=folder, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ---------------------------------------------------------
# TRANSLATE ENDPOINT
# ---------------------------------------------------------
@app.get("/translate")
def translate(q: str):
    log.info("Translate request received", query=q)

    try:
        translated = translate_query(q)
        log.info("Query translated", original=q, translated=translated)
        return {"input": q, "translated": translated}

    except Exception as e:
        log.error("Translation failed", query=q, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ---------------------------------------------------------
# TEXT SEARCH ENDPOINT
# ---------------------------------------------------------
@app.get("/search-text")
def search_text_endpoint(
    q: str,
    k: int = Query(5, ge=1, le=50),
    category: Optional[str] = None,
    save_results: bool = False,
):
    log.info("Text search request received", query=q, k=k, category=category)

    try:
        translated = translate_query(q)
        log.info("Query translated for text search", translated=translated)

        metadata_filter = {"category": category} if category else None

        results = search_service.search_by_text(
            translated,
            k=k,
            metadata_filter=metadata_filter,
        )

        log.info("Text search completed", total_results=len(results.points))

        resp = _unique_result_items(results)

        folder = None
        if save_results and results.points:
            folder = search_service.save_results(results)
            log.info("Search results saved locally", folder=folder)

        return {
            "query": q,
            "translated": translated,
            "k": k,
            "saved_folder": folder,
            "results": resp,
        }

    except Exception as e:
        log.error("Text search failed", query=q, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ---------------------------------------------------------
# IMAGE SEARCH ENDPOINT
# ---------------------------------------------------------
@app.post("/search-image")
def search_image_endpoint(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=50),
    category: Optional[str] = None,
    save_results: bool = False,
):
    log.info("Image search request received", filename=file.filename)

    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "Only image files allowed"})

        Config.QUERY_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        query_path = Config.QUERY_IMAGE_ROOT / file.filename

        with query_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        log.info("Uploaded query image saved", path=str(query_path))

        metadata_filter = {"category": category} if category else None

        results = search_service.search_by_image(
            str(query_path),
            k=k,
            metadata_filter=metadata_filter,
        )

        resp = _unique_result_items(results)

        folder = None
        if save_results and results.points:
            folder = search_service.save_results(results)
            log.info("Search results saved locally", folder=folder)

        return {
            "query_image": str(query_path),
            "k": k,
            "saved_folder": folder,
            "results": resp,
        }

    except Exception as e:
        log.error("Image search failed", filename=file.filename, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ---------------------------------------------------------
# RESET COLLECTION
# ---------------------------------------------------------
@app.post("/reset")
def reset_collection():
    log.warning("Reset request received")
    try:
        index_service.clear_collection()
        log.info("Collection cleared successfully")
        return {"message": "Collection cleared"}
    except Exception as e:
        log.error("Reset failed", error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ---------------------------------------------------------
# REINDEX (CLEAR + INGEST)
# ---------------------------------------------------------
@app.post("/reindex")
def reindex_images(
    folder_path: Optional[str] = Query(None, description="Folder of images to reindex"),
):
    folder = folder_path or str(Config.IMAGES_ROOT)
    log.warning("Reindex request received", folder=folder)
    try:
        index_service.clear_collection()
        index_service.index_folder(folder)
        log.info("Reindex completed", folder=folder)
        return {"message": f"Reindexed images from {folder}"}
    except Exception as e:
        log.error("Reindex failed", folder=folder, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})

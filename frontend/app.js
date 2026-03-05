const apiBase = "http://127.0.0.1:8000";

const modeInputs = document.querySelectorAll("input[name='mode']");
const textField = document.getElementById("textField");
const imageField = document.getElementById("imageField");
const queryText = document.getElementById("queryText");
const queryImage = document.getElementById("queryImage");
const kRange = document.getElementById("kRange");
const kValue = document.getElementById("kValue");
const searchBtn = document.getElementById("searchBtn");
const statusEl = document.getElementById("status");
const translatedEl = document.getElementById("translated");
const resultsEl = document.getElementById("results");

function setMode(mode) {
  if (mode === "text") {
    textField.classList.remove("hidden");
    imageField.classList.add("hidden");
  } else {
    textField.classList.add("hidden");
    imageField.classList.remove("hidden");
  }
}

modeInputs.forEach((el) => {
  el.addEventListener("change", (e) => setMode(e.target.value));
});

kRange.addEventListener("input", (e) => {
  kValue.textContent = e.target.value;
});

function setStatus(text) {
  statusEl.textContent = text;
}

function setTranslated(text) {
  if (text) {
    translatedEl.textContent = `Translated: ${text}`;
    translatedEl.classList.remove("hidden");
  } else {
    translatedEl.textContent = "";
    translatedEl.classList.add("hidden");
  }
}

function clearResults() {
  resultsEl.innerHTML = "";
}

function renderResults(items) {
  clearResults();
  if (!items || items.length === 0) {
    setStatus("No results");
    return;
  }

  items.forEach((item, idx) => {
    const li = document.createElement("li");
    li.className = "result-item";
    const imageUrl = item.image_url ? `${apiBase}${encodeURI(item.image_url)}` : "";
    li.innerHTML = `
      <div><strong>#${idx + 1}</strong></div>
      <div>Similarity score: ${item.score ?? ""}</div>
      ${imageUrl ? `<img class="result-image" src="${imageUrl}" alt="${item.filename ?? "result"}" />` : ""}
    `;
    resultsEl.appendChild(li);
  });
}

async function searchText() {
  const q = queryText.value.trim();
  const k = kRange.value;
  if (!q) {
    setStatus("Enter a text query");
    return;
  }

  setStatus("Searching...");
  searchBtn.disabled = true;

  const url = `${apiBase}/search-text?q=${encodeURIComponent(q)}&k=${k}`;
  const res = await fetch(url);
  const data = await res.json();
  setTranslated(data.translated);
  renderResults(data.results || []);
  setStatus("Done");
  searchBtn.disabled = false;
}

async function searchImage() {
  const file = queryImage.files[0];
  const k = kRange.value;
  if (!file) {
    setStatus("Select an image file");
    return;
  }

  setStatus("Searching...");
  searchBtn.disabled = true;

  const form = new FormData();
  form.append("file", file);

  const url = `${apiBase}/search-image?k=${k}`;
  const res = await fetch(url, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  setTranslated(data.translated);
  renderResults(data.results || []);
  setStatus("Done");
  searchBtn.disabled = false;
}

searchBtn.addEventListener("click", async () => {
  const mode = document.querySelector("input[name='mode']:checked").value;
  try {
    setTranslated("");
    if (mode === "text") {
      await searchText();
    } else {
      await searchImage();
    }
  } catch (err) {
    console.error(err);
    setStatus("Error - check console/server");
    searchBtn.disabled = false;
  }
});

setMode("text");

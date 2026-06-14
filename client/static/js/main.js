/**
 * client/static/js/main.js
 * SecureLens — Frontend Logic
 * Handles file upload, drag-drop, API calls, result rendering.
 */

// ── DOM References ─────────────────────────────────────────────────────
const dropZone    = document.getElementById("dropZone");
const dropContent = document.getElementById("dropContent");
const previewImg  = document.getElementById("previewImg");
const fileInput   = document.getElementById("fileInput");
const browseBtn   = document.getElementById("browseBtn");
const analyzeBtn  = document.getElementById("analyzeBtn");
const resetBtn    = document.getElementById("resetBtn");

const idleState    = document.getElementById("idleState");
const loadingState = document.getElementById("loadingState");
const resultState  = document.getElementById("resultState");

// Pipeline steps
const pipeSteps = [
  document.getElementById("pipe1"),
  document.getElementById("pipe2"),
  document.getElementById("pipe3"),
  document.getElementById("pipe4"),
];

// Loading steps
const loadSteps = [
  document.getElementById("ls1"),
  document.getElementById("ls2"),
  document.getElementById("ls3"),
  document.getElementById("ls4"),
];

let selectedFile = null;

// ── File Selection ─────────────────────────────────────────────────────

browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (e) => {
  if (e.target !== analyzeBtn && e.target !== browseBtn) {
    fileInput.click();
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// ── Drag & Drop ────────────────────────────────────────────────────────

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ── Handle Selected File ───────────────────────────────────────────────

function handleFile(file) {
  const allowed = ["image/png", "image/jpeg", "image/jpg"];
  if (!allowed.includes(file.type)) {
    alert("Please upload a PNG or JPEG image.");
    return;
  }
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    dropContent.classList.add("hidden");
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
  setPipelineStep(0);
}

// ── Analyze Button ─────────────────────────────────────────────────────

analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;

  // Switch to loading state
  showState("loading");
  analyzeBtn.disabled = true;
  setPipelineStep(1);

  // Animate loading steps
  await animateLoadingSteps();

  // Build form data
  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || "Server error");
    }

    setPipelineStep(3);
    renderResult(data);
    showState("result");

  } catch (err) {
    console.error("Analysis failed:", err);
    alert("Analysis failed: " + err.message);
    showState("idle");
    analyzeBtn.disabled = false;
    setPipelineStep(0);
  }
}

// ── Animate Loading Steps ──────────────────────────────────────────────

async function animateLoadingSteps() {
  const messages = [
    "Preprocessing image...",
    "Encrypting pixels with CKKS...",
    "Running encrypted inference...",
    "Decrypting result...",
  ];

  for (let i = 0; i < loadSteps.length; i++) {
    // Mark previous as done
    if (i > 0) {
      loadSteps[i - 1].classList.remove("active");
      loadSteps[i - 1].classList.add("done");
      loadSteps[i - 1].textContent =
        "✅ " + loadSteps[i - 1].textContent.replace("⏳ ", "");
    }
    loadSteps[i].classList.add("active");
    document.getElementById("loadingText").textContent = messages[i];

    // Update pipeline
    setPipelineStep(i + 1);

    // Wait (except last step — real API handles that)
    if (i < loadSteps.length - 1) {
      await sleep(600);
    }
  }
}

// ── Render Result ──────────────────────────────────────────────────────

function renderResult(data) {
  const isPneumonia = data.prediction === "Pneumonia";

  // Verdict box
  const verdictBox   = document.getElementById("verdictBox");
  const verdictIcon  = document.getElementById("verdictIcon");
  const verdictLabel = document.getElementById("verdictLabel");
  const verdictConf  = document.getElementById("verdictConf");

  verdictBox.className = "verdict " + (isPneumonia ? "pneumonia" : "normal");
  verdictIcon.textContent  = isPneumonia ? "⚠️" : "✅";
  verdictLabel.textContent = data.prediction;
  verdictConf.textContent  = `Confidence: ${data.confidence}%`;

  // Probability bars (animate after short delay)
  setTimeout(() => {
    document.getElementById("barNormal").style.width    = data.normal_score + "%";
    document.getElementById("barPneumonia").style.width = data.pneumonia_score + "%";
  }, 100);

  document.getElementById("pctNormal").textContent    = data.normal_score + "%";
  document.getElementById("pctPneumonia").textContent = data.pneumonia_score + "%";

  // Encryption info grid
  const encGrid = document.getElementById("encGrid");
  const info    = data.encryption_info;
  const encItems = [
    ["Scheme",    info.scheme],
    ["Library",   info.library],
    ["Security",  info.security_bits + "-bit"],
    ["Poly Mod",  info.poly_modulus_degree],
    ["Vec Size",  (info.feature_vector_size || info.pixel_vector_size || 512) + " values"],
    ["CT Size",   info.ciphertext_size_kb + " KB"],
  ];
  encGrid.innerHTML = encItems.map(([k, v]) =>
    `<div class="enc-item"><span class="enc-key">${k}: </span>
     <span class="enc-val">${v}</span></div>`
  ).join("");

  // Pipeline trace
  const stepsList = document.getElementById("stepsList");
  stepsList.innerHTML = data.pipeline_steps
    .map(s => `<li>${s}</li>`)
    .join("");

  // Mark all pipeline steps done
  setPipelineStep(4);
}

// ── Reset ──────────────────────────────────────────────────────────────

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  previewImg.src = "";
  previewImg.classList.add("hidden");
  dropContent.classList.remove("hidden");
  fileInput.value = "";
  analyzeBtn.disabled = true;

  // Reset loading steps text
  const stepTexts = [
    "⏳ Preprocessing image",
    "⏳ Encrypting pixels (CKKS)",
    "⏳ Running encrypted inference",
    "⏳ Decrypting result",
  ];
  loadSteps.forEach((s, i) => {
    s.className = "load-step";
    s.textContent = stepTexts[i];
  });

  showState("idle");
  setPipelineStep(0);
});

// ── Helpers ────────────────────────────────────────────────────────────

function showState(state) {
  idleState.classList.add("hidden");
  loadingState.classList.add("hidden");
  resultState.classList.add("hidden");

  if (state === "idle")    idleState.classList.remove("hidden");
  if (state === "loading") loadingState.classList.remove("hidden");
  if (state === "result")  resultState.classList.remove("hidden");
}

function setPipelineStep(activeIndex) {
  pipeSteps.forEach((step, i) => {
    step.classList.remove("active", "done");
    if (i < activeIndex)      step.classList.add("done");
    else if (i === activeIndex) step.classList.add("active");
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Health Check on Load ───────────────────────────────────────────────

window.addEventListener("load", async () => {
  try {
    const res  = await fetch("/health");
    const data = await res.json();
    if (!data.model_ready) {
      console.warn("Model not loaded — run train_model.py first.");
    }
  } catch (e) {
    console.warn("Server not reachable:", e.message);
  }
});
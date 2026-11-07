
const API = {
  mri: "/predict_mri",
  audio: "/predict_audio",
  fusion: "/predict_fusion",
};

const LABELS = ["Normal", "Leve", "Moderado"];
const RISK = {
  Normal:   { text: "Bajo",  css: "bajo",  detail: "No se observan señales relevantes de deterioro cognitivo." },
  Leve:     { text: "Medio", css: "medio", detail: "Hay señales leves. Se recomienda seguimiento con un profesional." },
  Moderado: { text: "Alto",  css: "alto",  detail: "Hay señales marcadas. Se recomienda consultar a un especialista pronto." },
};

const $ = (id) => document.getElementById(id);
const state = { mriFile: null, audioFile: null, recorder: null, chunks: [], timerId: null };

/* ---------- Selección de archivos ---------- */
function setMri(file) {
  if (!file) return;
  if (!/\.(png|jpe?g)$/i.test(file.name)) { return showStatus("La MRI debe ser PNG, JPG o JPEG.", true); }
  state.mriFile = file;
  $("mri-name").textContent = file.name;
  const img = $("mri-preview");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  showStatus("");
  refreshButtons();
}

function setAudio(file) {
  if (!file) return;
  state.audioFile = file;
  $("audio-name").textContent = file.name;
  const player = $("audio-preview");
  player.src = URL.createObjectURL(file);
  player.hidden = false;
  showStatus("");
  refreshButtons();
}

function setupDropzone(zoneId, inputId, handler) {
  const zone = $(zoneId), input = $(inputId);
  input.addEventListener("change", () => handler(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => handler(e.dataTransfer.files[0]));
}

/* ---------- Grabación de audio ---------- */
async function toggleRecording() {
  const btn = $("rec-btn");
  if (state.recorder && state.recorder.state === "recording") {
    state.recorder.stop();
    return;
  }
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    return showStatus("Tu navegador no permite grabar. Sube un archivo de audio.", true);
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.chunks = [];
    state.recorder = new MediaRecorder(stream);
    state.recorder.ondataavailable = (e) => state.chunks.push(e.data);
    state.recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(state.timerId);
      btn.textContent = "● Grabar voz";
      btn.classList.remove("recording");
      const type = state.recorder.mimeType || "audio/webm";
      const ext = type.includes("ogg") ? "ogg" : "webm";
      setAudio(new File(state.chunks, `grabacion.${ext}`, { type }));
    };
    state.recorder.start();
    btn.textContent = "■ Detener";
    btn.classList.add("recording");
    const t0 = Date.now();
    state.timerId = setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000);
      $("rec-timer").textContent =
        String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
    }, 250);
  } catch (err) {
    showStatus("No se pudo acceder al micrófono (revisa los permisos).", true);
  }
}

/* ---------- Llamadas a la API ---------- */
// Devuelve { probs:[3], demo:boolean }
async function predictFile(url, file) {
  try {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch(url, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (res.ok && Array.isArray(data.probabilities) && data.probabilities.length === 3) {
      return { probs: data.probabilities.map(Number), demo: false };
    }
    if (res.status >= 400 && res.status < 500) {
      throw new Error(data.error || "La API rechazó el archivo.");
    }
    // 503 / formato inesperado -> modo demo
  } catch (err) {
    if (err instanceof TypeError === false) throw err; // error "real" de validación
  }
  return { probs: fakeProbs(), demo: true };
}

async function predictFusion(mri, audio) {
  try {
    const res = await fetch(API.fusion, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mri_probabilities: mri, audio_probabilities: audio }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data.prediction && data.probabilities) {
      const probs = LABELS.map((l) => Number(data.probabilities[l] ?? 0));
      return { label: data.prediction, probs, demo: false };
    }
  } catch (_) { /* cae al promedio */ }
  const probs = mri.map((v, i) => (v + audio[i]) / 2);
  return { label: LABELS[argmax(probs)], probs, demo: true };
}

/* ---------- Flujo principal ---------- */
async function analyze() {
  const btn = $("analyze-btn");
  btn.disabled = true;
  $("result").hidden = true;
  showStatus("Analizando…");
  try {
    let mri = null, audio = null;
    if (state.mriFile) mri = await predictFile(API.mri, state.mriFile);
    if (state.audioFile) audio = await predictFile(API.audio, state.audioFile);

    let label, probs, demo, sources = [];
    if (mri && audio) {
      const fused = await predictFusion(mri.probs, audio.probs);
      ({ label, probs } = fused);
      demo = fused.demo || mri.demo || audio.demo;
      sources = ["Fusión de MRI + Audio"];
    } else {
      const single = mri || audio;
      probs = single.probs;
      label = LABELS[argmax(probs)];
      demo = single.demo;
      sources = [mri ? "Solo MRI (sube también un audio para un resultado más completo)"
                     : "Solo audio (sube también una MRI para un resultado más completo)"];
    }
    renderResult(label, probs, demo, sources);
    showStatus("");
  } catch (err) {
    showStatus(err.message || "Ocurrió un error al analizar.", true);
  } finally {
    refreshButtons();
  }
}

/* ---------- Render ---------- */
function renderResult(label, probs, demo, sources) {
  const risk = RISK[label] || RISK.Normal;
  const badge = $("risk-badge");
  badge.className = "risk " + risk.css;
  badge.textContent = "Riesgo " + risk.text;
  $("risk-detail").textContent = risk.detail;
  $("demo-badge").hidden = !demo;

  const bars = $("bars");
  bars.replaceChildren();
  LABELS.forEach((name, i) => {
    const pct = Math.round(probs[i] * 1000) / 10;
    const row = document.createElement("div");
    row.className = "bar-row";
    const n = document.createElement("span"); n.textContent = name;
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("span"); fill.style.width = pct + "%";
    bar.appendChild(fill);
    const p = document.createElement("span"); p.textContent = pct.toFixed(1) + "%";
    row.append(n, bar, p);
    bars.appendChild(row);
  });
  $("sources").textContent = sources.join(" · ");
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- Utilidades ---------- */
function argmax(arr) { return arr.indexOf(Math.max(...arr)); }

function fakeProbs() {
  const r = [Math.random(), Math.random(), Math.random()];
  const s = r.reduce((a, b) => a + b, 0);
  return r.map((v) => v / s);
}

function showStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
}

function refreshButtons() {
  $("analyze-btn").disabled = !(state.mriFile || state.audioFile);
}

function resetAll() {
  state.mriFile = null; state.audioFile = null;
  $("mri-input").value = ""; $("audio-input").value = "";
  $("mri-preview").hidden = true; $("audio-preview").hidden = true;
  $("mri-name").textContent = ""; $("audio-name").textContent = "";
  $("rec-timer").textContent = "00:00";
  $("result").hidden = true;
  showStatus("");
  refreshButtons();
}

/* ---------- Init ---------- */
setupDropzone("mri-drop", "mri-input", setMri);
setupDropzone("audio-drop", "audio-input", setAudio);
$("rec-btn").addEventListener("click", toggleRecording);
$("analyze-btn").addEventListener("click", analyze);
$("reset-btn").addEventListener("click", resetAll);
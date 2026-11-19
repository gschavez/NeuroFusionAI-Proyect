document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');
    const errorBox = document.getElementById('errorBox');
    const searchInput = document.getElementById('patientSearchInput');
    const patientSelect = document.getElementById('patientSelect');
    const mriInput = document.getElementById('mriFile');
    const audioInput = document.getElementById('audioFile');

    let mriPreviewUrl = null;     // object URL of the selected MRI image
    let recordedWav = null;       // File (WAV 16 kHz mono) recorded in the browser
    let recordedUrl = null;

    // ------------------------------------------------------------------
    // Patient search
    // ------------------------------------------------------------------
    if (searchInput && patientSelect) {
        searchInput.addEventListener('input', (e) => {
            const filter = e.target.value.toLowerCase();
            for (let i = 1; i < patientSelect.options.length; i++) {
                const txt = patientSelect.options[i].text.toLowerCase();
                patientSelect.options[i].style.display = txt.includes(filter) ? '' : 'none';
            }
        });
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                for (let i = 1; i < patientSelect.options.length; i++) {
                    if (patientSelect.options[i].style.display !== 'none') {
                        patientSelect.selectedIndex = i;
                        break;
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------------
    // MRI preview (the same image is shown again next to the prediction)
    // ------------------------------------------------------------------
    mriInput.addEventListener('change', () => {
        const file = mriInput.files[0];
        if (mriPreviewUrl) URL.revokeObjectURL(mriPreviewUrl);
        mriPreviewUrl = null;
        const wrap = document.getElementById('mriPreviewWrap');
        if (!file) { wrap.classList.add('d-none'); return; }
        mriPreviewUrl = URL.createObjectURL(file);
        document.getElementById('mriPreview').src = mriPreviewUrl;
        document.getElementById('mriPreviewName').textContent = file.name;
        wrap.classList.remove('d-none');
    });

    // ------------------------------------------------------------------
    // Voice recording in the browser -> WAV 16 kHz mono (what the model expects)
    // ------------------------------------------------------------------
    const MAX_SECONDS = 120;
    const SHORT_SECONDS = 30;
    const recStartBtn = document.getElementById('recStartBtn');
    const recStopBtn = document.getElementById('recStopBtn');
    const recStatus = document.getElementById('recStatus');
    const recTimer = document.getElementById('recTimer');
    const recResult = document.getElementById('recResult');
    const recPlayer = document.getElementById('recPlayer');
    const recInfo = document.getElementById('recInfo');
    let mediaRecorder = null, chunks = [], stream = null, timerId = null, startedAt = 0;

    const canRecord = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
    if (!canRecord) {
        document.getElementById('recUnsupported').classList.remove('d-none');
        recStartBtn.disabled = true;
    }

    function pickMimeType() {
        const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
        return types.find(t => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) || '';
    }

    function fmt(sec) {
        const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
        return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
    }

    recStartBtn.addEventListener('click', async () => {
        hideError();
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
        } catch (err) {
            showError('Microphone access was denied or is not available: ' + err.message);
            return;
        }
        chunks = [];
        const mimeType = pickMimeType();
        mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
        mediaRecorder.onstop = onRecordingStopped;
        mediaRecorder.start(250);
        startedAt = Date.now();
        recTimer.textContent = '00:00';
        recStatus.classList.remove('d-none');
        recStartBtn.classList.add('d-none');
        recStopBtn.classList.remove('d-none');
        recResult.classList.add('d-none');
        timerId = setInterval(() => {
            const sec = (Date.now() - startedAt) / 1000;
            recTimer.textContent = fmt(sec);
            if (sec >= MAX_SECONDS) stopRecording();
        }, 250);
    });

    recStopBtn.addEventListener('click', stopRecording);

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
        if (stream) stream.getTracks().forEach(t => t.stop());
        clearInterval(timerId);
        recStatus.classList.add('d-none');
        recStopBtn.classList.add('d-none');
        recStartBtn.classList.remove('d-none');
        recStartBtn.textContent = '● Record again';
    }

    async function onRecordingStopped() {
        try {
            const blob = new Blob(chunks, { type: (mediaRecorder && mediaRecorder.mimeType) || 'audio/webm' });
            recInfo.textContent = 'Converting recording…';
            const { wavBlob, seconds } = await toWav16kMono(blob);
            if (seconds < 1) throw new Error('The recording is empty.');
            recordedWav = new File([wavBlob], 'browser_recording.wav', { type: 'audio/wav' });
            if (recordedUrl) URL.revokeObjectURL(recordedUrl);
            recordedUrl = URL.createObjectURL(wavBlob);
            recPlayer.src = recordedUrl;
            recInfo.textContent = `Recorded ${fmt(seconds)} · WAV 16 kHz mono · ${(wavBlob.size / 1024).toFixed(0)} KB — this recording will be used.`;
            document.getElementById('recShortWarn').classList.toggle('d-none', seconds >= SHORT_SECONDS);
            recResult.classList.remove('d-none');
        } catch (err) {
            recordedWav = null;
            showError('Could not process the recording: ' + err.message);
        }
    }

    document.getElementById('recDiscardBtn').addEventListener('click', () => {
        recordedWav = null;
        if (recordedUrl) URL.revokeObjectURL(recordedUrl);
        recordedUrl = null;
        recPlayer.removeAttribute('src');
        recResult.classList.add('d-none');
        recStartBtn.textContent = '● Start recording';
    });

    async function toWav16kMono(blob) {
        const AC = window.AudioContext || window.webkitAudioContext;
        const ctx = new AC();
        const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
        ctx.close();
        const rate = 16000;
        const length = Math.max(1, Math.ceil(decoded.duration * rate));
        const offline = new OfflineAudioContext(1, length, rate);   // 1 channel = downmix to mono
        const src = offline.createBufferSource();
        src.buffer = decoded;
        src.connect(offline.destination);
        src.start();
        const rendered = await offline.startRendering();
        return { wavBlob: encodeWav(rendered.getChannelData(0), rate), seconds: decoded.duration };
    }

    function encodeWav(samples, rate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
        writeStr(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); writeStr(8, 'WAVE');
        writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
        view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
        writeStr(36, 'data'); view.setUint32(40, samples.length * 2, true);
        let off = 44;
        for (let i = 0; i < samples.length; i++, off += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return new Blob([view], { type: 'audio/wav' });
    }

    // Which voice sample will be sent: the active tab decides
    function selectedAudio() {
        const recordTabActive = document.getElementById('tab-record').classList.contains('active');
        if (recordTabActive) return recordedWav ? { file: recordedWav, source: 'recorded in browser' } : null;
        const f = audioInput.files[0];
        return f ? { file: f, source: 'uploaded file' } : null;
    }

    // ------------------------------------------------------------------
    // Submit: MRI + audio in parallel, then fusion
    // ------------------------------------------------------------------
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();
        const mriFile = mriInput.files[0];
        const audio = selectedAudio();
        const patientId = patientSelect ? patientSelect.value : '';

        if (!patientId) { showError('Please search and select a patient from the list first.'); return; }
        if (!mriFile) { showError('Please select an MRI image.'); return; }
        if (!audio) { showError('Please upload a voice file or record one in the “Record voice” tab.'); return; }

        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing models… (the first analysis can take up to a minute)';

        try {
            const mriData = new FormData(); mriData.append('file', mriFile);
            const audioData = new FormData(); audioData.append('file', audio.file);

            const [mriRes, audioRes] = await Promise.all([
                fetch('/predict_mri', { method: 'POST', body: mriData }),
                fetch('/predict_audio', { method: 'POST', body: audioData })
            ]);
            const mriJson = await readJson(mriRes);
            const audioJson = await readJson(audioRes);
            if (!mriRes.ok) throw new Error('MRI: ' + (mriJson.error || mriRes.status));
            if (!audioRes.ok) throw new Error('Audio: ' + (audioJson.error || audioRes.status));

            const fusionRes = await fetch('/predict_fusion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mri_probabilities: mriJson.probabilities,
                    audio_probabilities: audioJson.probabilities,
                    mri_filename: mriFile.name,
                    audio_filename: audio.file.name,
                    patient_id: patientId,
                    audio_quality: audioJson.quality || null
                })
            });
            const fusionJson = await readJson(fusionRes);
            if (!fusionRes.ok) throw new Error('Fusion: ' + (fusionJson.error || fusionRes.status));

            renderResults(mriJson, audioJson, fusionJson, mriFile, audio);
        } catch (error) {
            showError('Analysis error — ' + error.message);
            console.error(error);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Execute Multimodal Fusion';
        }
    });

    async function readJson(res) {
        try { return await res.json(); }
        catch { return { error: res.status === 401 ? 'Your session expired. Please log in again.' : `Server error (${res.status})` }; }
    }

    function renderResults(mriJson, audioJson, fusionJson, mriFile, audio) {
        resultsSection.classList.remove('d-none');
        const rep = fusionJson.report || {};
        const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';

        // Header
        const opt = patientSelect.options[patientSelect.selectedIndex];
        document.getElementById('repPatient').textContent = opt ? opt.text : '—';
        const now = new Date();
        document.getElementById('repDate').textContent = now.toLocaleString();
        document.getElementById('repId').textContent = 'NF-' + now.getTime().toString(36).toUpperCase();

        // 1. Result + summary
        const final = document.getElementById('finalDiagnosis');
        final.textContent = fusionJson.prediction;
        final.className = 'display-3 mb-1 result-' + fusionJson.prediction;
        const probs = fusionJson.probabilities;
        document.getElementById('finalConfidence').textContent = `Global confidence: ${pct(Math.max(...Object.values(probs)), 2)}`;
        const badge = document.getElementById('bandBadge');
        badge.textContent = (rep.confidence_band || '—') + ' confidence';
        badge.className = 'badge rounded-0 px-3 py-2 band-' + (rep.confidence_band || 'Low');
        document.getElementById('bandText').textContent = rep.confidence_text || '';
        document.getElementById('secondText').textContent = rep.second_class
            ? `Second most likely class: ${rep.second_class} (${pct(rep.second_probability)}).` : '';
        document.getElementById('fNormal').textContent = pct(probs.Normal || 0);
        document.getElementById('fMild').textContent = pct(probs.Mild || 0);
        document.getElementById('fModerate').textContent = pct(probs.Moderate || 0);
        const c = fusionJson.comment || {};
        document.getElementById('commentBox').className = 'comment-box mb-3 level-' + fusionJson.prediction;
        document.getElementById('commentSummary').textContent = c.summary || '';

        // 2a. MRI evidence
        document.getElementById('mriResultImg').src = mriPreviewUrl || URL.createObjectURL(mriFile);
        document.getElementById('mriClassLabel').textContent = `· ${mriJson.class_name} (${pct(mriJson.confidence)})`;
        document.getElementById('mriFinding').textContent = (rep.mri && rep.mri.finding) || '';
        updateBar('mriNormal', mriJson.probabilities[0]);
        updateBar('mriMild', mriJson.probabilities[1]);
        updateBar('mriModerate', mriJson.probabilities[2]);

        // 2b. Voice evidence
        document.getElementById('audioSourceLabel').textContent = '· ' + audio.source;
        document.getElementById('audioResultPlayer').src = (audio.file === recordedWav && recordedUrl) ? recordedUrl : URL.createObjectURL(audio.file);
        const v = rep.voice || {};
        const ap = audioJson.probabilities;
        document.getElementById('voiceSummary').textContent =
            `${v.label || ''}: P(Alzheimer's) = ${pct(v.p_alzheimer != null ? v.p_alzheimer : ap[1] + ap[2])} ` +
            `(shown to the fusion model as Normal ${pct(ap[0])} · Mild ${pct(ap[1])} · Moderate ${pct(ap[2])}).`;
        const body = document.getElementById('markersBody');
        body.innerHTML = '';
        const effText = { impairment: '▲ towards impairment', healthy: '▼ towards healthy', neutral: '— little effect' };
        (audioJson.markers || []).forEach(m => {
            const tr = document.createElement('tr');
            const td = (txt, cls) => { const el = document.createElement('td'); el.textContent = txt; if (cls) el.className = cls; return el; };
            const name = document.createElement('td');
            name.innerHTML = '';
            const strong = document.createElement('strong'); strong.textContent = m.label;
            const small = document.createElement('div'); small.className = 'text-muted'; small.textContent = m.meaning;
            name.appendChild(strong); name.appendChild(small);
            tr.appendChild(name);
            tr.appendChild(td(m.value == null ? '—' : String(m.value), 'text-end mono-number'));
            tr.appendChild(td(m.position ? `${m.position} average (z = ${m.z})` : 'combined'));
            tr.appendChild(td(effText[m.direction] || m.direction, 'eff-' + m.direction));
            body.appendChild(tr);
        });
        if (!(audioJson.markers || []).length) {
            const tr = document.createElement('tr');
            const tdEl = document.createElement('td'); tdEl.colSpan = 4; tdEl.className = 'text-muted';
            tdEl.textContent = 'Acoustic markers are not available for this sample.';
            tr.appendChild(tdEl); body.appendChild(tr);
        }

        // 3. Reliability, 4. Next steps, 5. Limitations
        document.getElementById('agreeText').innerHTML = '';
        const agree = document.createElement('span');
        agree.textContent = rep.modalities_agree
            ? `✔ MRI (${rep.mri ? rep.mri.label : mriJson.class_name}) and voice (${v.label || ''}) point in the same direction.`
            : `✖ MRI (${rep.mri ? rep.mri.label : mriJson.class_name}) and voice (${v.label || ''}) point in different directions.`;
        agree.className = rep.modalities_agree ? 'eff-healthy' : 'eff-impairment';
        document.getElementById('agreeText').appendChild(agree);
        fillList('reliabilityList', rep.reliability);
        fillList('nextSteps', rep.next_steps);
        fillList('limitationsList', rep.limitations);
        document.getElementById('commentDisclaimer').textContent = c.disclaimer || fusionJson.notice || '';

        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function fillList(id, items) {
        const ul = document.getElementById(id);
        ul.innerHTML = '';
        (items || []).forEach(t => { const li = document.createElement('li'); li.textContent = t; ul.appendChild(li); });
    }

    document.getElementById('printBtn').addEventListener('click', () => window.print());

    function updateBar(prefix, value) {
        const percentage = (value * 100).toFixed(2);
        document.getElementById(`${prefix}Val`).textContent = `${percentage}%`;
        document.getElementById(`${prefix}Bar`).style.width = `${percentage}%`;
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.classList.remove('d-none');
        errorBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    function hideError() { errorBox.classList.add('d-none'); }
});
document.addEventListener('DOMContentLoaded', () => {
    console.log("🔥 DASHBOARD_V2.JS CARGADO Y LISTO 🔥");

    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const mriFile = document.getElementById('mriFile').files[0];
        const audioFile = document.getElementById('audioFile').files[0];

        if (!mriFile || !audioFile) {
            alert("Please select both an MRI and an Audio file.");
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = "Processing models... (This may take a few seconds)";

        // Función para leer respuestas de forma segura
        async function safeJson(response) {
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                return await response.json();
            }
            if (response.status === 413) {
                throw new Error("El archivo de audio es demasiado grande para el servidor. Intenta con uno más ligero.");
            }
            throw new Error(`Error del servidor (Código ${response.status}).`);
        }

        try {
            // 1. Preparar datos
            const mriData = new FormData();
            mriData.append('file', mriFile);
            
            const audioData = new FormData();
            audioData.append('file', audioFile);

            // 2. Enviar peticiones al backend
            const [mriRes, audioRes] = await Promise.all([
                fetch('/predict_mri', { method: 'POST', body: mriData }),
                fetch('/predict_audio', { method: 'POST', body: audioData })
            ]);

            const mriJson = await safeJson(mriRes);
            const audioJson = await safeJson(audioRes);

            if (!mriRes.ok) throw new Error(mriJson.error || "MRI Error");
            if (!audioRes.ok) throw new Error(audioJson.error || "Audio Error");

            // 3. Llamar a la Fusión
            const fusionData = {
                mri_probabilities: mriJson.probabilities,
                audio_probabilities: audioJson.probabilities,
                mri_filename: mriFile.name,
                audio_filename: audioFile.name
            };

            const fusionRes = await fetch('/predict_fusion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(fusionData)
            });

            const fusionJson = await safeJson(fusionRes);
            if (!fusionRes.ok) throw new Error(fusionJson.error || "Fusion Error");

            // 4. Mostrar resultados
            resultsSection.classList.remove('d-none');
            resultsSection.scrollIntoView({ behavior: 'smooth' });

            document.getElementById('finalDiagnosis').textContent = fusionJson.prediction;
            const finalProb = Math.max(...Object.values(fusionJson.probabilities));
            document.getElementById('finalConfidence').textContent = `Global confidence: ${(finalProb * 100).toFixed(2)}%`;

            updateBar('mriNormal', mriJson.probabilities[0]);
            updateBar('mriMild', mriJson.probabilities[1]);
            updateBar('mriModerate', mriJson.probabilities[2]);

            updateBar('audioNormal', audioJson.probabilities[0]);
            updateBar('audioMild', audioJson.probabilities[1]);
            updateBar('audioModerate', audioJson.probabilities[2]);

        } catch (error) {
            alert("Analysis error: " + error.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Execute Multimodal Fusion";
        }
    });

    function updateBar(prefix, value) {
        const percentage = (value * 100).toFixed(2);
        document.getElementById(`${prefix}Val`).textContent = `${percentage}%`;
        document.getElementById(`${prefix}Bar`).style.width = `${percentage}%`;
    }
});
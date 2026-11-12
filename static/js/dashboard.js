document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // 1. Get files
        const mriFile = document.getElementById('mriFile').files[0];
        const audioFile = document.getElementById('audioFile').files[0];

        if (!mriFile || !audioFile) {
            alert("Please select both files.");
            return;
        }

        // UI feedback
        submitBtn.disabled = true;
        submitBtn.textContent = "Processing models... (This may take a few seconds)";

        try {
            // 2. Prepare FormData for MRI
            const mriData = new FormData();
            mriData.append('file', mriFile);

            // 3. Prepare FormData for Audio
            const audioData = new FormData();
            audioData.append('file', audioFile);

            // 4. Send requests in parallel
            const [mriRes, audioRes] = await Promise.all([
                fetch('/predict_mri', { method: 'POST', body: mriData }),
                fetch('/predict_audio', { method: 'POST', body: audioData })
            ]);

            const mriJson = await mriRes.json();
            const audioJson = await audioRes.json();

            if (!mriRes.ok) throw new Error(mriJson.error || "MRI Error");
            if (!audioRes.ok) throw new Error(audioJson.error || "Audio Error");

            // 5. Send to Fusion (with obtained probabilities and filenames)
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

            const fusionJson = await fusionRes.json();
            if (!fusionRes.ok) throw new Error(fusionJson.error || "Fusion Error");

            // 6. Update UI with results
            displayResults(mriJson.probabilities, audioJson.probabilities, fusionJson);

        } catch (error) {
            alert("Analysis error: " + error.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Execute Multimodal Fusion";
        }
    });

    function displayResults(mriProbs, audioProbs, fusionJson) {
        resultsSection.classList.remove('d-none');
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // --- Final Result ---
        document.getElementById('finalDiagnosis').textContent = fusionJson.prediction;
        const finalProb = Math.max(...Object.values(fusionJson.probabilities));
        document.getElementById('finalConfidence').textContent = `Global confidence: ${(finalProb * 100).toFixed(2)}%`;

        // --- Update MRI bars ---
        updateBar('mriNormal', mriProbs[0]);
        updateBar('mriMild', mriProbs[1]);
        updateBar('mriModerate', mriProbs[2]);

        // --- Update Audio bars ---
        updateBar('audioNormal', audioProbs[0]);
        updateBar('audioMild', audioProbs[1]);
        updateBar('audioModerate', audioProbs[2]);
    }

    function updateBar(prefix, value) {
        const percentage = (value * 100).toFixed(2);
        document.getElementById(`${prefix}Val`).textContent = `${percentage}%`;
        document.getElementById(`${prefix}Bar`).style.width = `${percentage}%`;
    }
});
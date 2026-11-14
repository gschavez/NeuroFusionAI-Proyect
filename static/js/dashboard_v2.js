document.addEventListener('DOMContentLoaded', () => {
    console.log("🔥 DASHBOARD_V2.JS CARGADO Y LISTO 🔥");

    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');
    
    // Search and dropdown elements
    const searchInput = document.getElementById('patientSearchInput');
    const patientSelect = document.getElementById('patientSelect');

    if (searchInput && patientSelect) {
        // 1. Filter options as the user types
        searchInput.addEventListener('input', function(e) {
            const filter = e.target.value.toLowerCase();
            const options = patientSelect.options;
            
            for (let i = 0; i < options.length; i++) {
                const txt = options[i].text.toLowerCase();
                if (i === 0) continue; // Keep the default option visible
                
                if (txt.includes(filter)) {
                    options[i].style.display = "";
                } else {
                    options[i].style.display = "none";
                }
            }
        });

        // 2. Auto-select first match when pressing Enter
        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault(); // Prevent form submission
                
                // Find the first visible option (skipping the default one)
                for (let i = 1; i < patientSelect.options.length; i++) {
                    if (patientSelect.options[i].style.display !== 'none') {
                        patientSelect.selectedIndex = i;
                        break;
                    }
                }
            }
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const mriFile = document.getElementById('mriFile').files[0];
        const audioFile = document.getElementById('audioFile').files[0];
        
        const patientId = patientSelect.value; 

        // Validation in English
        if (!patientId || patientId === "") {
            alert("Please search and select a patient from the list first.");
            return;
        }

        if (!mriFile || !audioFile) {
            alert("Please select both an MRI and an Audio file.");
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = "Processing models... (This may take a few seconds)";

        try {
            // 1. Prepare data
            const mriData = new FormData();
            mriData.append('file', mriFile);
            
            const audioData = new FormData();
            audioData.append('file', audioFile);

            // 2. Send requests to the backend in parallel
            const [mriRes, audioRes] = await Promise.all([
                fetch('/predict_mri', { method: 'POST', body: mriData }),
                fetch('/predict_audio', { method: 'POST', body: audioData })
            ]);

            const mriJson = await mriRes.json();
            const audioJson = await audioRes.json();

            if (!mriRes.ok) throw new Error(mriJson.error || "MRI Error");
            if (!audioRes.ok) throw new Error(audioJson.error || "Audio Error");

            // 3. Call Fusion with obtained probabilities
            const fusionData = {
                mri_probabilities: mriJson.probabilities,
                audio_probabilities: audioJson.probabilities,
                mri_filename: mriFile.name,
                audio_filename: audioFile.name,
                patient_id: patientId
            };

            const fusionRes = await fetch('/predict_fusion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(fusionData)
            });

            const fusionJson = await fusionRes.json();
            if (!fusionRes.ok) throw new Error(fusionJson.error || "Fusion Error");

            // 4. Show results on screen
            resultsSection.classList.remove('d-none');
            resultsSection.scrollIntoView({ behavior: 'smooth' });

            // Final Result (Fusion)
            document.getElementById('finalDiagnosis').textContent = fusionJson.prediction;
            const finalProb = Math.max(...Object.values(fusionJson.probabilities));
            document.getElementById('finalConfidence').textContent = `Global confidence: ${(finalProb * 100).toFixed(2)}%`;

            // Update UI Bars
            updateBar('mriNormal', mriJson.probabilities[0]);
            updateBar('mriMild', mriJson.probabilities[1]);
            updateBar('mriModerate', mriJson.probabilities[2]);

            updateBar('audioNormal', audioJson.probabilities[0]);
            updateBar('audioMild', audioJson.probabilities[1]);
            updateBar('audioModerate', audioJson.probabilities[2]);

        } catch (error) {
            alert("Analysis error: " + error.message);
            console.error("Detailed error:", error);
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
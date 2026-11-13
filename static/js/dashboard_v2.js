document.addEventListener('DOMContentLoaded', () => {
    console.log("🔥 EL NUEVO DASHBOARD_V2.JS ESTÁ CORRIENDO 🔥");

    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const mriFile = document.getElementById('mriFile').files[0];
        const audioFile = document.getElementById('audioFile').files[0];

        if (!mriFile) {
            alert("Please select an MRI file.");
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = "Processing models...";

        try {
            const mriData = new FormData();
            mriData.append('file', mriFile);

            const mriRes = await fetch('/predict_mri', { method: 'POST', body: mriData });
            const mriJson = await mriRes.json();
            
            if (!mriRes.ok) throw new Error(mriJson.error || "MRI Error");

            // Mostrar resultados del MRI (esto siempre debe pasar)
            resultsSection.classList.remove('d-none');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
            document.getElementById('finalDiagnosis').textContent = mriJson.class_name;
            document.getElementById('finalConfidence').textContent = `Global confidence: ${(mriJson.confidence * 100).toFixed(2)}%`;

            updateBar('mriNormal', mriJson.probabilities[0]);
            updateBar('mriMild', mriJson.probabilities[1]);
            updateBar('mriModerate', mriJson.probabilities[2]);

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
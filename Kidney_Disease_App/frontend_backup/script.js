// Configuration
const API_URL = 'http://localhost:5000';

// Handle form submission
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(document.getElementById('predictionForm'));
    const data = Object.fromEntries(formData);
    
    // Convert numeric fields to numbers
    const numericFields = [
        'serum_creatinine', 'gfr', 'bun', 'serum_calcium', 'ana', 'c3_c4',
        'hematuria', 'oxalate_levels', 'urine_ph', 'blood_pressure', 'water_intake', 'months'
    ];
    
    numericFields.forEach(field => {
        if (data[field]) {
            data[field] = parseFloat(data[field]);
        }
    });
    
    console.log('Sending prediction data:', data);
    
    try {
        // Show loading state
        const submitBtn = document.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="loading"></span> Processing...';
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Restore button
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        
        if (result.status === 'success') {
            handlePredictionResult(result);
        } else {
            alert('Error: ' + result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        
        // Restore button
        const submitBtn = document.querySelector('button[type="submit"]');
        submitBtn.disabled = false;
        submitBtn.textContent = originalText || '🔍 Get Prediction';
        
        alert('❌ Error connecting to server.\nMake sure:\n1. Backend is running (python app.py)\n2. Model is trained (python model_training.py)');
    }
});

// Handle prediction result
function handlePredictionResult(result) {
    const prediction = result.prediction;
    const confidence = result.confidence;
    
    console.log(`Prediction: ${prediction}, Confidence: ${confidence}%`);
    
    // Redirect to appropriate page
    if (prediction === 'No CKD') {
        // Person is HEALTHY
        window.location.href = `healthy.html?confidence=${confidence}`;
    } else {
        // Person has CKD - UNHEALTHY
        window.location.href = `unhealthy.html?confidence=${confidence}`;
    }
}

// Show page when loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('✓ Prediction app loaded and ready!');
    console.log('ℹ️ Make sure backend is running on http://localhost:5000');
});

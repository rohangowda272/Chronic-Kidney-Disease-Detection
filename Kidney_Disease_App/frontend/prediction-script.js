// Configuration
const API_URL = 'http://localhost:5000';

// Authentication check and page initialization
document.addEventListener('DOMContentLoaded', () => {
    const loggedInUser = JSON.parse(sessionStorage.getItem('loggedInUser'));
    if (!loggedInUser) {
        window.location.href = 'index.html';
        return;
    }
    
    // Set user greeting
    const userGreeting = document.getElementById('userGreeting');
    if (userGreeting) {
        userGreeting.textContent = `Welcome, ${loggedInUser.fullName}!`;
    }
    
    console.log('✓ Prediction app loaded for:', loggedInUser.fullName);
    
    // Attach form submission handler
    attachFormHandler();
});

// Logout function
function handleLogout() {
    if (confirm('Are you sure you want to logout?')) {
        sessionStorage.removeItem('loggedInUser');
        localStorage.removeItem('rememberMe');
        localStorage.removeItem('userEmail');
        window.location.href = 'index.html';
    }
}

// Attach form submission handler
function attachFormHandler() {
    const predictionForm = document.getElementById('predictionForm');
    if (!predictionForm) {
        console.error('❌ Prediction form not found!');
        return;
    }
    
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleFormSubmit();
    });
    
    console.log('✓ Form handler attached successfully');
}

// Handle form submission
async function handleFormSubmit() {
    try {
        const formData = new FormData(document.getElementById('predictionForm'));
        const data = Object.fromEntries(formData);
        
        // Remove empty fields
        const cleanData = {};
        Object.keys(data).forEach(key => {
            if (data[key] !== '' && data[key] !== null && data[key] !== undefined) {
                cleanData[key] = data[key];
            }
        });
        
        // ====================================================================
        // NUMERIC FIELDS (11) - for kidney_disease_csv.csv
        // ====================================================================
        const numericFields = [
            'age',      // Age (years)
            'bp',       // Blood Pressure (mm/Hg)
            'sg',       // Specific Gravity
            'al',       // Albumin (0-5)
            'su',       // Sugar (0-5)
            'bgr',      // Blood Glucose Random (mgs/dl)
            'bu',       // Blood Urea (mgs/dl)
            'sc',       // Serum Creatinine (mgs/dl)
            'sod',      // Sodium (mEq/L)
            'pot',      // Potassium (mEq/L)
            'hemo'      // Hemoglobin (gms)
        ];
        
        // ====================================================================
        // CATEGORICAL FIELDS (13) - for kidney_disease_csv.csv
        // ====================================================================
        const categoricalFields = [
            'rbc',      // Red Blood Cells: normal/abnormal
            'pc',       // Pus Cell: normal/abnormal
            'pcc',      // Pus Cell Clumps: present/notpresent
            'ba',       // Bacteria: present/notpresent
            'pcv',      // Packed Cell Volume (numeric)
            'wc',       // White Blood Cell Count (cells/cumm)
            'rc',       // Red Blood Cell Count (millions/cmm)
            'htn',      // Hypertension: yes/no
            'dm',       // Diabetes Mellitus: yes/no
            'cad',      // Coronary Artery Disease: yes/no
            'appet',    // Appetite: good/poor
            'pe',       // Pedal Edema: yes/no
            'ane'       // Anemia: yes/no
        ];
        
        // Convert numeric fields to numbers
        numericFields.forEach(field => {
            if (cleanData[field]) {
                cleanData[field] = parseFloat(cleanData[field]);
            }
        });
        
        // Ensure categorical fields are lowercase strings (except pcv, wc, rc which are numeric)
        categoricalFields.forEach(field => {
            if (cleanData[field]) {
                // pcv, wc, rc are numeric in the dataset
                if (field === 'pcv' || field === 'wc' || field === 'rc') {
                    cleanData[field] = parseFloat(cleanData[field]);
                } else {
                    cleanData[field] = String(cleanData[field]).toLowerCase().trim();
                }
            }
        });
        
        console.log('📊 Sending prediction data:', cleanData);
        
        // Check if any data is provided
        if (Object.keys(cleanData).length === 0) {
            alert('⚠️ Please fill in at least one health parameter');
            return;
        }
        
        // Show loading state
        const submitBtn = document.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '⏳ Processing...';
        
        // ====================================================================
        // SEND TO CORRECT API ENDPOINT
        // ====================================================================
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(cleanData),
        });
        
        // Restore button immediately
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        
        // Check if response is ok
        if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        
        // Parse JSON response
        let result;
        try {
            result = await response.json();
        } catch (parseError) {
            console.error('Invalid JSON response:', parseError);
            throw new Error('Invalid server response format');
        }
        
        // Validate response structure
        if (!result || typeof result !== 'object') {
            throw new Error('Unexpected response format');
        }
        
        // Handle prediction result
        if (result.status === 'success' && result.prediction) {
            console.log(`✓ Prediction successful: ${result.prediction}, Confidence: ${result.confidence}%`);
            handlePredictionResult(result);
        } else if (result.error || result.message) {
            throw new Error(result.error || result.message);
        } else {
            throw new Error('Invalid prediction response');
        }
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        
        // Restore button
        const submitBtn = document.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = '🔍 Analyze Health Data';
        }
        
        // Provide detailed error message
        if (error.message.includes('Failed to fetch')) {
            alert('❌ Cannot connect to backend server.\n\nPlease ensure:\n1. Backend is running (python app.py in backend folder)\n2. Server is on http://localhost:5000\n3. No firewall is blocking the connection');
        } else if (error.message.includes('timeout')) {
            alert('❌ Request timed out.\n\nPlease ensure:\n1. Model is fully trained (python model_training.py)\n2. Backend is running properly\n3. Try again in a moment');
        } else {
            alert(`❌ Error: ${error.message}`);
        }
    }
}

// Handle prediction result
function handlePredictionResult(result) {
    const prediction = result.prediction.toLowerCase().trim();
    const confidence = (result.confidence || 0).toFixed(2);
    
    console.log(`Redirecting to result page - Prediction: ${prediction}, Confidence: ${confidence}%`);
    
    // ====================================================================
    // HANDLE 'ckd' AND 'notckd' PREDICTIONS
    // ====================================================================
    if (prediction === 'notckd' || prediction === 'no ckd' || prediction.includes('not')) {
        // No kidney disease detected
        console.log('✓ Redirecting to healthy.html');
        window.location.href = `healthy.html?confidence=${confidence}`;
    } else if (prediction === 'ckd' || prediction.includes('ckd')) {
        // Kidney disease detected
        console.log('✓ Redirecting to unhealthy.html');
        window.location.href = `unhealthy.html?confidence=${confidence}`;
    } else {
        console.warn('⚠️ Unknown prediction value:', prediction);
        alert(`⚠️ Unknown prediction result: "${prediction}"\n\nPlease check the backend response.`);
    }
}

// Show page when fully loaded
console.log('✓ Prediction script loaded and ready!');
console.log('ℹ️ API URL: ' + API_URL);
console.log('ℹ️ Dataset: kidney_disease_csv.csv');
console.log('ℹ️ Features: 24 (11 numeric + 13 categorical)');
console.log('ℹ️ Make sure backend is running: cd backend && python app.py');
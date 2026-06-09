from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

print("\n🔄 Loading trained model...")
try:
    model = joblib.load('trained_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    global_scaler = joblib.load('global_scaler.pkl')
    pca = joblib.load('pca.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("✓ Model loaded successfully!\n")
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print("   Run 'python model_training.py' first to generate model files.")
    exit()

# Load dataset to get feature columns
data = pd.read_csv('kidney_disease_csv.csv')

# Remove 'id' and 'classification' columns
feature_cols = [col for col in data.columns if col not in ['id', 'classification']]

print(f"✓ Expected features ({len(feature_cols)}):")
print(f"  Numeric (11): age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo")
print(f"  Categorical (13): rbc, pc, pcc, ba, pcv, wc, rc, htn, dm, cad, appet, pe, ane\n")

# ============================================================================
# SERVE FRONTEND FILES
# ============================================================================

@app.route('/')
def serve_index():
    """Serve the main index.html from frontend folder"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_frontend_files(path):
    """Serve all frontend files (HTML, CSS, JS)"""
    return send_from_directory('../frontend', path)

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/status')
def api_status():
    """API health check"""
    return jsonify({
        'status': 'running',
        'message': 'Kidney Disease Prediction API is operational',
        'features': len(feature_cols),
        'dataset': 'kidney_disease_csv.csv'
    })

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Main prediction endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        user_data = request.get_json()
        
        if user_data is None:
            return jsonify({'status': 'error', 'message': 'No data received'}), 400
        
        print(f"\n📊 Received data: {user_data}")
        
        # Create DataFrame with all feature columns
        input_df = pd.DataFrame([user_data])
        
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = None
        
        # Reorder columns to match training data
        input_df = input_df[feature_cols]
        
        # Define numeric and categorical columns
        numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        
        # Convert numeric columns
        for col in numeric_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                except Exception as e:
                    print(f"⚠️ Warning converting {col}: {e}")
                    input_df[col] = None
        
        # Fill missing numeric values with dataset mean
        for col in numeric_cols:
            if col in input_df.columns and input_df[col].isna().any():
                mean_value = data[col].mean()
                input_df[col] = input_df[col].fillna(mean_value)
        
        # Handle categorical columns
        for col in categorical_cols:
            if col in input_df.columns:
                # Convert to string and lowercase
                if input_df[col].notna().any():
                    input_df[col] = input_df[col].astype(str).str.lower().str.strip()
                
                # Fill missing with mode
                if input_df[col].isna().any() or input_df[col].iloc[0] == 'nan':
                    mode_value = data[col].mode()[0] if len(data[col].mode()) > 0 else 'no'
                    input_df[col] = input_df[col].fillna(mode_value)
        
        # Handle pcv, wc, rc (mixed numeric/object type in original data)
        for col in ['pcv', 'wc', 'rc']:
            if col in input_df.columns:
                try:
                    # Try to convert to numeric
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                    # Fill with mean if numeric
                    if input_df[col].isna().any():
                        mean_val = pd.to_numeric(data[col], errors='coerce').mean()
                        input_df[col] = input_df[col].fillna(mean_val)
                except:
                    # Keep as object if conversion fails
                    pass
        
        print(f"📊 Processed input shape: {input_df.shape}")
        print(f"📊 Input data:\n{input_df.to_dict('records')[0]}")
        
        # Transform through pipeline
        X_preprocessed = preprocessor.transform(input_df)
        X_scaled = global_scaler.transform(X_preprocessed)
        X_pca = pca.transform(X_scaled)
        
        # Make prediction
        prediction = model.predict(X_pca)[0]
        probability = model.predict_proba(X_pca)[0]
        
        result_class = label_encoder.classes_[prediction]
        confidence = float(probability[prediction] * 100)
        
        print(f"✅ Prediction: {result_class} (Confidence: {confidence:.2f}%)\n")
        
        return jsonify({
            'status': 'success',
            'prediction': result_class,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    print("="*70)
    print("🚀 KIDNEY DISEASE PREDICTION - FULL STACK APPLICATION")
    print("="*70)
    print("\n📊 DATASET: kidney_disease_csv.csv")
    print("   • Records: 400")
    print("   • Features: 24 (11 numeric + 13 categorical)")
    print("   • Target: ckd / notckd")
    print("\n📍 Server running on: http://localhost:5000")
    print("\n🌐 ACCESS YOUR APPLICATION:")
    print("   Main Page:       http://localhost:5000/")
    print("   Prediction Form: http://localhost:5000/prediction.html")
    print("   API Status:      http://localhost:5000/api/status")
    print("\n⚠️ IMPORTANT:")
    print("   • Backend serves frontend files automatically")
    print("   • Dataset: kidney_disease_csv.csv")
    print("   • Keep this window OPEN while using the app")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='localhost', port=5000, use_reloader=False)

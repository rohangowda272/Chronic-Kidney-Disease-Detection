import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("🧬 KIDNEY DISEASE PREDICTION - MODEL TRAINING")
print("Dataset: kidney_disease_csv.csv (400 records, 24 features)")
print("="*80)

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("\n[STEP 1] Loading dataset...")
try:
    data = pd.read_csv('kidney_disease_csv.csv')
    print(f"✓ Dataset loaded: {data.shape}")
    print(f"✓ Records: {len(data)}")
    print(f"✓ Features: {len(data.columns)-2}")  # Excluding id and classification
except FileNotFoundError:
    print("❌ ERROR: kidney_disease_csv.csv not found!")
    print("   Please ensure the file is in the same folder as this script.")
    exit()

# ============================================================================
# STEP 2: Data Cleaning
# ============================================================================
print("\n[STEP 2] Cleaning data...")

# Remove 'id' column (not a feature)
if 'id' in data.columns:
    data = data.drop('id', axis=1)
    print("✓ Removed 'id' column")

# Clean target variable - fix the 'ckd\t' issue
data['classification'] = data['classification'].str.strip()
print(f"✓ Target values after cleaning: {data['classification'].unique()}")

# Separate features (X) and target (y)
X = data.drop('classification', axis=1).copy()
y = data['classification'].copy()

print(f"✓ Data cleaned: {X.shape}")
print(f"✓ Missing values in X: {X.isnull().sum().sum()}")

# ============================================================================
# STEP 3: Identify Feature Types
# ============================================================================
print("\n[STEP 3] Identifying features...")

# Numeric features (11)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Categorical features (13)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"✓ Numeric features ({len(numeric_cols)}):")
print(f"  {numeric_cols}")
print(f"✓ Categorical features ({len(categorical_cols)}):")
print(f"  {categorical_cols}")

# ============================================================================
# STEP 4: Build Preprocessing Pipeline
# ============================================================================
print("\n[STEP 4] Building preprocessing pipeline...")

# Numeric feature pipeline: KNN Imputation + Robust Scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('robust_scaler', RobustScaler())
])

# Categorical feature pipeline: Most Frequent Imputation + One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)

print("✓ Preprocessing pipeline built")

# ============================================================================
# STEP 5: Apply Preprocessing
# ============================================================================
print("\n[STEP 5] Preprocessing data...")
X_preprocessed = preprocessor.fit_transform(X)
print(f"✓ Preprocessed shape: {X_preprocessed.shape}")

# ============================================================================
# STEP 6: Global Scaling
# ============================================================================
print("\n[STEP 6] Global scaling...")
global_scaler = StandardScaler()
X_scaled = global_scaler.fit_transform(X_preprocessed)
print(f"✓ Scaled shape: {X_scaled.shape}")

# ============================================================================
# STEP 7: PCA Dimensionality Reduction
# ============================================================================
print("\n[STEP 7] Applying PCA...")
pca = PCA(n_components=15, random_state=42)  # 15 components for this dataset
X_pca = pca.fit_transform(X_scaled)
explained_variance = sum(pca.explained_variance_ratio_) * 100
print(f"✓ PCA shape: {X_pca.shape}")
print(f"✓ Explained variance: {explained_variance:.2f}%")

# ============================================================================
# STEP 8: Encode Target Variable
# ============================================================================
print("\n[STEP 8] Encoding target...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"✓ Classes: {label_encoder.classes_}")

# Count each class
unique, counts = np.unique(y_encoded, return_counts=True)
class_distribution = dict(zip(label_encoder.classes_, counts))
print(f"✓ Class distribution: {class_distribution}")

# ============================================================================
# STEP 9: Train-Test Split
# ============================================================================
print("\n[STEP 9] Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# ============================================================================
# STEP 10: Train SVM Model
# ============================================================================
print("\n[STEP 10] Training SVM model...")
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"✓ Train Accuracy: {train_accuracy*100:.2f}%")
print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")

# ============================================================================
# STEP 11: Save Model Files
# ============================================================================
print("\n[STEP 11] Saving model files...")

joblib.dump(model, 'trained_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(global_scaler, 'global_scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✓ trained_model.pkl")
print("✓ preprocessor.pkl")
print("✓ global_scaler.pkl")
print("✓ pca.pkl")
print("✓ label_encoder.pkl")

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)
print("\n📝 NEXT STEPS:")
print("   1. Check that all .pkl files were created")
print("   2. Run 'python app.py' to start the Flask backend")
print("   3. Open browser to http://localhost:5000")
print("\n" + "="*80 + "\n")

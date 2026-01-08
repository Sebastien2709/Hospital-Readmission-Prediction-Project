import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, TargetEncoder # TargetEncoder dispo dans scikit-learn récent >1.3
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# --- 1. CHARGEMENT & NETTOYAGE ---
df = pd.read_csv('data/diabetic_data.csv')
df = df.replace('?', np.nan)
df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
df = df.sort_values('encounter_id').drop_duplicates(subset=['patient_nbr'], keep='first')
df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# --- 2. FEATURE ENGINEERING AVANCÉ ---

# A. Mapping simple
age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, 
           'dict[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
df['age_num'] = df['age'].map(age_map).fillna(70)

# B. Feature "Complexité"
df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
df['interaction_age_visits'] = df['age_num'] * df['total_visits']

# C. Préparation des colonnes pour le pipeline
# On garde les codes bruts pour le Target Encoding (plus précis que le grouping)
# Mais on réduit la cardinalité des diagnostics rares
def simplify_diag(code):
    if pd.isna(code) or 'V' in str(code) or 'E' in str(code): return 'Other'
    return str(code)[:3] # On garde les 3 premiers chiffres

for c in ['diag_1', 'diag_2', 'diag_3']:
    df[c] = df[c].apply(simplify_diag)

categorical_cols = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 
                    'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'change', 'diabetesMed']
numerical_cols = ['age_num', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                  'num_medications', 'number_outpatient', 'number_emergency', 
                  'number_inpatient', 'number_diagnoses', 'total_visits', 'interaction_age_visits']

# Conversion en string pour être sûr
for c in categorical_cols:
    df[c] = df[c].astype(str)

X = df[numerical_cols + categorical_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- 3. PIPELINE & TARGET ENCODING ---
# Le Target Encoding est risqué (data leakage), donc on l'utilise DANS le pipeline

# Pipeline Numérique
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline Catégoriel (Target Encoding ! Le secret)
# Note: Si TargetEncoder plante (vielle version sklearn), utiliser OneHotEncoder(handle_unknown='ignore')
# Mais TargetEncoder est bien meilleur pour les diag_1 (beaucoup de valeurs)
from sklearn.preprocessing import TargetEncoder 
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', TargetEncoder(smooth=5.0)) # Smooth évite l'overfitting
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

# --- 4. ENSEMBLE MODELING (VOTING) ---

# Modèle 1: Gradient Boosting (Le leader)
clf1 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=10, l2_regularization=1.0, random_state=42)

# Modèle 2: Random Forest (La robustesse)
clf2 = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)

# Modèle 3: Régression Logistique (Pour capturer les relations linéaires simples)
clf3 = LogisticRegression(solver='liblinear', C=0.1, random_state=42)

# Le Vote (Soft = moyenne des probabilités)
voting_clf = VotingClassifier(
    estimators=[('hgb', clf1), ('rf', clf2), ('lr', clf3)],
    voting='soft',
    weights=[3, 2, 1] # On fait plus confiance au Boosting (3) qu'à la régression (1)
)

# Pipeline final
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', voting_clf)])

print("Entraînement de l'Ensemble (ça peut prendre 1-2 min)...")
pipeline.fit(X_train, y_train)

# --- 5. RÉSULTATS ---
y_pred = pipeline.predict_proba(X_test)[:, 1]
final_auc = roc_auc_score(y_test, y_pred)

print(f"🎯 SCORE FINAL (Ensemble + Target Enc): {final_auc:.4f}")

# Plot
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Ensemble Model (AUC = {final_auc:.3f})', color='purple', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title('Performance Finale - Ensemble Learning')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
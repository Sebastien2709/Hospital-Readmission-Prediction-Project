import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. CHARGEMENT & NETTOYAGE ---
df = pd.read_csv('data/diabetic_data.csv')

# Gestion des valeurs manquantes et incohérentes
df = df.replace('?', np.nan)
# On exclut les patients décédés ou en hospice (Codes 11, 13, 14, 19, 20, 21)
df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

# Création de la cible : Réadmission < 30 jours
df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# --- 2. FEATURE ENGINEERING (Le secret de la performance) ---

# Groupement des codes Diagnostics (ICD-9)
def map_icd9(code):
    if pd.isna(code): return 'Other'
    if 'V' in str(code) or 'E' in str(code): return 'Other'
    try:
        n = float(code)
        if 390 <= n <= 459 or n == 785: return 'Circulatory'
        if 460 <= n <= 519 or n == 786: return 'Respiratory'
        if 250 <= n < 251: return 'Diabetes'
        # ... autres groupes ...
    except: return 'Other'
    return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[f'{col}_group'] = df[col].apply(map_icd9)

# Complexité Médicamenteuse
med_cols = ['metformin', 'insulin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone'] # etc
df['num_med_changes'] = 0
for col in med_cols:
    # On compte si le médicament a été augmenté (Up) ou diminué (Down)
    df['num_med_changes'] += df[col].apply(lambda x: 1 if x in ['Up', 'Down'] else 0)

# --- 3. PRÉPARATION POUR ML ---
age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, 
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, 
    '[80-90)': 85, '[90-100)': 95
}
df['age'] = df['age'].map(age_map)
# Sélection des colonnes et Encodage
cols_to_keep = ['race', 'gender', 'age', 'admission_type_id', 'time_in_hospital', 
                'num_lab_procedures', 'num_medications', 'number_inpatient', 
                'diag_1_group', 'num_med_changes', 'A1Cresult', 'target']

df_model = df[cols_to_keep].copy()
df_model = pd.get_dummies(df_model, drop_first=True)

# Split
X = df_model.drop('target', axis=1)
y = df_model['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- 4. MODÉLISATION (XGBoost) ---
# scale_pos_weight est CRUCIAL car les classes sont déséquilibrées (peu de réadmissions <30j)
ratio = float(np.sum(y == 0)) / np.sum(y == 1)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# --- 5. ÉVALUATION ---
preds = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC Score: {roc_auc_score(y_test, preds):.4f}")
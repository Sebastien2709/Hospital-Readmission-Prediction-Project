"""
Hospital Readmission Prediction Project
---------------------------------------
A Machine Learning pipeline to predict hospital readmissions within 30 days
for diabetic patients.

Author: [Ton Nom/Pseudo]
Dataset: UCI Diabetes 130-US Hospitals
Model: HistGradientBoostingClassifier (Optimized for tabular data with NaNs)
Performance: ~0.68 AUC-ROC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# --- CONFIGURATION ---
DATA_PATH = 'data/diabetic_data.csv' # Assure-toi que le fichier est au même endroit
MODEL_SAVE_PATH = 'readmission_model.pkl'
RANDOM_SEED = 42

def load_and_clean_data(filepath):
    """
    Charge les données et effectue le nettoyage initial critique.
    """
    print("📥 Chargement des données...")
    df = pd.read_csv(filepath)
    
    # 1. Gestion des valeurs manquantes standardisées
    df = df.replace('?', np.nan)
    
    # 2. Suppression des décès et hospices
    # On ne peut pas prédire une réadmission si le patient décède ou part en soins palliatifs.
    # IDs: 11, 13, 14, 19, 20, 21
    excl_ids = [11, 13, 14, 19, 20, 21]
    initial_len = len(df)
    df = df[~df['discharge_disposition_id'].isin(excl_ids)]
    print(f"   - Suppression des patients décédés/hospice: {initial_len - len(df)} lignes retirées.")
    
    # 3. Création de la cible (Target)
    # 1 = Réadmission < 30 jours, 0 = Sinon (NO ou >30)
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    return df

def feature_engineering(df):
    """
    Transforme les données brutes en features utilisables par le modèle.
    """
    print("⚙️ Feature Engineering en cours...")
    
    # 1. Mapping de l'âge (Intervalles -> Numérique)
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, 
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, 
        '[80-90)': 85, '[90-100)': 95
    }
    df['age_num'] = df['age'].map(age_map).fillna(70)
    
    # 2. Gestion des variables Catégorielles
    # HistGradientBoosting gère bien les catégories encodées en entiers (0, 1, 2...)
    # On utilise LabelEncoder pour tout transformer en chiffres.
    
    cat_cols = [
        'race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
        'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
    ]
    
    # Ajout des médicaments à la liste catégorielle
    meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    cat_cols.extend(meds)
    
    # On garde une trace des encodeurs si on veut inverser plus tard (optionnel)
    encoders = {} 
    
    # Identification des colonnes à "Faible Cardinalité" pour le support natif HGB
    # HGB (sklearn) accepte max 255 catégories. Au-dessus, on traite comme du numérique ordinal.
    low_card_cols_indices = []
    
    # Sélection des colonnes finales
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                'num_medications', 'number_outpatient', 'number_emergency', 
                'number_inpatient', 'number_diagnoses', 'age_num']
    
    final_cols = num_cols + cat_cols
    df_model = df[final_cols].copy()
    
    # Encodage
    for i, col in enumerate(final_cols):
        if col in cat_cols:
            df_model[col] = df_model[col].astype(str)
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            
            # Si < 255 valeurs uniques, on dit au modèle "C'est une catégorie !"
            if df_model[col].nunique() <= 255:
                low_card_cols_indices.append(i)
                
    return df_model, df['target'], low_card_cols_indices

def train_evaluate_model(X, y, cat_indices):
    """
    Entraîne le modèle HistGradientBoosting et évalue les performances.
    """
    print("🚀 Entraînement du modèle (HistGradientBoosting)...")
    
    # Split Train/Test (Stratifié pour garder la proportion de réadmissions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    # Calcul des poids pour gérer le déséquilibre (moins de classe 1 que de 0)
    sample_weights = class_weight.compute_sample_weight(
        class_weight='balanced', y=y_train
    )
    
    # Modèle : HistGradientBoosting (Rapide et gère les NaNs et Catégories)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_depth=12,
        l2_regularization=1.0,
        categorical_features=cat_indices, # Support natif
        early_stopping=True,
        random_state=RANDOM_SEED
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Prédictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    
    # --- Métriques ---
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n🎯 RÉSULTAT FINAL - AUC Score: {auc:.4f}")
    print("\nRapport de Classification :")
    print(classification_report(y_test, y_pred_class))
    
    return model, X_test, y_test, y_pred_proba

def plot_results(model, X_test, y_test, y_pred_proba):
    """
    Génère les graphiques pour l'analyse.
    """
    # 1. Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Model AUC = {roc_auc_score(y_test, y_pred_proba):.2f}', color='teal', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC - Prédiction de Réadmission')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve_final.png')
    print("📊 Graphique ROC sauvegardé sous 'roc_curve_final.png'")
    
    # 2. Matrice de Confusion
    cm = confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion')
    plt.savefig('confusion_matrix_final.png')

def main():
    # 1. Load
    df = load_and_clean_data(DATA_PATH)
    
    # 2. Prepare
    X, y, cat_indices = feature_engineering(df)
    
    # 3. Train
    model, X_test, y_test, y_probs = train_evaluate_model(X, y, cat_indices)
    
    # 4. Visualize
    plot_results(model, X_test, y_test, y_probs)
    
    # 5. Save
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"💾 Modèle sauvegardé sous '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()
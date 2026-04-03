"""
Modulo per il training e l'ottimizzazione dei modelli di Machine Learning.

Questo modulo gestisce l'intera pipeline di apprendimento:
1. Split temporale rigoroso (prevenzione del Data Leakage)
2. Preparazione e scalatura delle features (StandardScaler)
3. Addestramento dei modelli "Base" (Default)
4. Ottimizzazione (Grid Search) applicata a TUTTI i modelli base
5. Creazione di un Super-Ensemble composto dai modelli già ottimizzati
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# ==============================================================================
# 1. SPLIT TEMPORALE
# ==============================================================================
def train_test_split_temporal(dataset_df, test_size=0.2):
    """
    Effettua la divisione cronologica dei dati per simulare un ambiente reale.
    
    ATTENZIONE: Non utilizza lo split casuale (random split)!
    
    Motivazione: In un dominio sportivo/temporale, uno split casuale inserirebbe nel Train Set informazioni provenienti dal futuro rispetto ai dati del Test Set, causando 
    Data Leakage e prestazioni illusorie.
    Questa funzione assegna rigorosamente l'80% delle partite più vecchie al Train e il 20% delle partite più recenti al Test, forzando il modello a predire il futuro.
    
    Args:
        dataset_df : Dataset completo già ordinato cronologicamente.
        test_size : Proporzione del dataset da riservare al test (default 20%).
    """
    split_index = int(len(dataset_df) * (1 - test_size))
    
    # Divisione puramente sequenziale (basata sull'indice ordinato)
    train_df = dataset_df.iloc[:split_index].copy()
    test_df = dataset_df.iloc[split_index:].copy()
    
    print(f"\n   ✓ Dati divisi cronologicamente per prevenire Data Leakage.")
    print(f"     Train: {len(train_df)} partite | Test: {len(test_df)} partite")
    
    # =================================================================
    # NUOVA AGGIUNTA: Salvataggio fisico del train e test set in formato CSV per trasparenza e futura analisi.
    # =================================================================
    import os
    os.makedirs('datasets', exist_ok=True)
    train_df.to_csv('datasets/train_set.csv', index=False)
    test_df.to_csv('datasets/test_set.csv', index=False)
    print("     ✓ File 'train_set.csv' e 'test_set.csv' esportati con successo.")
    # =================================================================
    
    
    return train_df, test_df


# ==============================================================================
# 2. PREPARAZIONE DATI E SCALING
# ==============================================================================

def prepare_data(train_df, test_df, feature_columns):
    """
    Costruisce le matrici delle features (X) e i vettori target (y) e applica
    la normalizzazione statistica.
    
    Regola dello Scaler:
    Lo StandardScaler (z = (x - μ) / σ) viene "fittato" (impara media e varianza)
    ESCLUSIVAMENTE sul Train Set. Poi viene usato per trasformare sia Train che Test.
    Questo evita che informazioni statistiche del Test set "inquinino" il Train set.
    """
    # -------------------------------------------------------------------------
    # Step 1: Creazione Target Binario (y)
    # -------------------------------------------------------------------------
    # Target = 1 se vince Player 1, 0 se vince Player 2
    train_df['target_winner'] = (train_df['winner'] == train_df['player_1']).astype(int)
    test_df['target_winner'] = (test_df['winner'] == test_df['player_1']).astype(int)
    
    # -------------------------------------------------------------------------
    # Step 2: Estrazione Matrici
    # -------------------------------------------------------------------------
    X_train_raw = train_df[feature_columns].values
    y_train = train_df['target_winner'].values
    
    X_test_raw = test_df[feature_columns].values
    y_test = test_df['target_winner'].values
    
    # -------------------------------------------------------------------------
    # Step 3: Normalizzazione (StandardScaler)
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    print(f"\n   ✓ StandardScaler applicato. Train Target Balance: {np.bincount(y_train)} (P2/P1)")
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


# ==============================================================================
# 3. TRAINING MODELLI BASE 
# ==============================================================================
def train_baseline_models(X_train_scaled, y_train):
    """Addestra i modelli con i parametri di default per avere un punto di partenza."""
    print("\n   -> Inizializzazione dei modelli con parametri standard...")
    models = {}
    
    # Logistic Regression Base
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    
    # Random Forest Base
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
    # Neural Network Base
    nn = MLPClassifier(hidden_layer_sizes=(20, 10), early_stopping=True, max_iter=1000, random_state=42)
    nn.fit(X_train_scaled, y_train)
    models['Neural Network'] = nn
    
    print("      ✓ Modelli Baseline addestrati con successo.")
    return models

# ==============================================================================
# 4. HYPERPARAMETER TUNING SU TUTTI I MODELLI (FASE 9)
# ==============================================================================
def tune_all_models(X_train_scaled, y_train):
    """
    Applica la Grid Search a TUTTI i modelli per trovare la configurazione ideale
    prima di confrontarli o unirli nell'Ensemble. 
    
    In pratica, proviamo diverse combinazioni di "impostazioni" (iperparametri) 
    per ogni algoritmo e teniamo solo la versione che ottiene le performance migliori in CV.
    """
    tuned_models = {}
    
    # --- 1. Tuning Logistic Regression ---
    print("\n   -> Esecuzione Grid Search su Logistic Regression...")
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10]
    }
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), 
                           param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
    grid_lr.fit(X_train_scaled, y_train)
    
    tuned_models['Logistic Regression'] = grid_lr.best_estimator_
    print(f"      ✓ Tuning completato. Parametri ottimali: {grid_lr.best_params_}")

    # --- 2. Tuning Random Forest ---
    print("\n   -> Esecuzione Grid Search su Random Forest...")
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(X_train_scaled, y_train)
    tuned_models['Random Forest'] = grid_rf.best_estimator_
    print(f"      ✓ Tuning completato. Parametri ottimali: {grid_rf.best_params_}")

    # --- 3. Tuning Neural Network ---
    print("\n   -> Esecuzione Grid Search su Neural Network...")
    param_grid_nn = {
        'alpha': [0.0001, 0.001, 0.01],
        'hidden_layer_sizes': [(20, 10), (30, 20)]
    }
    grid_nn = GridSearchCV(MLPClassifier(max_iter=1000, early_stopping=True, random_state=42), 
                           param_grid_nn, cv=5, scoring='accuracy', n_jobs=-1)
    grid_nn.fit(X_train_scaled, y_train)
    tuned_models['Neural Network'] = grid_nn.best_estimator_
    print(f"      ✓ Tuning completato. Parametri ottimali: {grid_nn.best_params_}")

    return tuned_models


# ==============================================================================
# 5. CREAZIONE SUPER-ENSEMBLE (FASE 10)
# ==============================================================================
def build_optimized_ensemble(tuned_models_dict, X_train_scaled, y_train):
    """
    Costruisce l'Ensemble utilizzando ESCLUSIVAMENTE i modelli che hanno 
    già superato con successo l'ottimizzazione (Grid Search).
    """
    print("\n   -> Costruzione Super-Ensemble Voting (Soft)...")
    
    ensemble_voting_model = VotingClassifier(
        estimators=[
            ('logistic', tuned_models_dict['Logistic Regression']), 
            ('random_forest', tuned_models_dict['Random Forest']), 
            ('neural_network', tuned_models_dict['Neural Network'])
        ],
        voting='soft'
    )
    
    # Addestriamo l'ensemble finale
    ensemble_voting_model.fit(X_train_scaled, y_train)
    print("      ✓ Ensemble unificato e addestrato con i modelli ottimizzati.")
    
    return ensemble_voting_model
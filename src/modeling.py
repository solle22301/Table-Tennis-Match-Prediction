"""
Modulo per il training e l'ottimizzazione dei modelli di Machine Learning.

Questo modulo gestisce l'intera pipeline di apprendimento:
1. Split temporale rigoroso (prevenzione del Data Leakage)
2. Preparazione e scalatura delle features (StandardScaler)
3. Addestramento di 3 algoritmi base + 1 Meta-Modello Ensemble (Soft Voting)
4. Ottimizzazione degli Iperparametri tramite Grid Search e Cross Validation
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np


# ==============================================================================
# 1. SPLIT TEMPORALE (DATA SPLITTING)
# ==============================================================================

def train_test_split_temporal(dataset_df, test_size=0.2):
    """
    Effettua la divisione cronologica dei dati per simulare un ambiente reale.
    
    ATTENZIONE: Non utilizza lo split casuale (random split)!
    
    Motivazione Accademica:
    In un dominio sportivo/temporale, uno split casuale inserirebbe nel Train Set
    informazioni provenienti dal futuro rispetto ai dati del Test Set, causando 
    Data Leakage e prestazioni illusorie.
    Questa funzione assegna rigorosamente l'80% delle partite più vecchie al Train
    e il 20% delle partite più recenti al Test, forzando il modello a predire il futuro.
    
    Args:
        dataset_df (DataFrame): Dataset completo già ordinato cronologicamente.
        test_size (float): Proporzione del dataset da riservare al test (default 20%).
    
    Returns:
        tuple: (train_df, test_df)
    """
    split_index = int(len(dataset_df) * (1 - test_size))
    
    # Divisione puramente sequenziale (basata sull'indice ordinato)
    train_df = dataset_df.iloc[:split_index].copy()
    test_df = dataset_df.iloc[split_index:].copy()
    
    print(f"\n✓ Split temporale completato:")
    print(f"   Train Set: {len(train_df)} partite "
          f"({train_df['date'].min().strftime('%Y-%m-%d')} → "
          f"{train_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"   Test Set:  {len(test_df)} partite "
          f"({test_df['date'].min().strftime('%Y-%m-%d')} → "
          f"{test_df['date'].max().strftime('%Y-%m-%d')})")
    
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
    
    print(f"\n✓ Dati preparati e normalizzati (Z-Score Scaling)")
    print(f"   X_train shape: {X_train_scaled.shape}")
    print(f"   X_test shape:  {X_test_scaled.shape}")
    print(f"   Distribuzione target train: {np.bincount(y_train)} (P2 Vince / P1 Vince)")
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


# ==============================================================================
# 3. TRAINING ALGORITMI
# ==============================================================================

def train_models(X_train_scaled, y_train):
    """
    Addestra i modelli di classificazione selezionati.
    
    Modelli Implementati:
    1. Logistic Regression: Ottima baseline lineare, fornisce probabilità calibrate.
    2. Random Forest: Algoritmo non lineare basato su alberi, robusto all'overfitting.
    3. Neural Network (MLP): Rete neurale profonda in grado di estrarre pattern complessi.
    4. Ensemble (Soft Voting): Media le probabilità dei primi tre modelli per 
       ridurre la varianza e aumentare la robustezza predittiva.
    """
    print("\n" + "=" * 80)
    print("ADDESTRAMENTO MODELLI DI MACHINE LEARNING")
    print("=" * 80)
    
    trained_models_dictionary = {}
    
    # -------------------------------------------------------------------------
    # Modello 1: Logistic Regression
    # -------------------------------------------------------------------------
    print("\n1. Addestramento Logistic Regression...")
    logistic_model = LogisticRegression(max_iter=1000, random_state=123)
    logistic_model.fit(X_train_scaled, y_train)
    trained_models_dictionary['Logistic Regression'] = logistic_model
    print("   ✓ Completato")
    
    # -------------------------------------------------------------------------
    # Modello 2: Random Forest
    # -------------------------------------------------------------------------
    print("\n2. Addestramento Random Forest...")
    # max_depth=5 per limitare la complessità ed evitare overfitting sui dati di training
    random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    random_forest_model.fit(X_train_scaled, y_train)
    trained_models_dictionary['Random Forest'] = random_forest_model
    print("   ✓ Completato")
    
    # -------------------------------------------------------------------------
    # Modello 3: Neural Network (Multi-Layer Perceptron)
    # -------------------------------------------------------------------------
    print("\n3. Addestramento Neural Network (MLP)...")
    neural_network_model = MLPClassifier(
        hidden_layer_sizes=(20, 10),   # Architettura: 2 hidden layers
        early_stopping=True,           # Previene l'overfitting fermando il training se non migliora
        validation_fraction=0.2,       # Riserva il 20% del train set per monitorare l'early stopping
        n_iter_no_change=10,
        alpha=0.001,                   # Regolarizzazione L2 (Ridge)
        max_iter=1000,
        random_state=42,
        verbose=False
    )
    neural_network_model.fit(X_train_scaled, y_train)
    trained_models_dictionary['Neural Network'] = neural_network_model
    print("   ✓ Completato")
    
    # -------------------------------------------------------------------------
    # Modello 4: Ensemble (Soft Voting Classifier)
    # -------------------------------------------------------------------------
    print("\n4. Generazione Ensemble (Soft Voting)...")
    ensemble_voting_model = VotingClassifier(
        estimators=[
            ('logistic', logistic_model), 
            ('random_forest', random_forest_model), 
            ('neural_network', neural_network_model)
        ],
        voting='soft'  # 'soft' calcola la media pesata delle probabilità stimate
    )
    ensemble_voting_model.fit(X_train_scaled, y_train)
    trained_models_dictionary['Ensemble Voting'] = ensemble_voting_model
    print("   ✓ Completato")
    
    return trained_models_dictionary


# ==============================================================================
# 4. HYPERPARAMETER TUNING
# ==============================================================================

def hyperparameter_tuning(X_train_scaled, y_train, best_model_name):
    """
    Esegue la ricerca esaustiva (GridSearchCV) per trovare i migliori iperparametri
    del modello risultato vincente nella fase di Cross-Validation.
    
    Args:
        X_train_scaled: Matrice delle features di training.
        y_train: Target di training.
        best_model_name (str): Il nome del modello da ottimizzare.
        
    Returns:
        Il modello ri-addestrato con i parametri ottimali.
    """
    print("\n" + "=" * 80)
    print("OTTIMIZZAZIONE IPERPARAMETRI (HYPERPARAMETER TUNING)")
    print("=" * 80)
    
    if best_model_name == 'Random Forest':
        print("\nAvvio GridSearchCV per Random Forest...")
        print("Nota: Questa operazione esplora molteplici combinazioni e richiederà tempo.")
        
        parameter_grid = {
            'n_estimators': [50, 100, 200],         # Numero di alberi decisionali
            'max_depth': [5, 10, 15],               # Profondità massima (overfitting control)
            'min_samples_split': [2, 5]             # Samples minimi per dividere un nodo
        }
        
        rf_base_model = RandomForestClassifier(random_state=42)
        grid_search_rf = GridSearchCV(
            estimator=rf_base_model, 
            param_grid=parameter_grid, 
            cv=5,                                   # Inner 5-Fold Cross Validation
            scoring='accuracy', 
            verbose=1, 
            n_jobs=-1                               # Parallelizzazione su tutti i core CPU
        )
        grid_search_rf.fit(X_train_scaled, y_train)
        
        print(f"\n✓ Ottimizzazione completata")
        print(f"   Parametri ottimali: {grid_search_rf.best_params_}")
        print(f"   Accuracy in CV:     {grid_search_rf.best_score_:.4f}")
        return grid_search_rf.best_estimator_
        
    elif best_model_name == 'Logistic Regression':
        print("\nAvvio GridSearchCV per Logistic Regression...")
        
        parameter_grid = {
            'C': [0.01, 0.1, 1, 10, 100],           # Inverso della forza di regolarizzazione
            'penalty': ['l1', 'l2'],                # Tipo di penalità
            'solver': ['liblinear']                 # Solver ottimizzato per L1/L2
        }
        
        lr_base_model = LogisticRegression(max_iter=1000, random_state=123)
        grid_search_lr = GridSearchCV(
            estimator=lr_base_model, 
            param_grid=parameter_grid,
            cv=5,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )
        grid_search_lr.fit(X_train_scaled, y_train)
        
        print(f"\n✓ Ottimizzazione completata")
        print(f"   Parametri ottimali: {grid_search_lr.best_params_}")
        print(f"   Accuracy in CV:     {grid_search_lr.best_score_:.4f}")
        return grid_search_lr.best_estimator_
        
    else:
        print(f"\n⚠️ Tuning non implementato specificamente per '{best_model_name}'.")
        print("  (I modelli complessi come le Reti Neurali o gli Ensemble richiedono")
        print("   tempi computazionali troppo elevati per una Grid Search locale).")
        return None
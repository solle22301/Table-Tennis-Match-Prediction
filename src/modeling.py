"""
Modulo per il training dei modelli di Machine Learning.

Questo modulo gestisce:
- Split temporale dei dati (evita data leakage)
- Preparazione features e target
- Normalizzazione con StandardScaler
- Training di 3 modelli di classificazione
- Hyperparameter tuning con GridSearchCV
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np


# ==============================================================================
# SPLIT TEMPORALE
# ==============================================================================

def train_test_split_temporal(df, test_size=0.2):
    """
    Effettua split temporale dei dati per evitare data leakage.
    
    IMPORTANTE: Non usa split random!
    
    Motivo:
    - Split random: partite del 10 gennaio in train, del 5 gennaio in test
      → Il modello "vede il futuro" durante il training (data leakage)
    
    - Split temporale: train = prime 80% partite, test = ultime 20%
      → Simula deployment reale: addestro su passato, testo su futuro
    
    Args:
        df (DataFrame): Dataset completo ordinato per data
        test_size (float): Percentuale test set (default 0.2 = 20%)
    
    Returns:
        tuple: (train_df, test_df)
    """
    # Calcola indice di split
    split_idx = int(len(df) * (1 - test_size))
    
    # Split semplice per indice (il df è già ordinato per data)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    # Stampa informazioni split
    print(f"\n✓ Split temporale:")
    print(f"   Train: {len(train)} partite "
          f"({train['date'].min().strftime('%Y-%m-%d')} → "
          f"{train['date'].max().strftime('%Y-%m-%d')})")
    print(f"   Test:  {len(test)} partite "
          f"({test['date'].min().strftime('%Y-%m-%d')} → "
          f"{test['date'].max().strftime('%Y-%m-%d')})")
    
    return train, test


# ==============================================================================
# PREPARAZIONE DATI
# ==============================================================================

def prepare_data(train, test, feature_cols):
    """
    Prepara features (X) e target (y), normalizza features.
    
    Pipeline:
    1. Crea target binario: 1 se vince player_1, 0 altrimenti
    2. Separa features (X) da target (y)
    3. Normalizza features con StandardScaler
    
    Normalizzazione StandardScaler:
    - Formula: z = (x - μ) / σ
    - Trasforma ogni feature in distribuzione con media=0, std=1
    - Necessaria per modelli sensibili alla scala (Logistic Regression, MLP)
    - Non necessaria per Random Forest ma non danneggia
    
    Args:
        train (DataFrame): Training set
        test (DataFrame): Test set
        feature_cols (list): Lista nomi features da usare
    
    Returns:
        tuple: (X_train, t_train, X_test, t_test, scaler)
               X normalizzati, y binari (0/1), scaler fitted
    """
    # -------------------------------------------------------------------------
    # Step 1: Crea target binario
    # -------------------------------------------------------------------------
    # Target = 1 se vince player_1, 0 se vince player_2
    train['target'] = (train['winner'] == train['player_1']).astype(int)
    test['target'] = (test['winner'] == test['player_1']).astype(int)
    
    # -------------------------------------------------------------------------
    # Step 2: Separa features da target
    # -------------------------------------------------------------------------
    X_train = train[feature_cols].values
    t_train = train['target'].values
    X_test = test[feature_cols].values
    t_test = test['target'].values
    
    # -------------------------------------------------------------------------
    # Step 3: Normalizzazione con StandardScaler
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    
    # Fit solo su train (evita data leakage)
    # Transform sia train che test
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Stampa info
    print(f"\n✓ Dati preparati e normalizzati")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   Distribuzione target train: {np.bincount(t_train)}")
    
    return X_train, t_train, X_test, t_test, scaler


# ==============================================================================
# TRAINING MODELLI
# ==============================================================================

def train_models(X_train, t_train):
    """
    Addestra 3 modelli di classificazione con parametri di default.
    
    Modelli scelti:
    
    1. Logistic Regression:
       - Modello lineare semplice e interpretabile
       - Veloce da addestrare
       - Buona baseline per problemi di classificazione binaria
    
    2. Random Forest:
       - Ensemble di alberi decisionali
       - Non lineare, cattura interazioni complesse tra features
       - Fornisce feature importance
       - Robusto a overfitting (con parametri corretti)
    
    3. Neural Network (MLP):
       - Multi-Layer Perceptron
       - Può apprendere pattern non lineari complessi
       - Richiede più dati e normalizzazione
    
    Args:
        X_train (array): Features normalizzate training
        t_train (array): Target training
    
    Returns:
        dict: Dizionario {nome_modello: modello_addestrato}
    """
    print("\n" + "=" * 80)
    print("TRAINING MODELLI")
    print("=" * 80)
    
    models = {}
    
    # -------------------------------------------------------------------------
    # Modello 1: Logistic Regression
    # -------------------------------------------------------------------------
    print("\n1. Logistic Regression...")
    
    # max_iter=1000: iterazioni massime per convergenza
    # random_state=123: riproducibilità
    lr = LogisticRegression(max_iter=1000, random_state=123)
    lr.fit(X_train, t_train)
    models['Logistic Regression'] = lr
    
    print("   ✓ Addestrato")
    
    # -------------------------------------------------------------------------
    # Modello 2: Random Forest
    # -------------------------------------------------------------------------
    print("\n2. Random Forest...")
    
    # n_estimators=100: numero di alberi nell'ensemble
    # max_depth=10: profondità massima alberi (previene overfitting)
    # random_state=42: riproducibilità
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, t_train)
    models['Random Forest'] = rf
    
    print("   ✓ Addestrato")
    
    # -------------------------------------------------------------------------
    # Modello 3: Neural Network (MLP)
    # -------------------------------------------------------------------------
    print("\n3. Neural Network (MLP)...")
    
    # hidden_layer_sizes=(32, 16): 2 hidden layer con 32 e 16 neuroni
    # max_iter=500: epoche massime
    # random_state=17: riproducibilità
    mlp = MLPClassifier(
        hidden_layer_sizes=(20, 10),
        early_stopping=True,           # Attiva early stopping
        validation_fraction=0.2,        # 20% train diventa validation
        n_iter_no_change=10,           # Ferma se non migliora per 10 iterazioni
        
        # Regolarizzazione aggiuntiva
        alpha=0.001,                   # L2 penalty (Ridge)
        max_iter=1000,
        random_state=42,
        verbose=False
)


    mlp.fit(X_train, t_train)
    models['Neural Network'] = mlp
    
    print("   ✓ Addestrato")
    
    return models


# ==============================================================================
# HYPERPARAMETER TUNING
# ==============================================================================

def hyperparameter_tuning(X_train, t_train, best_model_name):
    """
    Ottimizza iperparametri del modello migliore con GridSearchCV.
    
    GridSearchCV:
    - Prova tutte le combinazioni di parametri nella griglia
    - Usa cross-validation (cv=5) per valutare ogni combinazione
    - Restituisce il modello con i parametri migliori
    
    Parametri ottimizzati:
    
    Random Forest:
    - n_estimators: numero alberi (più alberi = più robusto ma più lento)
    - max_depth: profondità alberi (controlla overfitting)
    - min_samples_split: campioni minimi per split (controlla overfitting)
    
    Logistic Regression:
    - C: inverso della forza di regolarizzazione (C alto = meno regolarizzazione)
    - penalty: tipo regolarizzazione (L1 o L2)
    
    Args:
        X_train (array): Features training
        t_train (array): Target training
        best_model_name (str): Nome del modello da ottimizzare
    
    Returns:
        model: Modello ottimizzato, oppure None se non implementato
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Ottimizzazione Random Forest
    # -------------------------------------------------------------------------
    if best_model_name == 'Random Forest':
        print("\nOttimizzazione Random Forest con GridSearchCV...")
        print("Nota: Può richiedere diversi minuti...")
        
        # Griglia parametri da testare
        param_grid = {
            'n_estimators': [50, 100, 200],        # 3 valori
            'max_depth': [5, 10, 15],              # 3 valori
            'min_samples_split': [2, 5]            # 2 valori
        }
        # Totale combinazioni: 3 * 3 * 2 = 18
        # Con cv=5: 18 * 5 = 90 modelli addestrati
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=5,                    # 5-fold cross-validation
            scoring='accuracy',      # metrica da ottimizzare
            verbose=1,               # stampa progress
            n_jobs=-1                # usa tutti i core CPU
        )
        grid_search.fit(X_train, t_train)
        
        print(f"\n✓ Ottimizzazione completata")
        print(f"   Migliori parametri: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    # -------------------------------------------------------------------------
    # Ottimizzazione Logistic Regression
    # -------------------------------------------------------------------------
    elif best_model_name == 'Logistic Regression':
        print("\nOttimizzazione Logistic Regression con GridSearchCV...")
        
        # Griglia parametri da testare
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],    # 5 valori
            'penalty': ['l1', 'l2'],          # 2 valori
            'solver': ['liblinear']           # solver compatibile con L1/L2
        }
        # Totale combinazioni: 5 * 2 = 10
        # Con cv=5: 10 * 5 = 50 modelli addestrati
        
        lr = LogisticRegression(max_iter=1000, random_state=123)
        grid_search = GridSearchCV(
            lr, param_grid,
            cv=5,
            scoring='accuracy',
            verbose=1
        )
        grid_search.fit(X_train, t_train)
        
        print(f"\n✓ Ottimizzazione completata")
        print(f"   Migliori parametri: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    # -------------------------------------------------------------------------
    # Modelli non supportati
    # -------------------------------------------------------------------------
    else:
        print(f"\nHyperparameter tuning non implementato per {best_model_name}")
        return None

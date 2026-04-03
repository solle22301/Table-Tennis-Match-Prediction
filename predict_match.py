"""
Modulo per la simulazione e validazione delle predizioni.
Permette di inserire i nomi di due giocatori per testare il modello.

Il sistema ricerca automaticamente l'ultimo scontro tra i due atleti
all'interno del Test Set (l'ultimo 20% cronologico dei match),
garantendo la valutazione su dati mai visti durante l'addestramento.
La predizione avviene ricreando lo stato di forma esatto che i giocatori
avevano in quel preciso istante nel passato (Zero Data Leakage).
"""

import pandas as pd
import numpy as np
import unicodedata
import joblib
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# UTILITY: Normalizzazione Nomi
# ============================================================================
def normalize_player_name(raw_name):
    if pd.isna(raw_name):
        return ""
    normalized_string = ''.join(char for char in unicodedata.normalize('NFD', str(raw_name)) 
                                if unicodedata.category(char) != 'Mn')
    return normalized_string.lower().strip()

def invert_first_and_last_name(name):
    name_parts = name.strip().split()
    if len(name_parts) >= 2:
        return f"{name_parts[-1]} {name_parts[0]}"
    return name

# ============================================================================
# FASE 1: Caricamento Modelli Predittivi
# ============================================================================
def load_trained_models():
    print("   -> Inizializzazione sistema predittivo...")
    try:
        models_dictionary = {
            'logistic_regression': joblib.load('models/logistic_regression.pkl'),
            'random_forest': joblib.load('models/random_forest.pkl'),
            'neural_network': joblib.load('models/neural_network.pkl'),
            'ensemble_voting': joblib.load('models/ensemble_voting.pkl'),
            'scaler': joblib.load('models/scaler.pkl')
        }
        print("      ✓ Tutti i modelli di Machine Learning caricati con successo!")
        return models_dictionary
    except FileNotFoundError as error:
        print(f"\n   ❌ ERRORE CRITICO: File del modello non trovato.")
        print("      Assicurati di aver prima eseguito 'main.py' per addestrare e salvare i modelli.")
        return None

# ============================================================================
# FASE 2: Estrazione Dati Storici
# ============================================================================
def fetch_player_elo_rating(target_player_name, ranking_dataset):
    dataset_copy = ranking_dataset.copy()
    
    if 'Nome Giocatore' in dataset_copy.columns:
        dataset_copy = dataset_copy.rename(columns={'Nome Giocatore': 'Player'})
    if 'Rating ELO' in dataset_copy.columns:
        dataset_copy = dataset_copy.rename(columns={'Rating ELO': 'ELO'})

    dataset_copy['normalized_name'] = dataset_copy['Player'].apply(normalize_player_name)
    
    target_normalized = normalize_player_name(target_player_name)
    direct_match = dataset_copy[dataset_copy['normalized_name'] == target_normalized]

    if len(direct_match) > 0:
        return direct_match.iloc[0]['ELO'], direct_match.iloc[0]['Player']

    inverted_name = invert_first_and_last_name(target_player_name)
    inverted_normalized = normalize_player_name(inverted_name)
    inverted_match = dataset_copy[dataset_copy['normalized_name'] == inverted_normalized]

    if len(inverted_match) > 0:
        real_name_found = inverted_match.iloc[0]['Player']
        return inverted_match.iloc[0]['ELO'], real_name_found

    return None, None

def calculate_current_streak(results_list):
    current_streak = 0
    for match_result in reversed(results_list):
        if match_result == 1:
            if current_streak >= 0:
                current_streak += 1
            else:
                break
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                break
    return current_streak

# ============================================================================
# FASE 3: Generazione Features per la Simulazione
# ============================================================================
def generate_live_features(player_1_input_name, player_2_input_name, historical_matches_df, ranking_df):
    """Ricostruisce le 10 features esatte basandosi SOLO sul dataframe storico fornito."""
    
    p1_elo_score, p1_real_name = fetch_player_elo_rating(player_1_input_name, ranking_df)
    p2_elo_score, p2_real_name = fetch_player_elo_rating(player_2_input_name, ranking_df)

    if p1_elo_score is None or p2_elo_score is None:
        return None, None, None

    elo_difference = p1_elo_score - p2_elo_score

    df = historical_matches_df.copy()
    df['p1_norm'] = df['player_1'].apply(normalize_player_name)
    df['p2_norm'] = df['player_2'].apply(normalize_player_name)
    df['winner_norm'] = df['winner'].apply(normalize_player_name)

    p1_norm_name = normalize_player_name(p1_real_name)
    p2_norm_name = normalize_player_name(p2_real_name)

    p1_history_df = df[(df['p1_norm'] == p1_norm_name) | (df['p2_norm'] == p1_norm_name)]
    p2_history_df = df[(df['p1_norm'] == p2_norm_name) | (df['p2_norm'] == p2_norm_name)]

    p1_results_list = (p1_history_df['winner_norm'] == p1_norm_name).astype(int).tolist()
    p2_results_list = (p2_history_df['winner_norm'] == p2_norm_name).astype(int).tolist()

    p1_last_5_results = p1_results_list[-5:] if len(p1_results_list) >= 5 else p1_results_list
    p2_last_5_results = p2_results_list[-5:] if len(p2_results_list) >= 5 else p2_results_list
    
    p1_win_rate_last5 = np.mean(p1_last_5_results) if p1_last_5_results else 0.0
    p2_win_rate_last5 = np.mean(p2_last_5_results) if p2_last_5_results else 0.0

    head_to_head_matches = df[
        ((df['p1_norm'] == p1_norm_name) & (df['p2_norm'] == p2_norm_name)) |
        ((df['p1_norm'] == p2_norm_name) & (df['p2_norm'] == p1_norm_name))
    ]

    p1_head_to_head_wins = (head_to_head_matches['winner_norm'] == p1_norm_name).sum()
    p2_head_to_head_wins = (head_to_head_matches['winner_norm'] == p2_norm_name).sum()
    
    total_head_to_head = p1_head_to_head_wins + p2_head_to_head_wins
    p1_head_to_head_win_ratio = (p1_head_to_head_wins / total_head_to_head) if total_head_to_head > 0 else 0.5

    p1_current_streak = calculate_current_streak(p1_results_list)
    p2_current_streak = calculate_current_streak(p2_results_list)

    p1_form_volatility = np.std(p1_last_5_results) if len(p1_last_5_results) >= 3 else 0.0
    p2_form_volatility = np.std(p2_last_5_results) if len(p2_last_5_results) >= 3 else 0.0

    features_dict = {
        'elo_diff': elo_difference,
        'p1_win_rate_last5': p1_win_rate_last5,
        'p2_win_rate_last5': p2_win_rate_last5,
        'p1_head_to_head_wins': p1_head_to_head_wins,
        'p2_head_to_head_wins': p2_head_to_head_wins,
        'p1_head_to_head_win_ratio': p1_head_to_head_win_ratio,
        'p1_streak': p1_current_streak,
        'p2_streak': p2_current_streak,
        'p1_form_volatility': p1_form_volatility,
        'p2_form_volatility': p2_form_volatility,
        'context_p1_elo': p1_elo_score,
        'context_p2_elo': p2_elo_score
    }
    
    return features_dict, p1_real_name, p2_real_name

# ============================================================================
# FASE 4: Predizione Vincitore e Verifica Ground Truth
# ============================================================================
def predict_and_verify_match(player_1_input, player_2_input, full_dataset, test_dataset, ranking_df, models_dict):
    """Cerca i giocatori nel Test Set, isola il passato, calcola e verifica."""
    
    p1_norm = normalize_player_name(player_1_input)
    p2_norm = normalize_player_name(player_2_input)
    
    # Lavoriamo su una copia per non modificare l'originale
    search_df = test_dataset.copy()
    search_df['p1_norm'] = search_df['player_1'].apply(normalize_player_name)
    search_df['p2_norm'] = search_df['player_2'].apply(normalize_player_name)
    
    # Cerchiamo lo scontro specifico nel test set
    match_found = search_df[
        ((search_df['p1_norm'] == p1_norm) & (search_df['p2_norm'] == p2_norm)) |
        ((search_df['p1_norm'] == p2_norm) & (search_df['p2_norm'] == p1_norm))
    ]
    
    if len(match_found) == 0:
        print(f"\n ⚠️  Scontro non trovato nel Test Set.")
        print(f"    Assicurati di aver inserito due giocatori presenti nel file 'test_set.csv'.")
        return

    # Prendiamo l'ultimo scontro tra di loro trovato nel test set
    target_match = match_found.iloc[-1]
    
    # Dobbiamo trovare l'indice esatto di questa partita nel file completo (storico totale)
    # per poter tagliare il passato in modo preciso.
    target_date = target_match['date']
    player_1_real = target_match['player_1']
    player_2_real = target_match['player_2']
    actual_winner = target_match['winner']

    # AGGIUNTA IMPORTANTE: usiamo anche il match_id per individuare la riga esatta senza ambiguità
    match_absolute_index = full_dataset[
        (full_dataset['date'] == target_date) & 
        (full_dataset['player_1'] == player_1_real) & 
        (full_dataset['player_2'] == player_2_real) &
        (full_dataset['match_id'] == target_match['match_id'])
    ].index[0]

    print("\n" + "="*80)
    print(f" 🎯 VERIFICA MATCH: {player_1_real.upper()} vs {player_2_real.upper()}")
    print(f" 📅 Data incontro: {target_date}")
    print("="*80)

    # LA MACCHINA DEL TEMPO: Tagliamo il database un attimo prima dell'incontro
    past_matches_df = full_dataset.iloc[:match_absolute_index]
    
    features, p1_name, p2_name = generate_live_features(player_1_real, player_2_real, past_matches_df, ranking_df)
    
    if features is None:
        print(" ❌ Errore: Uno dei giocatori non è presente nel ranking ELO.")
        return

    print("\n 📈 STATO DI FORMA AL MOMENTO DELLA PARTITA (Le 10 Features):")
    print("-" * 80)
    print(f"   Differenza ELO (P1 - P2)        : {features['elo_diff']:8.1f}  (P1 ELO: {features['context_p1_elo']} | P2 ELO: {features['context_p2_elo']})")
    print(f"   Forma Recente P1 (Win Rate)     : {features['p1_win_rate_last5']*100:8.1f}%")
    print(f"   Forma Recente P2 (Win Rate)     : {features['p2_win_rate_last5']*100:8.1f}%")
    print(f"   Scontri Diretti (Vittorie P1/P2): {features['p1_head_to_head_wins']:.0f} / {features['p2_head_to_head_wins']:.0f}")
    print(f"   Supremazia Scontri Diretti (P1) : {features['p1_head_to_head_win_ratio']*100:8.1f}%")
    print(f"   Momentum P1 (Streak attuale)    : {features['p1_streak']:8.0f}")
    print(f"   Momentum P2 (Streak attuale)    : {features['p2_streak']:8.0f}")
    print(f"   Volatilità Rendimento P1        : {features['p1_form_volatility']:8.3f}")
    print(f"   Volatilità Rendimento P2        : {features['p2_form_volatility']:8.3f}")
    print("-" * 80)

    X_raw_array = np.array([[
        features['elo_diff'], features['p1_win_rate_last5'], features['p2_win_rate_last5'],
        features['p1_head_to_head_wins'], features['p2_head_to_head_wins'], features['p1_head_to_head_win_ratio'],
        features['p1_streak'], features['p2_streak'], features['p1_form_volatility'], features['p2_form_volatility']
    ]])

    X_scaled_array = models_dict['scaler'].transform(X_raw_array)

    print("\n 🤖 RESPONSO DEI MODELLI DI INTELLIGENZA ARTIFICIALE:")
    print("=" * 80)

    model_keys = ['logistic_regression', 'random_forest', 'neural_network', 'ensemble_voting']
    display_names = ['Logistic Regression (Modello Lineare)', 
                     'Random Forest (Modello ad Albero)', 
                     'Neural Network (Rete Neurale Profonda)', 
                     '🏆 ENSEMBLE VOTING (Super-Modello Unificato)']

    ensemble_prediction = None

    for key, display_name in zip(model_keys, display_names):
        model = models_dict[key]
        prediction_class = model.predict(X_scaled_array)[0]
        prediction_probabilities = model.predict_proba(X_scaled_array)[0]
        
        predicted_winner = p1_name if prediction_class == 1 else p2_name
        confidence_score = prediction_probabilities[1] if prediction_class == 1 else prediction_probabilities[0]
        
        if key == 'ensemble_voting':
            ensemble_prediction = predicted_winner
            
        print(f"   ▶ {display_name}")
        print(f"      Favorito: {predicted_winner}  |  Confidenza: {confidence_score*100:.2f}%\n")
        
    print("=" * 80)
    print(" 🏁 VERIFICA RISULTATO REALE")
    print("=" * 80)
    print(f"   Il Vincitore reale è stato : {actual_winner.upper()}")
    
    if ensemble_prediction.lower() == actual_winner.lower():
        print(f"   ✅ ESITO: IL SISTEMA HA PREVISTO CORRETTAMENTE IL VINCITORE!")
    else:
        print(f"   ❌ ESITO: IL SISTEMA NON HA INDOVINATO IL VINCITORE.")
    print("=" * 80)

# ============================================================================
# RUNNER INTERATTIVO
# ============================================================================
def run_terminal_interface():
    print("\n" + "="*80)
    print(" 🎾 TABLE TENNIS PREDICTION SYSTEM - TERMINALE TEST")
    print("="*80)

    print("\n   -> Caricamento database in memoria...")
    try:
        # Carica il dataset completo per calcolare la cronologia passata
        full_matches_df = pd.read_csv('datasets/TT_Elite_Series_Dataset.csv')
        full_matches_df['date'] = pd.to_datetime(full_matches_df['date'])
        # CORREZIONE: Ordinamento stabile per data E match_id
        full_matches_df = full_matches_df.sort_values(['date', 'match_id']).reset_index(drop=True)
        
        ranking_df = pd.read_csv('datasets/RANKING-TT-ELITE-SERIES.csv')
        
        # Carica il Test Set specifico salvato dal modulo di modeling
        test_matches_df = pd.read_csv('datasets/test_set.csv')
        
        print("      ✓ Database e Test Set pronti.")
        print(f"      (Partite disponibili per il test: {len(test_matches_df)})")
        
    except FileNotFoundError:
        print("      ❌ ERRORE: Dataset non trovati. Assicurati di aver eseguito 'main.py' per generare i file.")
        return

    trained_models = load_trained_models()
    if not trained_models: 
        return

    print("\n" + "*" * 80)
    print(" Inserisci i nomi di due giocatori presenti nel file 'test_set.csv'.")
    print(" Digita 'exit' per uscire dal terminale.")
    print("*" * 80)

    while True:
        player_1_input = input("\n 👤 Inserisci Nome Player 1 (es. 'Kamil Kostyk'): ").strip()
        if player_1_input.lower() in ['exit', 'quit', 'esci']:
            print("\n 👋 Chiusura sistema. Arrivederci!\n")
            break

        player_2_input = input(" 👤 Inserisci Nome Player 2 (es. 'Amirreza Abbasi'): ").strip()
        if player_2_input.lower() in ['exit', 'quit', 'esci']:
            print("\n 👋 Chiusura sistema. Arrivederci!\n")
            break
            
        if player_1_input == "" or player_2_input == "":
            print(" ⚠️  Nomi non validi. Riprova.")
            continue

        predict_and_verify_match(player_1_input, player_2_input, full_matches_df, test_matches_df, ranking_df, trained_models)

if __name__ == "__main__":
    run_terminal_interface()
"""
Modulo per la simulazione live delle predizioni.
Permette di inserire due giocatori e calcolare le probabilità di vittoria
utilizzando i modelli di Machine Learning addestrati.

Rispetta rigorosamente la stessa logica di Feature Engineering (Rolling Window)
utilizzata in fase di addestramento, garantendo coerenza matematica (No Training-Serving Skew).
"""

import pandas as pd
import numpy as np
import unicodedata
import joblib

# ============================================================================
# UTILITY: Normalizzazione Nomi (Clean Code)
# ============================================================================
def normalize_player_name(raw_name):
    """Rimuove accenti e caratteri speciali per garantire un match perfetto nei database."""
    if pd.isna(raw_name):
        return ""
    # Normalizzazione Unicode per rimuovere i segni diacritici (accenti)
    normalized_string = ''.join(char for char in unicodedata.normalize('NFD', str(raw_name)) 
                                if unicodedata.category(char) != 'Mn')
    return normalized_string.lower().strip()

def invert_first_and_last_name(name):
    """Inverte 'Nome Cognome' in 'Cognome Nome' per gestire i formati misti nei dataset."""
    name_parts = name.strip().split()
    if len(name_parts) >= 2:
        return f"{name_parts[-1]} {name_parts[0]}"
    return name

# ============================================================================
# FASE 1: Caricamento Modelli Predittivi
# ============================================================================
def load_trained_models():
    """Carica in memoria i modelli serializzati durante l'esecuzione di main.py."""
    try:
        models_dictionary = {
            'logistic_regression': joblib.load('models/logistic_regression.pkl'),
            'random_forest': joblib.load('models/random_forest.pkl'),
            'neural_network': joblib.load('models/neural_network.pkl'),
            'ensemble_voting': joblib.load('models/ensemble_voting.pkl'), # Aggiunto il modello avanzato
            'scaler': joblib.load('models/scaler.pkl')
        }
        print("✅ Modelli di Machine Learning caricati con successo!")
        return models_dictionary
    except FileNotFoundError as error:
        print(f"❌ Errore critico: {error}")
        print("   Assicurati di aver prima eseguito 'main.py' per addestrare e salvare i modelli.")
        return None

# ============================================================================
# FASE 2: Estrazione Dati Storici
# ============================================================================
def fetch_player_elo_rating(target_player_name, ranking_dataset):
    """
    Cerca il punteggio ELO ufficiale del giocatore nel dataset di Ranking,
    provando sia il formato standard che quello invertito.
    """
    dataset_copy = ranking_dataset.copy()
    
    # Standardizzazione preventiva dei nomi delle colonne
    if 'Nome Giocatore' in dataset_copy.columns:
        dataset_copy = dataset_copy.rename(columns={'Nome Giocatore': 'Player'})
    if 'Rating ELO' in dataset_copy.columns:
        dataset_copy = dataset_copy.rename(columns={'Rating ELO': 'ELO'})

    dataset_copy['normalized_name'] = dataset_copy['Player'].apply(normalize_player_name)
    
    # Tentativo 1: Ricerca con il nome fornito dall'utente
    target_normalized = normalize_player_name(target_player_name)
    direct_match = dataset_copy[dataset_copy['normalized_name'] == target_normalized]

    if len(direct_match) > 0:
        return direct_match.iloc[0]['ELO'], target_player_name

    # Tentativo 2: Ricerca invertendo Cognome e Nome
    inverted_name = invert_first_and_last_name(target_player_name)
    inverted_normalized = normalize_player_name(inverted_name)
    inverted_match = dataset_copy[dataset_copy['normalized_name'] == inverted_normalized]

    if len(inverted_match) > 0:
        real_name_found = inverted_match.iloc[0]['Player']
        print(f"   ℹ️  Giocatore '{target_player_name}' trovato nel ranking come '{real_name_found}'")
        return inverted_match.iloc[0]['ELO'], real_name_found

    return None, None

def calculate_current_streak(results_list):
    """
    Calcola la striscia aperta ininterrotta di vittorie (positiva) o sconfitte (negativa).
    Legge la lista dei risultati partendo dalla partita più recente (in fondo alla lista).
    """
    current_streak = 0
    for match_result in reversed(results_list):
        if match_result == 1: # Vittoria
            if current_streak >= 0:
                current_streak += 1
            else:
                break # La striscia si è interrotta
        else: # Sconfitta (0)
            if current_streak <= 0:
                current_streak -= 1
            else:
                break
    return current_streak

# ============================================================================
# FASE 3: Generazione Features per la Simulazione
# ============================================================================
def generate_live_features(player_1_input_name, player_2_input_name, historical_matches_df, ranking_df):
    """Ricostruisce le 10 features esatte richieste dal modello per la predizione live."""
    
    # 1. Recupero ELO
    p1_elo_score, p1_real_name = fetch_player_elo_rating(player_1_input_name, ranking_df)
    p2_elo_score, p2_real_name = fetch_player_elo_rating(player_2_input_name, ranking_df)

    if p1_elo_score is None:
        print(f"❌ Impossibile procedere: '{player_1_input_name}' non è presente nel Ranking ufficiale.")
        return None
    if p2_elo_score is None:
        print(f"❌ Impossibile procedere: '{player_2_input_name}' non è presente nel Ranking ufficiale.")
        return None

    # FEATURE 1: Ranking
    elo_difference = p1_elo_score - p2_elo_score

    # Preparazione dataset storico per le altre features
    historical_matches_df['p1_norm'] = historical_matches_df['player_1'].apply(normalize_player_name)
    historical_matches_df['p2_norm'] = historical_matches_df['player_2'].apply(normalize_player_name)
    historical_matches_df['winner_norm'] = historical_matches_df['winner'].apply(normalize_player_name)

    p1_norm_name = normalize_player_name(p1_real_name)
    p2_norm_name = normalize_player_name(p2_real_name)

    # Estrazione cronologica dello storico dei due giocatori
    p1_history_df = historical_matches_df[
        (historical_matches_df['p1_norm'] == p1_norm_name) | 
        (historical_matches_df['p2_norm'] == p1_norm_name)
    ]
    p2_history_df = historical_matches_df[
        (historical_matches_df['p1_norm'] == p2_norm_name) | 
        (historical_matches_df['p2_norm'] == p2_norm_name)
    ]

    # Trasformiamo i DataFrame in semplici liste di 1 (Vittoria) e 0 (Sconfitta)
    p1_results_list = (p1_history_df['winner_norm'] == p1_norm_name).astype(int).tolist()
    p2_results_list = (p2_history_df['winner_norm'] == p2_norm_name).astype(int).tolist()

    # FEATURES 2 & 3: Stato di forma recente (Win Rate su ultime 5)
    p1_last_5_results = p1_results_list[-5:] if len(p1_results_list) >= 5 else p1_results_list
    p2_last_5_results = p2_results_list[-5:] if len(p2_results_list) >= 5 else p2_results_list
    
    p1_win_rate_last5 = np.mean(p1_last_5_results) if p1_last_5_results else 0.0
    p2_win_rate_last5 = np.mean(p2_last_5_results) if p2_last_5_results else 0.0

    # FEATURES 4, 5 & 6: Scontri Diretti (Head-to-Head)
    head_to_head_matches = historical_matches_df[
        ((historical_matches_df['p1_norm'] == p1_norm_name) & (historical_matches_df['p2_norm'] == p2_norm_name)) |
        ((historical_matches_df['p1_norm'] == p2_norm_name) & (historical_matches_df['p2_norm'] == p1_norm_name))
    ]

    p1_head_to_head_wins = (head_to_head_matches['winner_norm'] == p1_norm_name).sum()
    p2_head_to_head_wins = (head_to_head_matches['winner_norm'] == p2_norm_name).sum()
    
    total_head_to_head = p1_head_to_head_wins + p2_head_to_head_wins
    p1_head_to_head_win_ratio = (p1_head_to_head_wins / total_head_to_head) if total_head_to_head > 0 else 0.5

    # FEATURES 7 & 8: Momentum (Strisce aperte)
    p1_current_streak = calculate_current_streak(p1_results_list)
    p2_current_streak = calculate_current_streak(p2_results_list)

    # FEATURES 9 & 10: Volatilità (Deviazione standard della forma recente)
    p1_form_volatility = np.std(p1_last_5_results) if len(p1_last_5_results) >= 3 else 0.0
    p2_form_volatility = np.std(p2_last_5_results) if len(p2_last_5_results) >= 3 else 0.0

    # Restituiamo un dizionario pulito con le esatte 10 features
    return {
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
        # Dati di contesto non passati al modello:
        'context_p1_elo': p1_elo_score,
        'context_p2_elo': p2_elo_score
    }

# ============================================================================
# FASE 4: Predizione Vincitore
# ============================================================================
def predict_match_outcome(player_1_name, player_2_name, historical_matches_df, ranking_df, models_dict):
    """Esegue la predizione finale stampando un report dettagliato."""
    print("\n" + "="*80)
    print(f"🎯 SIMULAZIONE MATCH: {player_1_name} vs {player_2_name}")
    print("="*80)

    features = generate_live_features(player_1_name, player_2_name, historical_matches_df, ranking_df)
    if features is None:
        return None

    print("\n📈 Features fornite in input ai Modelli (Le 10 definitive):")
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

    # Costruzione dell'Array X nel rigoroso ordine richiesto dai modelli
    X_raw_array = np.array([[
        features['elo_diff'],
        features['p1_win_rate_last5'],
        features['p2_win_rate_last5'],
        features['p1_head_to_head_wins'],
        features['p2_head_to_head_wins'],
        features['p1_head_to_head_win_ratio'],
        features['p1_streak'],
        features['p2_streak'],
        features['p1_form_volatility'],
        features['p2_form_volatility']
    ]])

    # Normalizzazione dei dati tramite lo scaler addestrato
    X_scaled_array = models_dict['scaler'].transform(X_raw_array)

    print("\n🤖 RESPONSO DEI MODELLI DI MACHINE LEARNING:")
    print("=" * 80)

    model_keys = ['logistic_regression', 'random_forest', 'neural_network', 'ensemble_voting']
    display_names = ['Logistic Regression (Linear)', 'Random Forest (Tree-Based)', 'Neural Network (MLP)', '🏆 ENSEMBLE VOTING (Best Model)']

    for key, display_name in zip(model_keys, display_names):
        model = models_dict[key]
        prediction_class = model.predict(X_scaled_array)[0]
        prediction_probabilities = model.predict_proba(X_scaled_array)[0]
        
        predicted_winner = player_1_name if prediction_class == 1 else player_2_name
        confidence_score = prediction_probabilities[1] if prediction_class == 1 else prediction_probabilities[0]
        
        print(f"   {display_name}:")
        print(f"     > Favorito: {predicted_winner}")
        print(f"     > Sicurezza predizione: {confidence_score*100:.2f}%\n")
        
    print("=" * 80)
    return True

# ============================================================================
# RUNNER INTERATTIVO
# ============================================================================
def start_interactive_session():
    """Avvia la console interattiva per interrogare il modello."""
    print("\n" + "="*80)
    print("🎾 TABLE TENNIS PREDICTION SYSTEM - TERMINALE LIVE")
    print("="*80)

    try:
        matches_dataset = pd.read_csv('datasets/TT_Elite_COMBINED_9415_matches.csv')
        ranking_dataset = pd.read_csv('datasets/RANKING-TT-ELITE-SERIES.csv')
    except FileNotFoundError:
        print("❌ Dataset non trovati nella cartella 'datasets/'.")
        return

    trained_models = load_trained_models()
    if not trained_models: return

    while True:
        print("\n" + "-"*80)
        player_1_input = input("\n👤 Inserisci Nome Player 1 (o scrivi 'exit' per uscire): ").strip()
        if player_1_input.lower() in ['exit', 'quit']:
            print("\n👋 Chiusura sistema. Arrivederci!\n")
            break

        player_2_input = input("👤 Inserisci Nome Player 2: ").strip()

        predict_match_outcome(player_1_input, player_2_input, matches_dataset, ranking_dataset, trained_models)

if __name__ == "__main__":
    start_interactive_session()

import pandas as pd
import numpy as np
import unicodedata
import joblib
import os
from datetime import datetime

# ============================================================================
# FUNZIONE 1: Normalizzazione nomi
# ============================================================================
def normalize_name(name):
    """Normalizza nome per matching"""
    if pd.isna(name):
        return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', str(name)) 
                   if unicodedata.category(c) != 'Mn')
    return name.lower().strip()

def invert_name(name):
    """Inverte 'Nome Cognome' in 'Cognome Nome' e viceversa"""
    parts = name.strip().split()
    if len(parts) >= 2:
        # Prende primo e ultimo (gestisce anche nomi multipli)
        return f"{parts[-1]} {parts[0]}"
    return name

# ============================================================================
# FUNZIONE 2: Carica i modelli addestrati
# ============================================================================
def load_models():
    """Carica modelli salvati da main.py"""
    try:
        models = {
            'logistic': joblib.load('models/logistic_regression.pkl'),
            'random_forest': joblib.load('models/random_forest.pkl'),
            'neural_network': joblib.load('models/neural_network.pkl'),
            'scaler': joblib.load('models/scaler.pkl')
        }
        print("✅ Modelli caricati con successo!")
        return models
    except FileNotFoundError as e:
        print(f"❌ Errore: {e}")
        print("   Assicurati di aver eseguito main.py per addestrare i modelli")
        return None

# ============================================================================
# FUNZIONE 3: Trova giocatore nel ranking (con inversione automatica)
# ============================================================================
def find_player_elo(player_name, ranking_df):
    """
    Cerca giocatore nel ranking provando sia 'Nome Cognome' che 'Cognome Nome'
    """
    # Rinomina colonne se necessario
    ranking_df_copy = ranking_df.copy()
    if 'Nome Giocatore' in ranking_df_copy.columns:
        ranking_df_copy = ranking_df_copy.rename(columns={'Nome Giocatore': 'Player'})
    if 'Rating ELO' in ranking_df_copy.columns:
        ranking_df_copy = ranking_df_copy.rename(columns={'Rating ELO': 'ELO'})

    ranking_df_copy['player_norm'] = ranking_df_copy['Player'].apply(normalize_name)

    # Prova 1: Nome come inserito
    p_norm = normalize_name(player_name)
    match = ranking_df_copy[ranking_df_copy['player_norm'] == p_norm]

    if len(match) > 0:
        return match.iloc[0]['ELO'], player_name

    # Prova 2: Nome invertito
    p_inverted = invert_name(player_name)
    p_inverted_norm = normalize_name(p_inverted)
    match = ranking_df_copy[ranking_df_copy['player_norm'] == p_inverted_norm]

    if len(match) > 0:
        print(f"   ℹ️  '{player_name}' trovato come '{match.iloc[0]['Player']}'")
        return match.iloc[0]['ELO'], match.iloc[0]['Player']

    return None, None

# ============================================================================
# FUNZIONE 4: Calcola features per nuova partita
# ============================================================================
def calculate_match_features(player1_name, player2_name, matches_df, ranking_df):
    """Calcola le 12 features necessarie per la predizione"""

    # ========================================================================
    # FEATURE 1-2: ELO Rating (con inversione automatica)
    # ========================================================================
    player1_elo, p1_real_name = find_player_elo(player1_name, ranking_df)
    player2_elo, p2_real_name = find_player_elo(player2_name, ranking_df)

    if player1_elo is None:
        print(f"❌ Giocatore non trovato nel ranking: {player1_name}")
        return None
    if player2_elo is None:
        print(f"❌ Giocatore non trovato nel ranking: {player2_name}")
        return None

    # Usa nomi reali trovati nel ranking
    p1_norm = normalize_name(p1_real_name)
    p2_norm = normalize_name(p2_real_name)

    # ========================================================================
    # FEATURE 3-4: ELO Difference e Sum
    # ========================================================================
    elo_diff = player1_elo - player2_elo
    elo_sum = player1_elo + player2_elo

    # ========================================================================
    # Prepara dataset partite
    # ========================================================================
    matches_df = matches_df.copy()
    matches_df['player1_norm'] = matches_df['player_1'].apply(normalize_name)
    matches_df['player2_norm'] = matches_df['player_2'].apply(normalize_name)
    matches_df['winner_norm'] = matches_df['winner'].apply(normalize_name)

    # Partite di P1 e P2
    p1_matches = matches_df[
        (matches_df['player1_norm'] == p1_norm) | 
        (matches_df['player2_norm'] == p1_norm)
    ].copy()

    p2_matches = matches_df[
        (matches_df['player1_norm'] == p2_norm) | 
        (matches_df['player2_norm'] == p2_norm)
    ].copy()

    # ========================================================================
    # FEATURE 5-6: Win Rate Overall
    # ========================================================================
    def calc_winrate(player_norm, player_matches):
        if len(player_matches) == 0:
            return 0.5
        wins = (player_matches['winner_norm'] == player_norm).sum()
        return wins / len(player_matches)

    p1_winrate_overall = calc_winrate(p1_norm, p1_matches)
    p2_winrate_overall = calc_winrate(p2_norm, p2_matches)

    # ========================================================================
    # FEATURE 7-8: Win Rate Last 5
    # ========================================================================
    p1_last5 = p1_matches.tail(5)
    p2_last5 = p2_matches.tail(5)

    p1_winrate_last5 = calc_winrate(p1_norm, p1_last5)
    p2_winrate_last5 = calc_winrate(p2_norm, p2_last5)

    # ========================================================================
    # FEATURE 9-10: Head-to-Head
    # ========================================================================
    h2h_matches = matches_df[
        ((matches_df['player1_norm'] == p1_norm) & (matches_df['player2_norm'] == p2_norm)) |
        ((matches_df['player1_norm'] == p2_norm) & (matches_df['player2_norm'] == p1_norm))
    ]

    if len(h2h_matches) > 0:
        h2h_p1_wins = (h2h_matches['winner_norm'] == p1_norm).sum()
        h2h_p2_wins = (h2h_matches['winner_norm'] == p2_norm).sum()
    else:
        h2h_p1_wins = 0
        h2h_p2_wins = 0

    # ========================================================================
    # FEATURE 11-12: Streak
    # ========================================================================
    def calc_streak(player_norm, player_matches):
        if len(player_matches) < 3:
            return 0
        last_3 = player_matches.tail(3)
        wins_last_3 = (last_3['winner_norm'] == player_norm).sum()
        return wins_last_3 - 1.5

    p1_streak = calc_streak(p1_norm, p1_matches)
    p2_streak = calc_streak(p2_norm, p2_matches)

    # ========================================================================
    # Ritorna features
    # ========================================================================
    features = {
        'player1_elo': player1_elo,
        'player2_elo': player2_elo,
        'elo_diff': elo_diff,
        'elo_sum': elo_sum,
        'p1_winrate_overall': p1_winrate_overall,
        'p2_winrate_overall': p2_winrate_overall,
        'p1_winrate_last5': p1_winrate_last5,
        'p2_winrate_last5': p2_winrate_last5,
        'h2h_p1_wins': h2h_p1_wins,
        'h2h_p2_wins': h2h_p2_wins,
        'p1_streak': p1_streak,
        'p2_streak': p2_streak,
        'p1_matches_count': len(p1_matches),
        'p2_matches_count': len(p2_matches)
    }

    return features

# ============================================================================
# FUNZIONE 5: Predizione vincitore
# ============================================================================
def predict_winner(player1_name, player2_name, matches_df, ranking_df, models):
    """Predice il vincitore della partita"""
    print("\n" + "="*80)
    print(f"🎯 PREDIZIONE: {player1_name} vs {player2_name}")
    print("="*80)

    print("\n📊 Calcolo features...")
    features = calculate_match_features(player1_name, player2_name, 
                                        matches_df, ranking_df)

    if features is None:
        print("❌ Impossibile calcolare features (giocatori non trovati)")
        return None

    print("\n📈 Features calcolate:")
    print("-"*80)
    print(f"   player1_elo              : {features['player1_elo']:8.1f}")
    print(f"   player2_elo              : {features['player2_elo']:8.1f}")
    print(f"   elo_diff                 : {features['elo_diff']:8.1f}")
    print(f"   p1_winrate_last5         : {features['p1_winrate_last5']:8.3f}")
    print(f"   p2_winrate_last5         : {features['p2_winrate_last5']:8.3f}")
    print(f"   h2h (P1/P2 wins)         : {features['h2h_p1_wins']:.0f}/{features['h2h_p2_wins']:.0f}")
    print(f"   p1_streak                : {features['p1_streak']:8.1f}")
    print(f"   p2_streak                : {features['p2_streak']:8.1f}")
    print(f"   Partite totali (P1/P2)   : {features['p1_matches_count']}/{features['p2_matches_count']}")
    print("-"*80)

    # Calcola h2h_ratio e form_volatility
    h2h_total = features['h2h_p1_wins'] + features['h2h_p2_wins']
    h2h_ratio = features['h2h_p1_wins'] / h2h_total if h2h_total > 0 else 0.5

    p1_form_volatility = 0
    p2_form_volatility = 0

    X = np.array([[
        features['player1_elo'],
        features['elo_diff'],
        features['p1_winrate_last5'],
        features['p2_winrate_last5'],
        features['h2h_p1_wins'],
        features['h2h_p2_wins'],
        h2h_ratio,
        features['p1_streak'],
        features['p2_streak'],
        p1_form_volatility,
        p2_form_volatility
    ]])

    X_scaled = models['scaler'].transform(X)

    # ========================================================================
    # Predizione con tutti i modelli
    # ========================================================================
    predictions = {}

    print("\n🤖 PREDIZIONI DEI MODELLI:")
    print("="*80)

    # Logistic Regression
    lr_pred = models['logistic'].predict(X_scaled)[0]
    lr_proba = models['logistic'].predict_proba(X_scaled)[0]
    predictions['logistic'] = {
        'winner': player1_name if lr_pred == 1 else player2_name,
        'confidence': lr_proba[1] if lr_pred == 1 else lr_proba[0]
    }
    print(f"📊 Logistic Regression:")
    print(f"   Winner: {predictions['logistic']['winner']}")
    print(f"   Confidence: {predictions['logistic']['confidence']*100:.1f}%")

    # Random Forest
    rf_pred = models['random_forest'].predict(X_scaled)[0]
    rf_proba = models['random_forest'].predict_proba(X_scaled)[0]
    predictions['random_forest'] = {
        'winner': player1_name if rf_pred == 1 else player2_name,
        'confidence': rf_proba[1] if rf_pred == 1 else rf_proba[0]
    }
    print(f"\n🌲 Random Forest:")
    print(f"   Winner: {predictions['random_forest']['winner']}")
    print(f"   Confidence: {predictions['random_forest']['confidence']*100:.1f}%")

    # Neural Network
    nn_pred = models['neural_network'].predict(X_scaled)[0]
    nn_proba = models['neural_network'].predict_proba(X_scaled)[0]
    predictions['neural_network'] = {
        'winner': player1_name if nn_pred == 1 else player2_name,
        'confidence': nn_proba[1] if nn_pred == 1 else nn_proba[0]
    }
    print(f"\n🧠 Neural Network:")
    print(f"   Winner: {predictions['neural_network']['winner']}")
    print(f"   Confidence: {predictions['neural_network']['confidence']*100:.1f}%")

    # Consensus
    print("\n" + "="*80)
    winners = [p['winner'] for p in predictions.values()]
    from collections import Counter
    winner_counts = Counter(winners)
    consensus_winner = winner_counts.most_common(1)[0][0]
    consensus_votes = winner_counts.most_common(1)[0][1]

    print(f"🏆 CONSENSUS (voto maggioranza):")
    print(f"   Winner: {consensus_winner}")
    print(f"   Voti: {consensus_votes}/3 modelli")

    avg_confidence = np.mean([p['confidence'] for p in predictions.values() 
                              if p['winner'] == consensus_winner])
    print(f"   Confidence media: {avg_confidence*100:.1f}%")
    print("="*80)

    return {
        'features': features,
        'predictions': predictions,
        'consensus': {
            'winner': consensus_winner,
            'votes': consensus_votes,
            'confidence': avg_confidence
        }
    }

# ============================================================================
# FUNZIONE 6: Interfaccia interattiva
# ============================================================================
def interactive_mode():
    """Modalità interattiva per predizioni multiple"""
    print("\n" + "="*80)
    print("🎾 TABLE TENNIS MATCH PREDICTOR")
    print("="*80)

    print("\n📂 Caricamento dataset...")
    try:
        matches_df = pd.read_csv('datasets/TT_Elite_COMBINED_9415_matches.csv')
        ranking_df = pd.read_csv('datasets/RANKING-TT-ELITE-SERIES.csv')
        print(f"   ✅ {len(matches_df)} partite caricate")
        print(f"   ✅ {len(ranking_df)} giocatori nel ranking")
    except FileNotFoundError as e:
        print(f"   ❌ Errore: {e}")
        return

    models = load_models()
    if models is None:
        return

    while True:
        print("\n" + "-"*80)
        player1 = input("\n👤 Nome Player 1 (o 'quit' per uscire): ").strip()
        if player1.lower() == 'quit':
            print("\n👋 Arrivederci!")
            break

        player2 = input("👤 Nome Player 2: ").strip()

        result = predict_winner(player1, player2, matches_df, ranking_df, models)

        if result:
            print("\n✅ Predizione completata!")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n🧪 TEST MODE: Esempio predizione")

    matches_df = pd.read_csv('datasets/TT_Elite_COMBINED_9415_matches.csv')
    ranking_df = pd.read_csv('datasets/RANKING-TT-ELITE-SERIES.csv')
    models = load_models()

    if models:
        # Esempio predizione
        predict_winner("Michal Olbrycht", "Jakub Zochniak", 
                      matches_df, ranking_df, models)

        print("\n\n💡 Vuoi provare altre predizioni?")
        choice = input("   Premi ENTER per modalità interattiva (o 'n' per uscire): ")
        if choice.lower() != 'n':
            interactive_mode()

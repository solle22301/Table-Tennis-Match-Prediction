"""
Modulo per la creazione delle features con rolling window.

Principio fondamentale: ZERO DATA LEAKAGE
Per ogni partita N, calcola features guardando SOLO partite da 1 a N-1.
Questo garantisce che il modello non "veda il futuro" durante il training.

Features create (12 totali - VERSIONE OTTIMIZZATA):
1. Ranking (3): ELO individuali, differenza (rimosso elo_sum - correlazione 0.01)
2. Win Rate (2): % vittorie ultime 5 (rimosso overall - VIF 17, ridondante)
3. Head-to-Head (3): vittorie P1, P2, ratio dominanza
4. Momentum (2): streak vittorie/sconfitte consecutive
5. Volatilità (2): consistenza performance ultime 5 partite
"""

import pandas as pd
import numpy as np

# ==============================================================================
# CREAZIONE FEATURES CON ROLLING WINDOW
# ==============================================================================

def create_features(df):
    """
    Crea tutte le features del modello usando rolling window.
    
    IMPORTANTE: Evita data leakage usando solo dati storici precedenti.
    
    Per ogni partita all'indice N:
    - Calcola statistiche usando SOLO partite da indice 0 a N-1
    - Aggiorna statistiche DOPO aver salvato le features
    
    Args:
        df (DataFrame): Dataset con colonne player_1, player_2, winner, 
                       player_1_elo, player_2_elo
    
    Returns:
        DataFrame: Dataset originale con 12 nuove colonne features
    
    Note:
        Operazione computazionalmente intensiva O(n²) ma necessaria
        per garantire zero data leakage. Progress ogni 1000 partite.
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    print("\nCreazione features con rolling window...")
    print("Nota: Per ogni partita uso SOLO dati storici precedenti")
    
    # --------------------------------------------------------------------------
    # Inizializzazione colonne features
    # --------------------------------------------------------------------------
    feature_cols = [
        # Ranking features (1 - rimosso elo_sum)
        'elo_diff',  # Differenza ELO: feature più predittiva
        
        # Win rate features (2 - rimosso overall, mantengo solo last5)
        'p1_win_rate_last5', 'p2_win_rate_last5',
        
        # Head-to-head features (3 - aggiunto h2h_ratio)
        'h2h_p1_wins', 'h2h_p2_wins', 'h2h_ratio',
        
        # Momentum features (2)
        'p1_streak', 'p2_streak',
        
        # Volatilità/Consistenza (2)
        'p1_form_volatility', 'p2_form_volatility'
    ]
    
    for col in feature_cols:
        df[col] = 0.0
    
    # --------------------------------------------------------------------------
    # Dizionari per tracking statistiche storiche
    # --------------------------------------------------------------------------
    
    # player_stats: statistiche per singolo giocatore
    # Struttura: {nome: {'wins': X, 'total': Y, 'last_results': [1,0,1], 'streak': Z}}
    player_stats = {}
    
    # h2h_stats: statistiche scontri diretti
    # Struttura: {(player_a, player_b): {'p1_wins': X, 'p2_wins': Y}}
    # Uso tupla ordinata (min, max) come chiave per simmetria
    h2h_stats = {}
    
    total_matches = len(df)
    
    # --------------------------------------------------------------------------
    # Loop principale: itera su ogni partita
    # --------------------------------------------------------------------------
    for idx, row in df.iterrows():
        p1 = row['player_1_clean']
        p2 = row['player_2_clean']
        winner = row['winner']
        
        # Progress indicator ogni 1000 partite
        if idx % 1000 == 0:
            print(f"  Processate {idx}/{total_matches} partite "
                  f"({idx/total_matches*100:.0f}%)")
        
        # Inizializza dizionari per nuovi giocatori
        if p1 not in player_stats:
            player_stats[p1] = {'wins': 0, 'total': 0, 'last_results': [], 'streak': 0}
        if p2 not in player_stats:
            player_stats[p2] = {'wins': 0, 'total': 0, 'last_results': [], 'streak': 0}
        
        # ----------------------------------------------------------------------
        # CATEGORIA 1: RANKING FEATURES (1 feature - ottimizzato)
        # ----------------------------------------------------------------------
        # Differenza ELO: LA feature più predittiva (correlazione 0.26 con target)
        # Misura la forza relativa tra i due giocatori
        # Positiva = P1 più forte, Negativa = P2 più forte
        df.at[idx, 'elo_diff'] = row['player_1_elo'] - row['player_2_elo']
        
        # NOTA: elo_sum RIMOSSA (correlazione 0.01, inutile)
        # player_1_elo e player_2_elo GIÀ presenti dal merge
        
        # ----------------------------------------------------------------------
        # CATEGORIA 2: WIN RATE LAST 5 (2 features - ottimizzato)
        # ----------------------------------------------------------------------
        # Percentuale vittorie ultime 5 partite (forma recente)
        # NOTA: win_rate_overall RIMOSSA (VIF 17, troppo correlata con last5)
        # NOTA: recent_form (ultime 3) RIMOSSA (VIF 15, ridondante con last5)
        
        # Player 1 - Win rate ultime 5
        if len(player_stats[p1]['last_results']) >= 5:
            # Se ha almeno 5 partite, prendi esattamente le ultime 5
            df.at[idx, 'p1_win_rate_last5'] = (
                sum(player_stats[p1]['last_results'][-5:]) / 5
            )
        elif len(player_stats[p1]['last_results']) > 0:
            # Se ha meno di 5 partite, usa tutte quelle disponibili
            df.at[idx, 'p1_win_rate_last5'] = (
                sum(player_stats[p1]['last_results']) / 
                len(player_stats[p1]['last_results'])
            )
        # else: rimane 0.0 (giocatore debutta)
        
        # Player 2 - Win rate ultime 5 (stessa logica)
        if len(player_stats[p2]['last_results']) >= 5:
            df.at[idx, 'p2_win_rate_last5'] = (
                sum(player_stats[p2]['last_results'][-5:]) / 5
            )
        elif len(player_stats[p2]['last_results']) > 0:
            df.at[idx, 'p2_win_rate_last5'] = (
                sum(player_stats[p2]['last_results']) / 
                len(player_stats[p2]['last_results'])
            )
        
        # ----------------------------------------------------------------------
        # CATEGORIA 3: HEAD-TO-HEAD (3 features - aggiunto h2h_ratio)
        # ----------------------------------------------------------------------
        # Storico scontri diretti tra i due giocatori specifici
        
        # Crea chiave ordinata per simmetria: sempre (min, max)
        # Es: (Alice, Bob) e (Bob, Alice) → stessa chiave (Alice, Bob)
        h2h_key = (p1, p2) if p1 < p2 else (p2, p1)
        
        if h2h_key in h2h_stats:
            # Gestisci inversione chiave per estrarre vittorie corrette
            if h2h_key == (p1, p2):
                # Chiave non invertita: p1 è effettivamente il primo
                df.at[idx, 'h2h_p1_wins'] = h2h_stats[h2h_key]['p1_wins']
                df.at[idx, 'h2h_p2_wins'] = h2h_stats[h2h_key]['p2_wins']
            else:
                # Chiave invertita: scambia i contatori
                df.at[idx, 'h2h_p1_wins'] = h2h_stats[h2h_key]['p2_wins']
                df.at[idx, 'h2h_p2_wins'] = h2h_stats[h2h_key]['p1_wins']
            
            # NUOVA FEATURE: H2H Ratio (rapporto normalizzato)
            # Misura la dominanza negli scontri diretti
            # Range: 0 (P1 ha sempre perso) a 1 (P1 ha sempre vinto)
            # 0.5 = equilibrio perfetto
            total_h2h = df.at[idx, 'h2h_p1_wins'] + df.at[idx, 'h2h_p2_wins']
            if total_h2h > 0:
                df.at[idx, 'h2h_ratio'] = df.at[idx, 'h2h_p1_wins'] / total_h2h
            # else: rimane 0.0 (mai affrontati prima)
        
        # ----------------------------------------------------------------------
        # CATEGORIA 4: MOMENTUM / STREAK (2 features)
        # ----------------------------------------------------------------------
        # Numero di vittorie/sconfitte consecutive
        # Valori positivi = streak vittorie, negativi = streak sconfitte
        # Es: +3 = 3 vittorie consecutive, -2 = 2 sconfitte consecutive
        df.at[idx, 'p1_streak'] = player_stats[p1]['streak']
        df.at[idx, 'p2_streak'] = player_stats[p2]['streak']
        
        # ----------------------------------------------------------------------
        # CATEGORIA 5: VOLATILITÀ/CONSISTENZA (2 features)
        # ----------------------------------------------------------------------
        # Deviazione standard dei risultati delle ultime 5 partite
        # Misura quanto è STABILE la performance del giocatore
        # Volatilità alta = altalenante (oggi vince, domani perde)
        # Volatilità bassa = consistente (performance prevedibile)
        
        # Player 1 - Form Volatility
        if len(player_stats[p1]['last_results']) >= 3:
            # Serve almeno 3 partite per calcolare std significativa
            last_results_p1 = player_stats[p1]['last_results'][-5:]
            df.at[idx, 'p1_form_volatility'] = np.std(last_results_p1)
        # else: rimane 0.0 (poche partite, volatilità indefinibile)
        
        # Player 2 - Form Volatility (stessa logica)
        if len(player_stats[p2]['last_results']) >= 3:
            last_results_p2 = player_stats[p2]['last_results'][-5:]
            df.at[idx, 'p2_form_volatility'] = np.std(last_results_p2)
        
        # ----------------------------------------------------------------------
        # AGGIORNAMENTO STATISTICHE (DOPO calcolo features - FONDAMENTALE!)
        # ----------------------------------------------------------------------
        # Questo blocco viene eseguito DOPO aver salvato tutte le features
        # così garantiamo che ogni partita usi SOLO info dal passato
        
        # Determina vincitore della partita corrente
        p1_won = 1 if winner == row['player_1'] else 0
        p2_won = 1 - p1_won
        
        # Update statistiche overall (totale vittorie e partite giocate)
        player_stats[p1]['wins'] += p1_won
        player_stats[p1]['total'] += 1
        player_stats[p1]['last_results'].append(p1_won)
        
        player_stats[p2]['wins'] += p2_won
        player_stats[p2]['total'] += 1
        player_stats[p2]['last_results'].append(p2_won)
        
        # Update streak (striscia vittorie/sconfitte consecutive)
        # Logica:
        # - Se P1 vince: incrementa streak positivo (o resetta da negativo a +1)
        # - Se P1 perde: incrementa streak negativo (o resetta da positivo a -1)
        if p1_won:
            player_stats[p1]['streak'] = max(0, player_stats[p1]['streak']) + 1
            player_stats[p2]['streak'] = min(0, player_stats[p2]['streak']) - 1
        else:
            player_stats[p2]['streak'] = max(0, player_stats[p2]['streak']) + 1
            player_stats[p1]['streak'] = min(0, player_stats[p1]['streak']) - 1
        
        # Update head-to-head (storico scontri diretti)
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {'p1_wins': 0, 'p2_wins': 0}
        
        # Incrementa contatore vittorie del vincitore
        # Gestisci correttamente chiave ordinata
        if h2h_key == (p1, p2):
            # Chiave non invertita
            if p1_won:
                h2h_stats[h2h_key]['p1_wins'] += 1
            else:
                h2h_stats[h2h_key]['p2_wins'] += 1
        else:
            # Chiave invertita: scambia contatori
            if p1_won:
                h2h_stats[h2h_key]['p2_wins'] += 1
            else:
                h2h_stats[h2h_key]['p1_wins'] += 1
    
    # --------------------------------------------------------------------------
    # Riepilogo finale
    # --------------------------------------------------------------------------
    print(f"\n✓ Features create: {len(feature_cols)} nuove features")
    print(f"  (+ player_1_elo e player_2_elo dal merge = {len(feature_cols) + 2} totali)")
    print(f"  Categorie ottimizzate:")
    print(f"    • Ranking: elo_diff (1)")
    print(f"    • Win Rate: last5 per P1 e P2 (2)")
    print(f"    • H2H: wins + ratio (3)")
    print(f"    • Momentum: streak (2)")
    print(f"    • Volatilità: form consistency (2)")
    print(f"\n  Features RIMOSSE per ottimizzazione:")
    print(f"    ✗ elo_sum (correlazione 0.01)")
    print(f"    ✗ win_rate_overall (VIF 17)")
    print(f"    ✗ recent_form (VIF 15)")
    print(f"    ✗ experience (correlazione 0.02)")
    
    return df

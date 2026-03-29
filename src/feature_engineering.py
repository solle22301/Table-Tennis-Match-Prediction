"""
Modulo per la creazione delle features predittive tramite Rolling Window.

Principio fondamentale: ZERO DATA LEAKAGE
Per calcolare le statistiche di una partita all'indice N, il sistema 
interroga ESCLUSIVAMENTE le partite comprese tra l'indice 0 e N-1.
Questo garantisce che il modello operi in uno stato di incertezza reale,
senza mai "vedere" i risultati futuri.

Le 10 Features create sono divise in 4 categorie logiche:
1. Ranking: Differenza matematica di ELO (elo_diff)
2. Stato di Forma: Win rate percentuale sulle ultime 5 partite
3. Scontri Diretti (Head-to-Head): Vittorie storiche assolute e ratio di supremazia
4. Inerzia Psicologica: Strisce di vittorie consecutive (streak) e volatilità
"""

import pandas as pd
import numpy as np

# ==============================================================================
# CREAZIONE FEATURES CON ROLLING WINDOW
# ==============================================================================

def create_features(df):
    """
    Costruisce la matrice delle features evitando il data leakage temporale.
    
    Flusso logico per ogni partita (N):
    1. Legge le statistiche storiche aggiornate fino alla partita N-1
    2. Salva queste statistiche come "features" per la partita N
    3. SCOPRE il risultato della partita N
    4. Aggiorna le statistiche storiche includendo il nuovo risultato
    
    Args:
        df (DataFrame): Dataset ordinato cronologicamente con colonne 
                        player_1, player_2, winner, player_1_elo, player_2_elo
    
    Returns:
        DataFrame: Il dataset originale arricchito con le 10 nuove features
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    print("\nCreazione features con approccio Rolling Window...")
    print("Garanzia Zero Data Leakage: in elaborazione...")
    
    # --------------------------------------------------------------------------
    # 1. Inizializzazione colonne features
    # --------------------------------------------------------------------------
    feature_cols = [
        'elo_diff',
        'p1_win_rate_last5', 'p2_win_rate_last5',
        'p1_head_to_head_wins', 'p2_head_to_head_wins', 'p1_head_to_head_win_ratio',
        'p1_streak', 'p2_streak',
        'p1_form_volatility', 'p2_form_volatility'
    ]
    
    for col in feature_cols:
        df[col] = 0.0
    
    # --------------------------------------------------------------------------
    # 2. Strutture dati per la memoria storica
    # --------------------------------------------------------------------------
    
    # player_stats: Dizionario che ricorda le performance passate di ogni singolo giocatore.
    # Salveremo quante partite ha vinto, quante ne ha giocate, la lista esatta dei risultati recenti e la sua striscia attuale.
    player_stats = {}
    
    # head_to_head_history: Dizionario che ricorda l'esito degli scontri diretti tra due specifici giocatori.
    # Nota: Usiamo sempre una tupla ordinata alfabeticamente (es. (Alice, Bob)) per evitare di contare la stessa coppia due volte.
    head_to_head_history = {}
    
    total_matches = len(df)
    
    # --------------------------------------------------------------------------
    # 3. Motore Rolling Window: Iterazione cronologica
    # --------------------------------------------------------------------------
    for idx, row in df.iterrows():
        # Estraiamo i nomi puliti e il vincitore della riga attuale
        player_1_name = row['player_1_clean']
        player_2_name = row['player_2_clean']
        match_winner = row['winner']
        
        if idx % 1000 == 0:
            print(f"   Processate {idx}/{total_matches} partite ({idx/total_matches*100:.0f}%)")
        
        # Se incontriamo questi giocatori per la prima volta, creiamo la loro "scheda" vuota
        if player_1_name not in player_stats:
            player_stats[player_1_name] = {'total_wins': 0, 'total_matches_played': 0, 'match_results_history': [], 'current_streak': 0}
        if player_2_name not in player_stats:
            player_stats[player_2_name] = {'total_wins': 0, 'total_matches_played': 0, 'match_results_history': [], 'current_streak': 0}
        
        # ======================================================================
        # FASE A: ESTRAZIONE FEATURES (Sulla base del solo passato)
        # ======================================================================
        
        # -- CATEGORIA 1: RANKING --
        # Differenza matematica tra i due ELO. Se è positiva, il P1 è favorito sulla carta.
        df.at[idx, 'elo_diff'] = row['player_1_elo'] - row['player_2_elo']
        
        # -- CATEGORIA 2: STATO DI FORMA RECENTE --
        # history_p1: Lista temporanea che contiene solo i risultati (1=vinta, 0=persa) delle partite passate del P1
        history_p1 = player_stats[player_1_name]['match_results_history']
        
        # Calcolo Win Rate per il Player 1
        if len(history_p1) >= 5:
            # Se ha giocato almeno 5 partite, calcoliamo la media esatta sulle ultime 5
            df.at[idx, 'p1_win_rate_last5'] = sum(history_p1[-5:]) / 5
        elif len(history_p1) > 0:
            # Se ne ha giocate meno di 5, facciamo la media su quelle disponibili
            df.at[idx, 'p1_win_rate_last5'] = sum(history_p1) / len(history_p1)
            
        # history_p2: Stessa lista temporanea per i risultati passati del P2
        history_p2 = player_stats[player_2_name]['match_results_history']
        
        # Calcolo Win Rate per il Player 2
        if len(history_p2) >= 5:
            df.at[idx, 'p2_win_rate_last5'] = sum(history_p2[-5:]) / 5
        elif len(history_p2) > 0:
            df.at[idx, 'p2_win_rate_last5'] = sum(history_p2) / len(history_p2)
            
        # -- CATEGORIA 3: SCONTRI DIRETTI (HEAD-TO-HEAD) --
        # players_matchup_key: È la chiave (tupla ordinata) per cercare questa esatta sfida nel dizionario storico
        players_matchup_key = (player_1_name, player_2_name) if player_1_name < player_2_name else (player_2_name, player_1_name)
        
        if players_matchup_key in head_to_head_history:
            if players_matchup_key == (player_1_name, player_2_name):
                # Se l'ordine alfabetico combacia con l'ordine delle colonne (P1, P2), assegniamo i valori direttamente
                df.at[idx, 'p1_head_to_head_wins'] = head_to_head_history[players_matchup_key]['p1_wins']
                df.at[idx, 'p2_head_to_head_wins'] = head_to_head_history[players_matchup_key]['p2_wins']
            else:
                # Altrimenti, scambiamo i valori per assegnare le vittorie al giocatore giusto
                df.at[idx, 'p1_head_to_head_wins'] = head_to_head_history[players_matchup_key]['p2_wins']
                df.at[idx, 'p2_head_to_head_wins'] = head_to_head_history[players_matchup_key]['p1_wins']
            
            # total_head_to_head_matches: Quante volte si sono già affrontati prima di oggi?
            total_head_to_head_matches = df.at[idx, 'p1_head_to_head_wins'] + df.at[idx, 'p2_head_to_head_wins']
            if total_head_to_head_matches > 0:
                # Calcoliamo la percentuale di dominio del Player 1 (es. 0.75 significa che P1 ha vinto il 75% delle volte)
                df.at[idx, 'p1_head_to_head_win_ratio'] = df.at[idx, 'p1_head_to_head_wins'] / total_head_to_head_matches
                
        # -- CATEGORIA 4: MOMENTUM E STREAK --
        # current_streak: Quante partite di fila sta vincendo (numero positivo) o perdendo (numero negativo) il giocatore?
        df.at[idx, 'p1_streak'] = player_stats[player_1_name]['current_streak']
        df.at[idx, 'p2_streak'] = player_stats[player_2_name]['current_streak']
        
        # -- CATEGORIA 5: VOLATILITÀ (CONSISTENZA) --
        # Calcoliamo la deviazione standard per capire se i risultati sono costanti o altalenanti.
        # Richiede un minimo di 3 partite giocate per avere un senso matematico.
        if len(history_p1) >= 3:
            df.at[idx, 'p1_form_volatility'] = np.std(history_p1[-5:])
            
        if len(history_p2) >= 3:
            df.at[idx, 'p2_form_volatility'] = np.std(history_p2[-5:])
        
        # ======================================================================
        # FASE B: AGGIORNAMENTO MEMORIA STORICA (Il presente diventa passato)
        # ======================================================================
        # NOTA BENE: Questo blocco aggiorna i dizionari e DEVE girare rigorosamente 
        # DOPO aver calcolato le features, per evitare che il modello veda il futuro.
        
        # is_player_1_winner: Variabile booleana che vale 1 se il Player 1 ha vinto, 0 se ha perso
        is_player_1_winner = 1 if match_winner == row['player_1'] else 0
        is_player_2_winner = 1 - is_player_1_winner
        
        # 1. Aggiorna contatori globali e storico risultati
        player_stats[player_1_name]['total_wins'] += is_player_1_winner
        player_stats[player_1_name]['total_matches_played'] += 1
        player_stats[player_1_name]['match_results_history'].append(is_player_1_winner)
        
        player_stats[player_2_name]['total_wins'] += is_player_2_winner
        player_stats[player_2_name]['total_matches_played'] += 1
        player_stats[player_2_name]['match_results_history'].append(is_player_2_winner)
        
        # 2. Aggiorna Momentum (Streak)
        if is_player_1_winner:
            # Se P1 vince: aumenta la sua striscia positiva. Azzera e manda in negativo quella del P2.
            player_stats[player_1_name]['current_streak'] = max(0, player_stats[player_1_name]['current_streak']) + 1
            player_stats[player_2_name]['current_streak'] = min(0, player_stats[player_2_name]['current_streak']) - 1
        else:
            player_stats[player_2_name]['current_streak'] = max(0, player_stats[player_2_name]['current_streak']) + 1
            player_stats[player_1_name]['current_streak'] = min(0, player_stats[player_1_name]['current_streak']) - 1
            
        # 3. Aggiorna storico scontri diretti
        if players_matchup_key not in head_to_head_history:
            head_to_head_history[players_matchup_key] = {'p1_wins': 0, 'p2_wins': 0}
            
        if players_matchup_key == (player_1_name, player_2_name):
            if is_player_1_winner: head_to_head_history[players_matchup_key]['p1_wins'] += 1
            else:                  head_to_head_history[players_matchup_key]['p2_wins'] += 1
        else:
            if is_player_1_winner: head_to_head_history[players_matchup_key]['p2_wins'] += 1
            else:                  head_to_head_history[players_matchup_key]['p1_wins'] += 1
            
    # --------------------------------------------------------------------------
    # Riepilogo finale
    # --------------------------------------------------------------------------
    print(f"\n✓ Features create con successo: {len(feature_cols)} variabili predittive")
    print(f"  (+ ELO dei singoli giocatori mantenuti dal merge)")
    print(f"\n  Categorie estratte:")
    print(f"    • Ranking: Differenza matematica (1)")
    print(f"    • Win Rate: Forma % sulle ultime 5 partite (2)")
    print(f"    • Head-to-Head: Scontri diretti e ratio di dominanza (3)")
    print(f"    • Momentum: Strisce di risultati consecutivi (2)")
    print(f"    • Volatilità: Consistenza delle performance (2)")
    
    return df
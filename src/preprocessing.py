"""
Modulo per la Normalizzazione Testuale e l'Unione dei Dataset (Merge).

Questo modulo si occupa di allineare i dati testuali tra lo storico partite 
e il ranking ufficiale.

IMPORTANTE: Il dataset del ranking è già stato pre-processato alla fonte 
e standardizzato nel formato "Nome Cognome". Non è quindi necessario 
applicare logiche di inversione stringhe durante il merge.
"""

import pandas as pd
import unicodedata

# ==============================================================================
# 1. NORMALIZZAZIONE TESTUALE
# ==============================================================================

def normalize_player_name(raw_name):
    """
    Standardizza una stringa di testo rimuovendo accenti, punteggiatura 
    e spazi anomali. Garantisce che varianti come "Michał" e "Michal" 
    vengano riconosciute come la stessa persona durante il join.
    """
    if pd.isna(raw_name):
        return raw_name
    
    # 1. Rimozione segni diacritici (accenti, cediglie, ecc.) tramite decomposizione Unicode
    normalized_string = ''.join(char for char in unicodedata.normalize('NFD', raw_name)
                                if unicodedata.category(char) != 'Mn')
    
    # 2. Pulizia punteggiatura e caratteri speciali
    normalized_string = normalized_string.replace('"', '').replace(',', '').replace('.', '').replace('-', ' ')
    
    # 3. Normalizzazione spaziature (rimuove spazi multipli)
    normalized_string = ' '.join(normalized_string.split())
    
    return normalized_string.strip()


# ==============================================================================
# 2. MERGE DEI DATASET
# ==============================================================================

def normalize_and_merge(matches_dataset, ranking_dataset):
    """
    Applica la normalizzazione a entrambi i dataset ed esegue una LEFT JOIN
    per associare il rating ELO a ciascun giocatore dello storico partite.
    """
    # Rimosso il print ridondante dell'intestazione (se ne occupa main.py)
    
    # -------------------------------------------------------------------------
    # Step 1: Normalizzazione delle colonne chiave
    # -------------------------------------------------------------------------
    print("\n   -> STEP 1: Pulizia e standardizzazione dei nomi...")
    matches_dataset['player_1_clean'] = matches_dataset['player_1'].apply(normalize_player_name)
    matches_dataset['player_2_clean'] = matches_dataset['player_2'].apply(normalize_player_name)
    ranking_dataset['normalized_name'] = ranking_dataset['Nome Giocatore'].apply(normalize_player_name)
    print("      ✓ Normalizzazione completata")
    
    # -------------------------------------------------------------------------
    # Step 2: Risoluzione Duplicati nel Ranking
    # -------------------------------------------------------------------------
    # Per evitare di moltiplicare le righe durante la JOIN a causa di omonimie 
    # o doppi inserimenti nel ranking ufficiale, forziamo l'univocità.
    print("\n   -> STEP 2: Controllo e risoluzione duplicati nel ranking...")
    duplicate_count = ranking_dataset['normalized_name'].duplicated().sum()
    print(f"      ✓ Anomalie (giocatori duplicati) risolte: {duplicate_count}")
    
    # In caso di duplicato, manteniamo il record con il Rating ELO più alto
    ranking_clean_dataset = (ranking_dataset
                             .sort_values('Rating ELO', ascending=False)
                             .drop_duplicates('normalized_name', keep='first'))
    
    # -------------------------------------------------------------------------
    # Step 3: Merge (LEFT JOIN sequenziale)
    # -------------------------------------------------------------------------
    print("\n   -> STEP 3: Esecuzione LEFT JOIN (Associazione ELO)...")
    
    # Associazione ELO per il Player 1
    merged_dataset = matches_dataset.merge(
        ranking_clean_dataset[['normalized_name', 'Rating ELO']],
        left_on='player_1_clean',
        right_on='normalized_name',
        how='left'
    ).rename(columns={'Rating ELO': 'player_1_elo'}).drop(columns=['normalized_name'])
    
    # Associazione ELO per il Player 2
    merged_dataset = merged_dataset.merge(
        ranking_clean_dataset[['normalized_name', 'Rating ELO']],
        left_on='player_2_clean',
        right_on='normalized_name',
        how='left'
    ).rename(columns={'Rating ELO': 'player_2_elo'}).drop(columns=['normalized_name'])
    
    # -------------------------------------------------------------------------
    # Statistiche di Copertura (Match Rate)
    # -------------------------------------------------------------------------
    total_matches = len(merged_dataset)
    p1_found_count = merged_dataset['player_1_elo'].notna().sum()
    p2_found_count = merged_dataset['player_2_elo'].notna().sum()
    matches_with_both_elos = ((merged_dataset['player_1_elo'].notna()) & 
                              (merged_dataset['player_2_elo'].notna())).sum()
    
    print(f"      ✓ Operazione di Merge completata con successo!")
    print(f"        Copertura ELO Player 1 : {p1_found_count}/{total_matches} ({p1_found_count/total_matches*100:.1f}%)")
    print(f"        Copertura ELO Player 2 : {p2_found_count}/{total_matches} ({p2_found_count/total_matches*100:.1f}%)")
    print(f"        Match validi (Entrambi): {matches_with_both_elos}/{total_matches} ({matches_with_both_elos/total_matches*100:.1f}%)")
    
    return merged_dataset
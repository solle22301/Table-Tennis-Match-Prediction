"""
Modulo per l'Acquisizione e la Validazione dei Dati (Data Loading).

Questo modulo costituisce il primo step della pipeline. Si occupa di:
1. Caricare i file CSV contenenti lo storico delle partite e i punteggi ELO.
2. Validare l'integrità strutturale dei file (presenza delle colonne obbligatorie).
3. Garantire l'ordine cronologico degli eventi (Fondamentale per prevenire il Data Leakage).
"""

import pandas as pd
import os

# ==============================================================================
# ACQUISIZIONE E VALIDAZIONE DATI
# ==============================================================================

def load_data(matches_file_path, ranking_file_path):
    """
    Carica i due dataset in memoria, ne verifica la correttezza strutturale 
    e imposta l'asse temporale corretto per le analisi successive.
    
    Args:
        matches_file_path (str): Percorso del file CSV contenente i risultati dei match.
        ranking_file_path (str): Percorso del file CSV contenente il ranking ELO ufficiale.
    
    Returns:
        tuple: (matches_dataset, ranking_dataset) - I due DataFrame pronti per il preprocessing.
    
    Raises:
        ValueError: Se i file risultano vuoti o privi delle colonne necessarie.
    """
    
    # --------------------------------------------------------------------------
    # 1. Caricamento e Validazione File Partite
    # --------------------------------------------------------------------------
    print(f"\n   -> Lettura file storico partite: {matches_file_path}")
    matches_dataset = pd.read_csv(matches_file_path)
    
    # Controllo di integrità: il file contiene dati?
    if len(matches_dataset) == 0:
        raise ValueError("ERRORE CRITICO: Il file delle partite è completamente vuoto.")
    
    # Controllo strutturale: ci sono tutte le informazioni necessarie?
    required_match_columns = ['player_1', 'player_2', 'winner', 'date', 
                              'player_1_sets_won', 'player_2_sets_won']
    missing_match_columns = [col for col in required_match_columns if col not in matches_dataset.columns]
    
    if missing_match_columns:
        raise ValueError(f"ERRORE CRITICO: Mancano colonne fondamentali nel file partite: {missing_match_columns}")
    
    # Cast del tipo di dato: converte la stringa di testo in un oggetto DateTime reale
    matches_dataset['date'] = pd.to_datetime(matches_dataset['date'])
    
    # ORDINAMENTO CRONOLOGICO (Cruciale per l'Ingegneria delle Features)
    # Assicura che la riga 0 sia la partita più vecchia in assoluto, in modo che la 
    # Rolling Window scorra il tempo in avanti senza mai rischiare di guardare al futuro.
# Corretto (Criterio di pareggio)
    matches_dataset = matches_dataset.sort_values(['date', 'match_id']).reset_index(drop=True)
    
    print(f"      ✓ File partite acquisito: {len(matches_dataset)} record elaborati.")
    print(f"      ✓ Finestra Temporale: {matches_dataset['date'].min().strftime('%Y-%m-%d')} → "
          f"{matches_dataset['date'].max().strftime('%Y-%m-%d')}")
    
    # --------------------------------------------------------------------------
    # 2. Caricamento e Validazione File Ranking ELO
    # --------------------------------------------------------------------------
    print(f"\n   -> Lettura file ranking ufficiale: {ranking_file_path}")
    ranking_dataset = pd.read_csv(ranking_file_path)
    
    # Controllo di integrità
    if len(ranking_dataset) == 0:
        raise ValueError("ERRORE CRITICO: Il file del ranking ELO è vuoto.")
    
    # Controllo strutturale
    required_ranking_columns = ['Nome Giocatore', 'Rating ELO']
    missing_ranking_columns = [col for col in required_ranking_columns if col not in ranking_dataset.columns]
    
    if missing_ranking_columns:
        raise ValueError(f"ERRORE CRITICO: Mancano colonne fondamentali nel file ranking: {missing_ranking_columns}")
    
    print(f"      ✓ File ranking acquisito: {len(ranking_dataset)} atleti censiti.")
    print(f"      ✓ Distribuzione ELO: {ranking_dataset['Rating ELO'].min():.0f} (Min) → "
          f"{ranking_dataset['Rating ELO'].max():.0f} (Max)")
    
    return matches_dataset, ranking_dataset
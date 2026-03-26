"""
Modulo per il caricamento e la validazione dei dataset.

Questo modulo gestisce:
- Caricamento file CSV
- Validazione colonne obbligatorie
- Conversione tipi di dato
- Ordinamento temporale
"""

import pandas as pd
import os


# ==============================================================================
# CARICAMENTO DATI
# ==============================================================================

def load_data(matches_file, ranking_file):
    """
    Carica i due dataset CSV del progetto.
    
    Args:
        matches_file (str): Path del file con le partite
        ranking_file (str): Path del file con i ranking ELO
    
    Returns:
        tuple: (df_matches, df_ranking) - DataFrame delle partite e del ranking
    
    Raises:
        ValueError: Se i file sono vuoti o mancano colonne obbligatorie
    """
    print("=" * 80)
    print("CARICAMENTO DATASET")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # Caricamento file partite
    # --------------------------------------------------------------------------
    print(f"\nCaricamento file partite: {matches_file}")
    df_matches = pd.read_csv(matches_file)
    
    # Validazione: file non vuoto
    if len(df_matches) == 0:
        raise ValueError("ERRORE: File partite vuoto")
    
    # Validazione: colonne obbligatorie presenti
    required_cols = ['player_1', 'player_2', 'winner', 'date', 
                     'player_1_sets_won', 'player_2_sets_won']
    missing = [col for col in required_cols if col not in df_matches.columns]
    if missing:
        raise ValueError(f"ERRORE: Colonne mancanti nel file partite: {missing}")
    
    # Conversione colonna date in datetime per ordinamento temporale
    df_matches['date'] = pd.to_datetime(df_matches['date'])
    
    # Ordinamento per data (IMPORTANTE per rolling features)
    df_matches = df_matches.sort_values('date').reset_index(drop=True)
    
    # Statistiche caricamento
    print(f"✓ File partite caricato: {len(df_matches)} righe")
    print(f"  Periodo: {df_matches['date'].min().strftime('%Y-%m-%d')} → "
          f"{df_matches['date'].max().strftime('%Y-%m-%d')}")
    
    # --------------------------------------------------------------------------
    # Caricamento file ranking
    # --------------------------------------------------------------------------
    print(f"\nCaricamento file ranking: {ranking_file}")
    df_ranking = pd.read_csv(ranking_file)
    
    # Validazione: file non vuoto
    if len(df_ranking) == 0:
        raise ValueError("ERRORE: File ranking vuoto")
    
    # Validazione: colonne obbligatorie presenti
    required_cols = ['Nome Giocatore', 'Rating ELO']
    missing = [col for col in required_cols if col not in df_ranking.columns]
    if missing:
        raise ValueError(f"ERRORE: Colonne mancanti nel file ranking: {missing}")
    
    # Statistiche caricamento
    print(f"✓ File ranking caricato: {len(df_ranking)} giocatori")
    print(f"  Range ELO: {df_ranking['Rating ELO'].min():.0f} → "
          f"{df_ranking['Rating ELO'].max():.0f}")
    
    return df_matches, df_ranking

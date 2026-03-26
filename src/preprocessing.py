"""
Modulo per la normalizzazione dei nomi e merge dei dataset.

IMPORTANTE: Il ranking è già stato pre-processato in formato "Nome Cognome"
quindi non serve più inversione!
"""

import pandas as pd
import unicodedata

def clean_name(name):
    """
    Pulisce un nome rimuovendo accenti e caratteri speciali.
    """
    if pd.isna(name):
        return name
    
    # Rimozione accenti
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                   if unicodedata.category(c) != 'Mn')
    
    # Rimozione caratteri speciali
    name = name.replace('"', '').replace(',', '').replace('.', '').replace('-', ' ')
    name = ' '.join(name.split())  # Normalizza spazi
    
    return name.strip()

def normalize_and_merge(df_matches, df_ranking):
    """
    Normalizza i nomi e unisce i dataset.
    NOTA: Nessuna inversione necessaria, entrambi usano formato "Nome Cognome"
    """
    print("\n" + "=" * 80)
    print("NORMALIZZAZIONE E MERGE")
    print("=" * 80)
    
    # STEP 1: Pulizia nomi
    print("\nSTEP 1: Pulizia nomi...")
    df_matches['player_1_clean'] = df_matches['player_1'].apply(clean_name)
    df_matches['player_2_clean'] = df_matches['player_2'].apply(clean_name)
    df_ranking['Nome_Clean'] = df_ranking['Nome Giocatore'].apply(clean_name)
    print("✓ Completata")
    
    # STEP 2: Risoluzione duplicati ranking
    print("\nSTEP 2: Risoluzione duplicati...")
    duplicates = df_ranking['Nome_Clean'].duplicated().sum()
    print(f"   Duplicati: {duplicates}")
    
    df_ranking_clean = (df_ranking
                       .sort_values('Rating ELO', ascending=False)
                       .drop_duplicates('Nome_Clean', keep='first'))
    
    # STEP 3: Merge LEFT JOIN
    print("\nSTEP 3: Merge...")
    df_merged = df_matches.merge(
        df_ranking_clean[['Nome_Clean', 'Rating ELO']],
        left_on='player_1_clean',
        right_on='Nome_Clean',
        how='left'
    ).rename(columns={'Rating ELO': 'player_1_elo'}).drop(columns=['Nome_Clean'])
    
    df_merged = df_merged.merge(
        df_ranking_clean[['Nome_Clean', 'Rating ELO']],
        left_on='player_2_clean',
        right_on='Nome_Clean',
        how='left'
    ).rename(columns={'Rating ELO': 'player_2_elo'}).drop(columns=['Nome_Clean'])
    
    # Statistiche
    p1_found = df_merged['player_1_elo'].notna().sum()
    p2_found = df_merged['player_2_elo'].notna().sum()
    both = ((df_merged['player_1_elo'].notna()) & 
            (df_merged['player_2_elo'].notna())).sum()
    
    print(f"\n✓ Merge completato!")
    print(f"   Player 1: {p1_found}/{len(df_merged)} ({p1_found/len(df_merged)*100:.1f}%)")
    print(f"   Player 2: {p2_found}/{len(df_merged)} ({p2_found/len(df_merged)*100:.1f}%)")
    print(f"   Entrambi: {both}/{len(df_merged)} ({both/len(df_merged)*100:.1f}%)")
    
    return df_merged

"""
Modulo per l'Exploratory Data Analysis (EDA).

Questo modulo implementa l'analisi esplorativa dei dati per:
- Verificare il bilanciamento delle classi (target)
- Analizzare la distribuzione dei rating ELO
- Identificare outliers con metodo IQR
- Studiare la relazione tra differenza ELO e probabilità di vittoria
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# ==============================================================================
# ANALISI 1: BILANCIAMENTO TARGET
# ==============================================================================

def analyze_target_balance(df):
    """
    Analizza il bilanciamento del dataset rispetto al target.
    
    Un dataset bilanciato (50%-50%) garantisce che il modello impari
    equamente da entrambe le classi. Se sbilanciato, il modello potrebbe
    semplicemente "imparare" a predire sempre la classe maggioritaria.
    
    Args:
        df (DataFrame): Dataset con colonna 'player_1_wins' (0 o 1)
    
    Returns:
        float: Baseline accuracy (percentuale classe maggioritaria)
    
    Output:
        - Stampa statistiche distribuzione
        - Salva grafico 'target_balance.png'
    """
    print("\n" + "=" * 80)
    print("ANALISI 1: BILANCIAMENTO TARGET")
    print("=" * 80)
    
    # Calcola distribuzione target
    target_counts = df['player_1_wins'].value_counts()
    target_pct = df['player_1_wins'].value_counts(normalize=True) * 100
    
    print(f"\nDistribuzione target:")
    print(f"   Player 1 vince (target=1): {target_counts[1]:,} ({target_pct[1]:.1f}%)")
    print(f"   Player 2 vince (target=0): {target_counts[0]:,} ({target_pct[0]:.1f}%)")
    
    # Baseline accuracy: accuracy ottenibile predicendo sempre classe maggioritaria
    baseline = target_pct.max()
    print(f"\nBaseline accuracy: {baseline:.1f}%")
    print(f"(Un modello che predice sempre la classe maggioritaria ottiene {baseline:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Visualizzazione grafico a barre
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    labels = ['Player 2 Vince', 'Player 1 Vince']
    colors = ['#FF6B6B', '#4ECDC4']  # Rosso per P2, Verde per P1
    
    plt.bar(labels, target_counts.values, color=colors, alpha=0.7, edgecolor='black')
    
    # Aggiungi etichette con valori sopra le barre
    for i, (count, pct) in enumerate(zip(target_counts.values, target_pct.values)):
        plt.text(i, count + 50, f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Linea orizzontale per bilanciamento perfetto 50%
    plt.axhline(y=len(df)/2, color='gray', linestyle='--', 
                linewidth=1.5, label='Bilanciamento perfetto (50%)')
    
    plt.xlabel('Classe Target', fontsize=11)
    plt.ylabel('Numero Partite', fontsize=11)
    plt.title('Bilanciamento Classi Target', fontweight='bold', fontsize=13)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Salva grafico
    plt.savefig('target_balance.png', dpi=150)
    print("\n✓ Grafico salvato: target_balance.png")
    plt.show()
    
    return baseline


# ==============================================================================
# ANALISI 2: DISTRIBUZIONE RATING ELO
# ==============================================================================

def analyze_elo_distributions(df):
    """
    Analizza la distribuzione dei rating ELO dei giocatori.
    
    Obiettivi:
    - Verificare che Player 1 e Player 2 abbiano distribuzioni simili
      (non ci dovrebbe essere bias sistematico)
    - Identificare outliers (giocatori con rating anomali)
    - Visualizzare range e concentrazione dei rating
    
    Args:
        df (DataFrame): Dataset con colonne 'player_1_elo', 'player_2_elo'
    
    Output:
        - Stampa statistiche outliers
        - Salva grafico 'elo_distributions.png' con istogrammi e boxplot
    """
    print("\n" + "=" * 80)
    print("ANALISI 2: DISTRIBUZIONE RATING ELO")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Subplot 1: Istogrammi sovrapposti
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Istogrammi sovrapposti per confronto visivo
    axes[0].hist(df['player_1_elo'], bins=30, alpha=0.6, color='#4ECDC4',
                label='Player 1', edgecolor='black')
    axes[0].hist(df['player_2_elo'], bins=30, alpha=0.6, color='#FF6B6B',
                label='Player 2', edgecolor='black')
    
    axes[0].set_xlabel('Rating ELO', fontsize=11)
    axes[0].set_ylabel('Frequenza', fontsize=11)
    axes[0].set_title('Distribuzione Rating ELO', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Subplot 2: Boxplot comparativo
    # -------------------------------------------------------------------------
    # Boxplot utile per identificare outliers visivamente
    # I "baffi" si estendono a 1.5*IQR, punti oltre sono outliers
    
    # Prepara dati in formato long per seaborn
    elo_data = pd.DataFrame({
        'ELO': list(df['player_1_elo']) + list(df['player_2_elo']),
        'Player': ['Player 1'] * len(df) + ['Player 2'] * len(df)
    })
    
    sns.boxplot(data=elo_data, x='Player', y='ELO', 
                palette=['#4ECDC4', '#FF6B6B'], ax=axes[1])
    axes[1].set_title('Boxplot ELO (Identificazione Outliers)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elo_distributions.png', dpi=150)
    print("\n✓ Grafico salvato: elo_distributions.png")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Identificazione outliers con metodo IQR
    # -------------------------------------------------------------------------
    # Metodo IQR (Interquartile Range):
    # - Q1 = 25° percentile, Q3 = 75° percentile
    # - IQR = Q3 - Q1
    # - Outliers: valori < Q1 - 1.5*IQR o > Q3 + 1.5*IQR
    
    print("\nIdentificazione outliers (metodo IQR):")
    
    for col, label in [('player_1_elo', 'Player 1'), ('player_2_elo', 'Player 2')]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calcola limiti
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identifica outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        print(f"\n   {label}:")
        print(f"      Q1 = {Q1:.0f}, Q3 = {Q3:.0f}, IQR = {IQR:.0f}")
        print(f"      Range normale: [{lower_bound:.0f}, {upper_bound:.0f}]")
        print(f"      Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")


# ==============================================================================
# ANALISI 3: RELAZIONE ELO DIFFERENCE vs VITTORIA
# ==============================================================================

def analyze_elo_vs_win(df):
    """
    Analizza la relazione tra differenza ELO e probabilità di vittoria.
    
    Ipotesi da verificare:
    - Se Player 1 ha ELO maggiore (+100, +200, ecc.), dovrebbe vincere più spesso
    - La relazione dovrebbe essere monotona crescente
    - Serve per validare che il rating ELO sia effettivamente predittivo
    
    Args:
        df (DataFrame): Dataset con colonne 'elo_diff', 'player_1_wins'
    
    Output:
        - Stampa tabella win rate per bin
        - Salva grafico 'elo_diff_vs_win.png' con bar plot e scatter
    """
    print("\n" + "=" * 80)
    print("ANALISI 3: RELAZIONE ELO DIFFERENCE vs VITTORIA")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Creazione bin per differenza ELO
    # -------------------------------------------------------------------------
    # Dividiamo la differenza ELO in range significativi
    bins = [-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf]
    labels = ['<-200', '-200/-100', '-100/-50', '-50/0', 
              '0/50', '50/100', '100/200', '>200']
    
    df['elo_diff_bin'] = pd.cut(df['elo_diff'], bins=bins, labels=labels)
    
    # Calcola win rate per ogni bin
    win_rate_by_bin = df.groupby('elo_diff_bin', observed=True)['player_1_wins'].agg(['mean', 'count'])
    win_rate_by_bin['win_rate_pct'] = win_rate_by_bin['mean'] * 100
    
    # Stampa tabella
    print("\nWin rate Player 1 per range ELO difference:")
    print("\nRange ELO     Win Rate    N. Partite")
    print("-" * 45)
    for idx, row in win_rate_by_bin.iterrows():
        print(f"{str(idx):12s}  {row['win_rate_pct']:6.1f}%     {int(row['count']):6d}")
    print("-" * 45)
    
    # -------------------------------------------------------------------------
    # Visualizzazione: Bar plot + Scatter plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ----- Subplot 1: Bar plot win rate per bin -----
    # Colora barre in base a win rate (rosso <50%, verde >50%)
    colors = ['#FF4444' if wr < 50 else '#44FF44' 
              for wr in win_rate_by_bin['win_rate_pct']]
    
    bars = axes[0].bar(range(len(win_rate_by_bin)), 
                      win_rate_by_bin['win_rate_pct'],
                      color=colors, alpha=0.7, edgecolor='black')
    
    # Linea riferimento 50% (vittoria casuale)
    axes[0].axhline(y=50, color='black', linestyle='--', 
                   linewidth=2, label='50% (Random)')
    
    axes[0].set_xticks(range(len(win_rate_by_bin)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_xlabel('ELO Difference (Player 1 - Player 2)', fontsize=11)
    axes[0].set_ylabel('Win Rate Player 1 (%)', fontsize=11)
    axes[0].set_title('Win Rate per Range ELO Difference', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Aggiungi percentuali sopra le barre
    for i, (bar, pct) in enumerate(zip(bars, win_rate_by_bin['win_rate_pct'])):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # ----- Subplot 2: Scatter plot con trend line -----
    # Scatter plot mostra ogni singola partita
    # Trend line mostra relazione media
    
    # Calcola trend line (media mobile su 20 bin)
    elo_diff_bins = pd.cut(df['elo_diff'], bins=20)
    trend_data = df.groupby(elo_diff_bins, observed=True)['player_1_wins'].mean()
    bin_centers = [interval.mid for interval in trend_data.index]
    
    # Scatter plot (alpha basso per gestire sovrapposizione)
    axes[1].scatter(df['elo_diff'], df['player_1_wins'], 
                   alpha=0.05, s=5, color='gray', label='Partite individuali')
    
    # Trend line
    axes[1].plot(bin_centers, trend_data.values, 
                color='red', linewidth=3, label='Trend medio')
    
    # Linea riferimento 50%
    axes[1].axhline(y=0.5, color='black', linestyle='--', 
                   linewidth=1.5, label='50%')
    
    axes[1].set_xlabel('ELO Difference (Player 1 - Player 2)', fontsize=11)
    axes[1].set_ylabel('Probabilità Vittoria Player 1', fontsize=11)
    axes[1].set_title('Scatter Plot: ELO Difference vs Vittoria', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elo_diff_vs_win.png', dpi=150)
    print("\n✓ Grafico salvato: elo_diff_vs_win.png")
    plt.show()


# ==============================================================================
# FUNZIONE WRAPPER: ESEGUE EDA COMPLETA
# ==============================================================================

def run_eda(df_clean):
    """
    Esegue tutte le analisi esplorative in sequenza.
    
    Pipeline:
    1. Crea colonne necessarie (player_1_wins, elo_diff)
    2. Analizza bilanciamento target
    3. Analizza distribuzione ELO
    4. Analizza relazione ELO vs vittoria
    
    Args:
        df_clean (DataFrame): Dataset pulito con colonne player_1_elo, 
                             player_2_elo, winner, player_1
    
    Returns:
        float: Baseline accuracy per confronto con modelli
    """
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Crea colonna target binario se non esiste
    # player_1_wins = 1 se vince Player 1, 0 altrimenti
    if 'player_1_wins' not in df_clean.columns:
        df_clean['player_1_wins'] = (df_clean['winner'] == df_clean['player_1']).astype(int)
    
    # Crea colonna differenza ELO se non esiste
    if 'elo_diff' not in df_clean.columns:
        df_clean['elo_diff'] = df_clean['player_1_elo'] - df_clean['player_2_elo']
    
    # Esegui analisi in sequenza
    baseline = analyze_target_balance(df_clean)
    analyze_elo_distributions(df_clean)
    analyze_elo_vs_win(df_clean)
    
    print("\n✓ EDA completata con successo")
    
    return baseline

# ==============================================================================
# OUTLIER DETECTION (Sezione 4.6 appunti corso)
# ==============================================================================

def detect_outliers(X, y, feature_names=None, verbose=True):
    """
    Rileva outliers usando metodi del corso (Sezione 4.6).
    
    Implementa 3 criteri:
    1. Studentized Residuals: |r_i*| > 3 → outlier nel target
    2. Leverage (Hat matrix): h_ii > 3*D/N → High Leverage Point
    3. DFFITS: |DFFITS_i| > 2*sqrt(D/N) → High Influence Point
    
    Args:
        X: Features (numpy array o DataFrame)
        y: Target (numpy array o Series)
        feature_names: Nomi features (opzionale)
        verbose: Stampa report (default True)
    
    Returns:
        dict con:
            - 'outlier_indices': indici sample da rimuovere
            - 'studentized_residuals': residui studentizzati
            - 'leverage': valori leverage
            - 'dffits': valori DFFITS
    """
    if verbose:
        print("\n" + "=" * 80)
        print("OUTLIER DETECTION (Sezione 4.6 Appunti)")
        print("=" * 80)
    
    N, D = X.shape
    
    # -------------------------------------------------------------------------
    # 1. Calcolo Hat Matrix H = X(X^T X)^-1 X^T
    # -------------------------------------------------------------------------
    # Aggiungi colonna di 1 per intercetta (bias)
    X_with_bias = np.column_stack([np.ones(N), X])
    
    # Hat matrix (da appunti: H mette il "cappello" su y)
    try:
        XtX_inv = np.linalg.inv(X_with_bias.T @ X_with_bias)
        H = X_with_bias @ XtX_inv @ X_with_bias.T
        leverage = np.diag(H)  # h_ii = leverage di ogni sample
    except np.linalg.LinAlgError:
        if verbose:
            print("⚠️  Matrice singolare, uso pseudo-inversa")
        XtX_inv = np.linalg.pinv(X_with_bias.T @ X_with_bias)
        H = X_with_bias @ XtX_inv @ X_with_bias.T
        leverage = np.diag(H)
    
    # -------------------------------------------------------------------------
    # 2. Calcolo Residui e Studentized Residuals
    # -------------------------------------------------------------------------
    # Fit modello lineare per calcolare residui
    w = XtX_inv @ X_with_bias.T @ y
    y_pred = X_with_bias @ w
    residuals = y - y_pred
    
    # MSE (Mean Squared Error)
    mse = np.sum(residuals**2) / (N - D - 1)
    
    # Studentized residuals: r_i* = e_i / (sigma * sqrt(1 - h_ii))
    # dove sigma^2 = MSE
    studentized_residuals = residuals / (np.sqrt(mse) * np.sqrt(1 - leverage))
    
    # -------------------------------------------------------------------------
    # 3. Calcolo DFFITS (Difference in Fits)
    # -------------------------------------------------------------------------
    # DFFITS_i misura quanto cambia la predizione se rimuovo sample i
    # Formula: DFFITS_i = r_i* * sqrt(h_ii / (1 - h_ii))
    dffits = studentized_residuals * np.sqrt(leverage / (1 - leverage))
    
    # -------------------------------------------------------------------------
    # 4. Criteri di rilevamento (da appunti corso)
    # -------------------------------------------------------------------------
    # Criterio 1: Outlier nel target
    outliers_residuals = np.abs(studentized_residuals) > 3
    
    # Criterio 2: High Leverage Point (HLP)
    leverage_threshold = 3 * (D + 1) / N
    high_leverage = leverage > leverage_threshold
    
    # Criterio 3: High Influence Point (HIP)
    dffits_threshold = 2 * np.sqrt((D + 1) / N)
    high_influence = np.abs(dffits) > dffits_threshold
    
    # Combina criteri: rimuovi se è OUTLIER E (HLP O HIP)
    # Logica: outlier da solo non è problema, ma se ha anche high leverage/influence
    # allora distorce il modello
    outliers_combined = outliers_residuals & (high_leverage | high_influence)
    
    outlier_indices = np.where(outliers_combined)[0]
    
    # -------------------------------------------------------------------------
    # 5. Report
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\nDataset: N={N} samples, D={D} features")
        print(f"\nSoglie (da appunti corso):")
        print(f"   Studentized Residuals: |r_i*| > 3")
        print(f"   Leverage: h_ii > {leverage_threshold:.4f}")
        print(f"   DFFITS: |DFFITS_i| > {dffits_threshold:.4f}")
        
        print(f"\nRisultati:")
        print(f"   Outliers (residui): {outliers_residuals.sum()} ({outliers_residuals.sum()/N*100:.1f}%)")
        print(f"   High Leverage Points: {high_leverage.sum()} ({high_leverage.sum()/N*100:.1f}%)")
        print(f"   High Influence Points: {high_influence.sum()} ({high_influence.sum()/N*100:.1f}%)")
        print(f"   → Outliers da rimuovere: {len(outlier_indices)} ({len(outlier_indices)/N*100:.1f}%)")
        
        if len(outlier_indices) > 0 and len(outlier_indices) < 10:
            print(f"\nIndici outliers: {outlier_indices.tolist()}")
        
        # Top 5 outliers più influenti
        top_influence_idx = np.argsort(np.abs(dffits))[-5:][::-1]
        print(f"\nTop 5 sample più influenti (DFFITS):")
        print(f"   {'Index':<8} {'DFFITS':<12} {'Leverage':<12} {'Stud.Res.':<12}")
        print(f"   {'-'*50}")
        for idx in top_influence_idx:
            print(f"   {idx:<8} {dffits[idx]:<12.4f} {leverage[idx]:<12.4f} {studentized_residuals[idx]:<12.4f}")
    
    return {
        'outlier_indices': outlier_indices,
        'studentized_residuals': studentized_residuals,
        'leverage': leverage,
        'dffits': dffits,
        'summary': {
            'n_outliers': len(outlier_indices),
            'n_high_leverage': high_leverage.sum(),
            'n_high_influence': high_influence.sum()
        }
    }


def remove_outliers(df, outlier_indices, verbose=True):
    """
    Rimuove outliers dal DataFrame.
    
    Args:
        df: DataFrame
        outlier_indices: Indici da rimuovere
        verbose: Stampa report
    
    Returns:
        DataFrame pulito
    """
    if len(outlier_indices) == 0:
        if verbose:
            print("\n✓ Nessun outlier da rimuovere")
        return df
    
    df_clean = df.drop(df.index[outlier_indices]).reset_index(drop=True)
    
    if verbose:
        print(f"\n✓ Rimossi {len(outlier_indices)} outliers")
        print(f"   Dataset: {len(df)} → {len(df_clean)} samples "
              f"(-{len(outlier_indices)/len(df)*100:.1f}%)")
    
    return df_clean




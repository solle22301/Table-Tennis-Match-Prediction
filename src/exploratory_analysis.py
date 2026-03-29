"""
Modulo per l'Exploratory Data Analysis (EDA) e la Pulizia Statistica.

Questo modulo è progettato per ispezionare la qualità dei dati prima dell'addestramento:
1. Bilanciamento Target: Verifica se una classe domina sull'altra.
2. Distribuzione ELO: Analisi della gaussiana dei punteggi dei giocatori.
3. Potere Predittivo: Relazione visiva tra Differenza ELO e probabilità di vittoria.
4. Outlier Detection (Metodi Robust Statistics): Identificazione e rimozione 
   di sample anomali che potrebbero distorcere i modelli (Studentized Residuals, Leverage, DFFITS).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os



# ==============================================================================
# ANALISI 1: BILANCIAMENTO TARGET
# ==============================================================================

def analyze_target_balance(df):
    """
    Analizza la distribuzione delle vittorie tra Player 1 e Player 2.
    
    Perché è importante:
    Un dataset bilanciato (~50%) permette di usare l'Accuracy come metrica affidabile.
    Se fosse sbilanciato (es. 90%-10%), un modello "stupido" che predice sempre
    la classe maggioritaria otterrebbe il 90% senza imparare nulla.
    
    Args:
        df (DataFrame): Dataset contenente la colonna 'player_1_wins'
    
    Returns:
        float: Accuracy della Baseline (percentuale della classe maggioritaria)
    """
    print("\n" + "=" * 80)
    print("ANALISI 1: BILANCIAMENTO DELLE CLASSI (TARGET)")
    print("=" * 80)
    
    # Calcolo frequenze assolute e percentuali
    win_counts = df['player_1_wins'].value_counts()
    win_percentages = df['player_1_wins'].value_counts(normalize=True) * 100
    
    print(f"\nDistribuzione del Target:")
    print(f"   Vittorie Player 1 (Target=1): {win_counts[1]:,} ({win_percentages[1]:.1f}%)")
    print(f"   Vittorie Player 2 (Target=0): {win_counts[0]:,} ({win_percentages[0]:.1f}%)")
    
    # La baseline è la percentuale più alta. Questo è il nostro "Punto Zero" da battere.
    majority_class_baseline = win_percentages.max()
    print(f"\nBaseline Accuracy: {majority_class_baseline:.1f}%")
    print(f"(Significa che tirando a indovinare sempre la classe maggioritaria si ottiene {majority_class_baseline:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Visualizzazione Grafica
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    class_labels = ['Vittoria Player 2', 'Vittoria Player 1']
    bar_colors = ['#FF6B6B', '#4ECDC4']  # Rosso per P2, Verde per P1
    
    bars = plt.bar(class_labels, win_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
    
    # Aggiunge le etichette di testo sopra ogni barra
    for index, (count, percentage) in enumerate(zip(win_counts.values, win_percentages.values)):
        plt.text(index, count + 50, f'{count:,}\n({percentage:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Linea tratteggiata per indicare il bilanciamento perfetto
    plt.axhline(y=len(df)/2, color='gray', linestyle='--', linewidth=1.5, label='Bilanciamento Perfetto (50%)')
    
    plt.xlabel('Esito Partita', fontsize=11)
    plt.ylabel('Numero di Partite', fontsize=11)
    plt.title('Bilanciamento delle Classi Target', fontweight='bold', fontsize=13)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    path_grafico = os.path.join("plots", "target_balance.png")
    plt.savefig(path_grafico, dpi=150)
    print("\n✓ Grafico esportato: target_balance.png")
    plt.show()
    
    return majority_class_baseline


# ==============================================================================
# ANALISI 2: DISTRIBUZIONE RATING ELO E OUTLIERS BASE
# ==============================================================================

def analyze_elo_distributions(df):
    """
    Analizza visivamente la distribuzione dei punteggi ELO e individua
    valori anomali (outliers unidimensionali) tramite il metodo IQR.
    """
    print("\n" + "=" * 80)
    print("ANALISI 2: DISTRIBUZIONE STATISTICA DEL RATING ELO")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # -- Subplot 1: Istogrammi Sovrapposti --
    axes[0].hist(df['player_1_elo'], bins=30, alpha=0.6, color='#4ECDC4', label='Player 1 ELO', edgecolor='black')
    axes[0].hist(df['player_2_elo'], bins=30, alpha=0.6, color='#FF6B6B', label='Player 2 ELO', edgecolor='black')
    axes[0].set_xlabel('Punteggio ELO', fontsize=11)
    axes[0].set_ylabel('Frequenza Assoluta', fontsize=11)
    axes[0].set_title('Distribuzione Gaussiana dei Rating ELO', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # -- Subplot 2: Boxplot Comparativo (Outlier visivi) --
    flattened_elo_data = pd.DataFrame({
        'ELO_Score': list(df['player_1_elo']) + list(df['player_2_elo']),
        'Player_Role': ['Player 1'] * len(df) + ['Player 2'] * len(df)
    })
    
    sns.boxplot(data=flattened_elo_data, x='Player_Role', y='ELO_Score', palette=['#4ECDC4', '#FF6B6B'], ax=axes[1])
    axes[1].set_title('Boxplot ELO (Rilevamento Outliers)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path_grafico = os.path.join("plots", "elo_distributions.png")
    plt.savefig(path_grafico, dpi=150)
    print("\n✓ Grafico esportato: elo_distributions.png")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Calcolo matematico Outliers (Metodo IQR - Interquartile Range)
    # -------------------------------------------------------------------------
    print("\nRicerca Outliers tramite IQR (Interquartile Range):")
    
    for column_name, display_label in [('player_1_elo', 'Player 1'), ('player_2_elo', 'Player 2')]:
        first_quartile = df[column_name].quantile(0.25)
        third_quartile = df[column_name].quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range
        
        outliers_found = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
        
        print(f"\n   {display_label}:")
        print(f"      Q1 = {first_quartile:.0f}, Q3 = {third_quartile:.0f}, IQR = {interquartile_range:.0f}")
        print(f"      Range di normalità: [{lower_bound:.0f}, {upper_bound:.0f}]")
        print(f"      Outliers identificati: {len(outliers_found)} ({len(outliers_found)/len(df)*100:.1f}%)")


# ==============================================================================
# ANALISI 3: VERIFICA POTERE PREDITTIVO ELO
# ==============================================================================

def analyze_elo_vs_win(df):
    """
    Verifica se all'aumentare della differenza di ELO a favore del Player 1,
    aumenta effettivamente la sua probabilità di vittoria (Trend Monotono).
    """
    print("\n" + "=" * 80)
    print("ANALISI 3: RELAZIONE DIFFERENZA ELO vs PROBABILITÀ DI VITTORIA")
    print("=" * 80)
    
    # Creazione di "secchielli" (bin) per raggruppare le differenze ELO
    elo_bins = [-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf]
    bin_labels = ['<-200', '-200/-100', '-100/-50', '-50/0', '0/50', '50/100', '100/200', '>200']
    
    df['elo_difference_category'] = pd.cut(df['elo_diff'], bins=elo_bins, labels=bin_labels)
    
    # Calcolo del Win Rate per ogni categoria
    win_rate_by_category = df.groupby('elo_difference_category', observed=True)['player_1_wins'].agg(['mean', 'count'])
    win_rate_by_category['win_rate_percentage'] = win_rate_by_category['mean'] * 100
    
    print("\nWin rate Player 1 per fasce di divario ELO:")
    print("\nFascia ELO       Win Rate    N. Partite")
    print("-" * 45)
    for category_idx, row in win_rate_by_category.iterrows():
        print(f"{str(category_idx):12s}  {row['win_rate_percentage']:6.1f}%     {int(row['count']):6d}")
    print("-" * 45)
    
    # -------------------------------------------------------------------------
    # Visualizzazione: Bar Plot + Scatter
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # -- Subplot 1: Bar Plot --
    bar_colors = ['#FF4444' if rate < 50 else '#44FF44' for rate in win_rate_by_category['win_rate_percentage']]
    
    bars = axes[0].bar(range(len(win_rate_by_category)), win_rate_by_category['win_rate_percentage'],
                       color=bar_colors, alpha=0.7, edgecolor='black')
    
    axes[0].axhline(y=50, color='black', linestyle='--', linewidth=2, label='Random Guess (50%)')
    axes[0].set_xticks(range(len(win_rate_by_category)))
    axes[0].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[0].set_xlabel('Vantaggio ELO (Player 1 - Player 2)', fontsize=11)
    axes[0].set_ylabel('Win Rate Player 1 (%)', fontsize=11)
    axes[0].set_title('Tasso di Vittoria per Fascia ELO', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, percentage in zip(bars, win_rate_by_category['win_rate_percentage']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # -- Subplot 2: Scatter Plot & Trend Line --
    dynamic_bins = pd.cut(df['elo_diff'], bins=20)
    trend_line_data = df.groupby(dynamic_bins, observed=True)['player_1_wins'].mean()
    bin_centers = [interval.mid for interval in trend_line_data.index]
    
    axes[1].scatter(df['elo_diff'], df['player_1_wins'], alpha=0.05, s=5, color='gray', label='Partite individuali')
    axes[1].plot(bin_centers, trend_line_data.values, color='red', linewidth=3, label='Trend Line (Media)')
    axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Soglia 50%')
    
    axes[1].set_xlabel('Vantaggio ELO (Player 1 - Player 2)', fontsize=11)
    axes[1].set_ylabel('Probabilità di Vittoria Player 1', fontsize=11)
    axes[1].set_title('Scatter Plot: Vantaggio ELO vs Esito', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    path_grafico = os.path.join("plots", "elo_diff_vs_win.png")
    plt.savefig(path_grafico, dpi=150)
    print("\n✓ Grafico esportato: elo_diff_vs_win.png")
    plt.show()


# ==============================================================================
# FUNZIONE WRAPPER: ESECUZIONE PIPELINE EDA
# ==============================================================================

def run_eda(df_clean):
    """Esegue le analisi esplorative in sequenza e restituisce la baseline."""
    print("=" * 80)
    print("AVVIO EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    if 'player_1_wins' not in df_clean.columns:
        df_clean['player_1_wins'] = (df_clean['winner'] == df_clean['player_1']).astype(int)
    
    if 'elo_diff' not in df_clean.columns:
        df_clean['elo_diff'] = df_clean['player_1_elo'] - df_clean['player_2_elo']
    
    baseline_accuracy = analyze_target_balance(df_clean)
    analyze_elo_distributions(df_clean)
    analyze_elo_vs_win(df_clean)
    
    print("\n✓ Pipeline EDA completata con successo")
    return baseline_accuracy

# ==============================================================================
# ADVANCED OUTLIER DETECTION (Basato su Appunti Sezione 4.6)
# ==============================================================================

def detect_outliers(feature_matrix_X, target_vector_y, feature_names=None, verbose=True):
    """
    Rileva anomalie multivariate usando l'algebra lineare (Hat Matrix e Residui).
    
    Criteri implementati:
    1. Studentized Residuals (Outlier nel Target): Errore di predizione anomalo.
    2. Leverage (High Leverage Point - HLP): Valori anomali nello spazio delle features.
    3. DFFITS (High Influence Point - HIP): Impatto estremo sul modello se rimosso.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("OUTLIER DETECTION AVANZATA (Rif. Sezione 4.6 Appunti Universitari)")
        print("=" * 80)
    
    num_samples, num_features = feature_matrix_X.shape
    
    # -------------------------------------------------------------------------
    # 1. Hat Matrix e Leverage
    # -------------------------------------------------------------------------
    # Aggiungiamo una colonna di '1' per l'intercetta (bias) del modello lineare
    feature_matrix_with_intercept = np.column_stack([np.ones(num_samples), feature_matrix_X])
    
    try:
        # Calcolo dell'inversa della matrice di covarianza (X^T * X)^-1
        inverse_covariance_matrix = np.linalg.inv(feature_matrix_with_intercept.T @ feature_matrix_with_intercept)
        # La Hat Matrix (H) proietta le Y reali sulle Y predette
        hat_matrix = feature_matrix_with_intercept @ inverse_covariance_matrix @ feature_matrix_with_intercept.T
        leverage_scores = np.diag(hat_matrix)
    except np.linalg.LinAlgError:
        if verbose: print("⚠️ Matrice singolare rilevata, utilizzo la pseudo-inversa (Moore-Penrose)")
        inverse_covariance_matrix = np.linalg.pinv(feature_matrix_with_intercept.T @ feature_matrix_with_intercept)
        hat_matrix = feature_matrix_with_intercept @ inverse_covariance_matrix @ feature_matrix_with_intercept.T
        leverage_scores = np.diag(hat_matrix)
    
    # -------------------------------------------------------------------------
    # 2. Residui Studentizzati (Studentized Residuals)
    # -------------------------------------------------------------------------
    # Calcolo pesi (w) e predizioni del modello lineare base
    linear_weights = inverse_covariance_matrix @ feature_matrix_with_intercept.T @ target_vector_y
    target_predictions = feature_matrix_with_intercept @ linear_weights
    prediction_errors = target_vector_y - target_predictions
    
    # Mean Squared Error
    mean_squared_error = np.sum(prediction_errors**2) / (num_samples - num_features - 1)
    
    # Calcolo residui studentizzati (normalizzati rispetto al leverage)
    studentized_residuals = prediction_errors / (np.sqrt(mean_squared_error) * np.sqrt(1 - leverage_scores))
    
    # -------------------------------------------------------------------------
    # 3. DFFITS (Difference in Fits)
    # -------------------------------------------------------------------------
    dffits_scores = studentized_residuals * np.sqrt(leverage_scores / (1 - leverage_scores))
    
    # -------------------------------------------------------------------------
    # 4. Applicazione Soglie Matematiche
    # -------------------------------------------------------------------------
    # Le soglie sono derivate dalla teoria statistica classica (Sez. 4.6)
    is_residual_outlier = np.abs(studentized_residuals) > 3
    
    leverage_threshold = 3 * (num_features + 1) / num_samples
    is_high_leverage_point = leverage_scores > leverage_threshold
    
    dffits_threshold = 2 * np.sqrt((num_features + 1) / num_samples)
    is_high_influence_point = np.abs(dffits_scores) > dffits_threshold
    
    # Regola Decisionale: Un punto viene rimosso solo se ha un errore anomalo 
    # E ALLO STESSO TEMPO altera pesantemente il modello (Leverage o Influenza alta)
    is_critical_outlier = is_residual_outlier & (is_high_leverage_point | is_high_influence_point)
    indices_to_drop = np.where(is_critical_outlier)[0]
    
    # -------------------------------------------------------------------------
    # 5. Report Accademico
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\nMetriche Dataset: {num_samples} Sample, {num_features} Features")
        print(f"\nSoglie Teoriche Applicate:")
        print(f"   Studentized Residuals (|r*|) : > 3.0")
        print(f"   Leverage Threshold (h_ii)    : > {leverage_threshold:.4f}")
        print(f"   DFFITS Threshold             : > {dffits_threshold:.4f}")
        
        print(f"\nRisultati Analisi:")
        print(f"   Outliers di Target (Residui) : {is_residual_outlier.sum()} ({is_residual_outlier.sum()/num_samples*100:.1f}%)")
        print(f"   Punti di High Leverage (HLP) : {is_high_leverage_point.sum()} ({is_high_leverage_point.sum()/num_samples*100:.1f}%)")
        print(f"   Punti di High Influence (HIP): {is_high_influence_point.sum()} ({is_high_influence_point.sum()/num_samples*100:.1f}%)")
        print(f"   >>> OUTLIERS CRITICI DA RIMUOVERE: {len(indices_to_drop)} ({len(indices_to_drop)/num_samples*100:.1f}%)")
        
        # Mostra i 5 sample peggiori per gravità DFFITS
        top_influence_indices = np.argsort(np.abs(dffits_scores))[-5:][::-1]
        print(f"\nTop 5 Sample più critici (ordinati per DFFITS):")
        print(f"   {'Indice Dataset':<16} {'DFFITS Score':<15} {'Leverage':<15} {'Residuo Stud.':<15}")
        print(f"   {'-'*65}")
        for idx in top_influence_indices:
            print(f"   {idx:<16} {dffits_scores[idx]:<15.4f} {leverage_scores[idx]:<15.4f} {studentized_residuals[idx]:<15.4f}")
    
    return {
        'outlier_indices': indices_to_drop,
        'studentized_residuals': studentized_residuals,
        'leverage': leverage_scores,
        'dffits': dffits_scores,
        'summary': {
            'n_outliers': len(indices_to_drop),
            'n_high_leverage': is_high_leverage_point.sum(),
            'n_high_influence': is_high_influence_point.sum()
        }
    }


def remove_outliers(df, outlier_indices_array, verbose=True):
    """Rimuove dal DataFrame le righe identificate come outliers critici."""
    if len(outlier_indices_array) == 0:
        if verbose: print("\n✓ Nessun outlier critico identificato. Dataset intatto.")
        return df
    
    cleaned_dataframe = df.drop(df.index[outlier_indices_array]).reset_index(drop=True)
    
    if verbose:
        print(f"\n✓ Operazione di Pulizia completata: Rimossi {len(outlier_indices_array)} outliers critici.")
        print(f"   Nuova dimensione Dataset: {len(df)} → {len(cleaned_dataframe)} sample validi.")
    
    return cleaned_dataframe
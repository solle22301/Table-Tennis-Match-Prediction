"""
Modulo per analisi avanzate dei modelli e delle features.

Questo modulo implementa:
- Matrice di correlazione tra features
- VIF (Variance Inflation Factor) per multicollinearità
- ROC Curve e AUC per valutazione modelli
- Feature importance con analisi per gruppi
- Cross-validation K-Fold
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc


# ==============================================================================
# MATRICE DI CORRELAZIONE
# ==============================================================================

def plot_correlation_matrix(df, feature_cols):
    """
    Visualizza matrice di correlazione tra features e target.
    
    Correlazione di Pearson:
    - Misura relazione lineare tra due variabili
    - Range: -1 (perfetta correlazione negativa) a +1 (perfetta positiva)
    - 0 = nessuna correlazione lineare
    
    Utile per:
    - Identificare features ridondanti (alta correlazione tra loro)
    - Trovare features più predittive del target
    - Rilevare multicollinearità
    
    Args:
        df (DataFrame): Dataset completo
        feature_cols (list): Lista nomi features
    
    Output:
        Salva grafico 'correlation_matrix.png'
    
    Returns:
        DataFrame: Matrice di correlazione
    """
    print("\n" + "=" * 80)
    print("MATRICE DI CORRELAZIONE")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Preparazione dati
    # -------------------------------------------------------------------------
    # Seleziona solo features + target
    # Imputa NaN con mediana (necessario per calcolo correlazione)
    df_corr = df[feature_cols + ['player_1_wins']].fillna(df[feature_cols].median())
    
    # Calcola matrice di correlazione
    corr_matrix = df_corr.corr()
    
    # -------------------------------------------------------------------------
    # Visualizzazione heatmap
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(corr_matrix,
                annot=True,              # mostra valori correlazione
                fmt='.2f',               # 2 decimali
                cmap='coolwarm',         # colormap rosso-blu
                center=0,                # centro scala a 0
                square=True,             # celle quadrate
                linewidths=0.5,          # linee separatrici
                cbar_kws={'label': 'Correlazione Pearson', 'shrink': 0.8},
                vmin=-1, vmax=1,         # range fisso -1 a +1
                annot_kws={'size': 8})
    
    plt.title('Matrice di Correlazione: Features + Target', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Salva grafico
    plt.savefig('correlation_matrix.png', dpi=200)
    print("\n✓ Grafico salvato: correlation_matrix.png")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Analisi correlazioni con target
    # -------------------------------------------------------------------------
    print("\nCorrelazioni con target (player_1_wins):")
    
    # Estrai correlazioni con target, ordina per valore assoluto
    target_corr = (corr_matrix['player_1_wins']
                   .drop('player_1_wins')
                   .sort_values(ascending=False))
    
    print("\nTop 5 features più correlate con target:")
    print("-" * 60)
    for i, (feat, corr) in enumerate(target_corr.head(5).items(), 1):
        # Interpretazione correlazione
        if abs(corr) > 0.5:
            strength = "forte"
        elif abs(corr) > 0.3:
            strength = "moderata"
        else:
            strength = "debole"
        
        direction = "positiva" if corr > 0 else "negativa"
        
        print(f"   {i}. {feat:30s}: {corr:+.4f} (correlazione {strength} {direction})")
    print("-" * 60)
    
    return corr_matrix


# ==============================================================================
# ANALISI MULTICOLLINEARITÀ (VIF)
# ==============================================================================

def calculate_vif(df, feature_cols):
    """
    Calcola VIF (Variance Inflation Factor) per rilevare multicollinearità.
    
    VIF (Variance Inflation Factor):
    - Misura quanto una feature è "spiegabile" dalle altre features
    - Formula: VIF_i = 1 / (1 - R²_i)
      dove R²_i è l'R² della regressione di feature_i sulle altre
    
    Interpretazione:
    - VIF = 1: nessuna correlazione con altre features
    - VIF < 5: multicollinearità accettabile
    - 5 < VIF < 10: multicollinearità moderata (attenzione)
    - VIF > 10: multicollinearità alta (problema serio)
    
    Problema multicollinearità:
    - Coefficienti instabili nei modelli lineari
    - Difficoltà interpretazione feature importance
    - Overfitting potenziale
    
    Soluzione:
    - Rimuovere una delle features correlate
    - Usare PCA (Principal Component Analysis)
    - Usare regolarizzazione (Ridge, Lasso)
    
    Args:
        df (DataFrame): Dataset completo
        feature_cols (list): Lista nomi features
    
    Returns:
        DataFrame: Tabella con VIF per ogni feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    print("\n" + "=" * 80)
    print("ANALISI MULTICOLLINEARITÀ (VIF)")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Preparazione dati
    # -------------------------------------------------------------------------
    # Seleziona features, imputa NaN con mediana
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # -------------------------------------------------------------------------
    # Calcolo VIF
    # -------------------------------------------------------------------------
    # Calcola VIF per ogni feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_cols
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(len(feature_cols))]
    
    # Ordina per VIF decrescente
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    # -------------------------------------------------------------------------
    # Stampa risultati
    # -------------------------------------------------------------------------
    print("\nVIF per feature:")
    print("\nInterpretazione:")
    print("   VIF < 5:      Nessuna multicollinearità")
    print("   5 < VIF < 10: Multicollinearità moderata")
    print("   VIF > 10:     Multicollinearità alta (considerare rimozione feature)")
    print("\n" + "-" * 60)
    print(f"{'Feature':<35s} {'VIF':>10s}  {'Status':<20s}")
    print("-" * 60)
    
    for _, row in vif_data.iterrows():
        feat = row['Feature']
        vif = row['VIF']
        
        # Determina status
        if vif < 5:
            status = "✓ OK"
        elif vif < 10:
            status = "⚠ Moderata"
        else:
            status = "✗ Alta"
        
        print(f"{feat:<35s} {vif:>10.2f}  {status:<20s}")
    
    print("-" * 60)
    
    # -------------------------------------------------------------------------
    # Identifica features problematiche
    # -------------------------------------------------------------------------
    high_vif = vif_data[vif_data['VIF'] > 10]
    
    if len(high_vif) > 0:
        print(f"\n⚠ Features con VIF > 10 (multicollinearità alta):")
        for _, row in high_vif.iterrows():
            print(f"   • {row['Feature']}: VIF = {row['VIF']:.2f}")
        print("\nRaccomandazione: Considera rimozione di una delle features correlate")
    else:
        print(f"\n✓ Nessuna feature con multicollinearità critica")
    
    return vif_data


# ==============================================================================
# ROC CURVE E AUC
# ==============================================================================

def plot_roc_curves(models, X_test, t_test):
    """
    Visualizza ROC Curve e calcola AUC per tutti i modelli.
    
    ROC (Receiver Operating Characteristic) Curve:
    - Grafico che mostra trade-off tra True Positive Rate e False Positive Rate
    - Asse Y: TPR = TP / (TP + FN) = Recall
    - Asse X: FPR = FP / (FP + TN)
    
    AUC (Area Under Curve):
    - Area sotto la curva ROC
    - Range: 0 a 1
    - Interpretazione:
        * AUC = 0.5: modello random (come tirare una moneta)
        * 0.5 < AUC < 0.7: performance scarsa
        * 0.7 < AUC < 0.8: performance accettabile
        * 0.8 < AUC < 0.9: performance buona
        * AUC > 0.9: performance eccellente
    
    Vantaggi AUC:
    - Indipendente dalla soglia di classificazione
    - Bilancia bene precision e recall
    - Utile per dataset sbilanciati
    
    Args:
        models (dict): Dizionario modelli addestrati
        X_test (array): Features test
        t_test (array): Target test
    
    Output:
        Salva grafico 'roc_curves.png'
    """
    print("\n" + "=" * 80)
    print("ROC CURVE E AUC")
    print("=" * 80)
    
    plt.figure(figsize=(10, 8))
    
    # -------------------------------------------------------------------------
    # Plot ROC curve per ogni modello
    # -------------------------------------------------------------------------
    for name, model in models.items():
        # Ottieni probabilità predette
        if hasattr(model, 'predict_proba'):
            # Probabilità classe positiva (player_1 vince)
            t_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Per modelli senza predict_proba (es. SVM lineare)
            t_proba = model.decision_function(X_test)
        
        # Calcola curve ROC
        fpr, tpr, thresholds = roc_curve(t_test, t_proba)
        
        # Calcola AUC
        roc_auc = auc(fpr, tpr)
        
        # Plot curva
        plt.plot(fpr, tpr, linewidth=2.5, 
                 label=f'{name} (AUC = {roc_auc:.4f})')
        
        # Stampa AUC
        print(f"\n{name}:")
        print(f"   AUC: {roc_auc:.4f}")
        
        # Interpretazione
        if roc_auc > 0.9:
            interpretation = "eccellente"
        elif roc_auc > 0.8:
            interpretation = "buona"
        elif roc_auc > 0.7:
            interpretation = "accettabile"
        elif roc_auc > 0.6:
            interpretation = "scarsa"
        else:
            interpretation = "molto scarsa"
        print(f"   Interpretazione: Performance {interpretation}")
    
    # -------------------------------------------------------------------------
    # Linea baseline (classificatore random)
    # -------------------------------------------------------------------------
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, 
             label='Random Classifier (AUC = 0.5000)')
    
    # -------------------------------------------------------------------------
    # Formattazione grafico
    # -------------------------------------------------------------------------
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Confronto Modelli', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Salva grafico
    plt.savefig('roc_curves.png', dpi=150)
    print("\n✓ Grafico salvato: roc_curves.png")
    plt.show()


# ==============================================================================
# FEATURE IMPORTANCE DETTAGLIATA
# ==============================================================================

def analyze_feature_importance(model, feature_cols):
    """
    Analizza feature importance con raggruppamento per categoria.
    
    Feature Importance (Random Forest):
    - Misura quanto ogni feature contribuisce alle decisioni del modello
    - Basata su riduzione impurità (Gini) durante split degli alberi
    - Valori normalizzati: somma = 1.0
    
    Categorie features:
    1. Ranking (4): ELO individuali, differenza, somma
    2. Win Rate (4): % vittorie overall e ultime 5
    3. Head-to-Head (2): vittorie P1 e P2 negli scontri diretti
    4. Momentum (2): streak vittorie/sconfitte
    
    Utile per:
    - Interpretare decisioni del modello
    - Identificare features ridondanti (importance bassa)
    - Feature selection (rimuovere features poco importanti)
    - Validare ipotesi (es. ELO dovrebbe essere importante)
    
    Args:
        model: Modello Random Forest addestrato
        feature_cols (list): Lista nomi features
    
    Output:
        Salva grafico 'feature_importance.png'
    
    Returns:
        DataFrame: Tabella feature importance ordinata
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Verifica supporto feature importance
    # -------------------------------------------------------------------------
    if not hasattr(model, 'feature_importances_'):
        print("\n⚠ Modello non supporta feature importance")
        print("   (disponibile solo per tree-based models: Random Forest, XGBoost, etc.)")
        return None
    
    # -------------------------------------------------------------------------
    # Estrazione e ordinamento
    # -------------------------------------------------------------------------
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # ordine decrescente
    
    # Crea DataFrame per visualizzazione
    importance_df = pd.DataFrame({
        'Feature': [feature_cols[i] for i in indices],
        'Importance': importances[indices],
        'Percentage': (importances[indices] / importances.sum()) * 100
    })
    
    # -------------------------------------------------------------------------
    # Stampa ranking
    # -------------------------------------------------------------------------
    print("\nRanking features per importanza:")
    print("-" * 80)
    print(f"{'Rank':<6s} {'Feature':<30s} {'Importance':<12s} {'%':<8s} {'Cumulative %':<12s}")
    print("-" * 80)
    
    cumulative = 0
    for i, row in importance_df.iterrows():
        cumulative += row['Percentage']
        print(f"{i+1:<6d} {row['Feature']:<30s} "
              f"{row['Importance']:<12.4f} {row['Percentage']:<8.2f} {cumulative:<12.1f}")
    
    print("-" * 80)
    
    # -------------------------------------------------------------------------
    # Visualizzazione bar plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Colori graduati con colormap
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importances)))
    
    plt.bar(range(len(importances)), importances[indices], 
            color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    plt.xticks(range(len(importances)), 
               [feature_cols[i] for i in indices], 
               rotation=45, ha='right', fontsize=10)
    
    plt.title('Feature Importance - Random Forest', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Importance', fontsize=11)
    plt.xlabel('Features', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Salva grafico
    plt.savefig('feature_importance.png', dpi=150)
    print("\n✓ Grafico salvato: feature_importance.png")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Analisi per gruppi
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("IMPORTANZA PER CATEGORIA")
    print("=" * 80)
    
    # Definizione gruppi
    groups = {
        'Ranking': ['player_1_elo', 'player_2_elo', 'elo_diff', 'elo_sum'],
        'Win Rate': ['p1_win_rate_overall', 'p2_win_rate_overall', 
                     'p1_win_rate_last5', 'p2_win_rate_last5'],
        'Head-to-Head': ['h2h_p1_wins', 'h2h_p2_wins'],
        'Momentum': ['p1_streak', 'p2_streak']
    }
    
    print("\nContributo per categoria di features:")
    print("-" * 60)
    
    group_importance = {}
    
    for group_name, group_features in groups.items():
        # Somma importance features nel gruppo
        total = importance_df[importance_df['Feature'].isin(group_features)]['Percentage'].sum()
        group_importance[group_name] = total
        
        print(f"\n{group_name} ({len(group_features)} features): {total:.2f}%")
        
        # Stampa dettaglio per ogni feature del gruppo
        for feat in group_features:
            if feat in importance_df['Feature'].values:
                feat_imp = importance_df[importance_df['Feature'] == feat]['Percentage'].values[0]
                # Calcola percentuale relativa dentro il gruppo
                rel_pct = (feat_imp / total * 100) if total > 0 else 0
                print(f"   • {feat:30s}: {feat_imp:5.2f}% (contributo al gruppo: {rel_pct:5.1f}%)")
    
    print("-" * 60)
    
    # -------------------------------------------------------------------------
    # Insights interpretativi
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INSIGHTS")
    print("=" * 80)
    
    top_feature = importance_df.iloc[0]
    print(f"\n✓ Feature più importante: {top_feature['Feature']}")
    print(f"   Contribuisce al {top_feature['Percentage']:.1f}% delle decisioni")
    
    top_3_pct = importance_df.iloc[:3]['Percentage'].sum()
    print(f"\n✓ Top 3 features spiegano {top_3_pct:.1f}% delle decisioni:")
    for i in range(3):
        feat = importance_df.iloc[i]
        print(f"   {i+1}. {feat['Feature']}: {feat['Percentage']:.1f}%")
    
    # Identifica features poco importanti (<5%)
    low_importance = importance_df[importance_df['Percentage'] < 5]
    if len(low_importance) > 0:
        print(f"\n⚠ Features con bassa importanza (<5%):")
        for _, row in low_importance.iterrows():
            print(f"   • {row['Feature']}: {row['Percentage']:.2f}%")
        print("   Considerare rimozione per semplificare modello")
    
    return importance_df


# ==============================================================================
# CROSS-VALIDATION K-FOLD
# ==============================================================================

def perform_cross_validation(model, X, y, cv=5):
    """
    Esegue K-Fold Cross-Validation per validazione robusta.
    
    K-Fold Cross-Validation:
    - Divide dataset in K fold (sottoinsiemi) di dimensione simile
    - Itera K volte, ogni volta usa K-1 fold per train, 1 per test
    - Calcola metriche su ogni fold, poi aggrega (media ± std)
    
    Vantaggi:
    - Usa tutti i dati sia per train che per test
    - Stima più robusta delle performance (riduce varianza)
    - Rileva overfitting (gap train-test)
    
    Metriche calcolate:
    - Train accuracy: performance sul training set di ogni fold
    - Test accuracy: performance sul test set di ogni fold
    - F1-score: media armonica precision/recall
    
    Args:
        model: Modello da validare
        X (array): Features complete
        y (array): Target completo
        cv (int): Numero di fold (default 5)
    
    Returns:
        dict: Risultati cross-validation
    """
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer, accuracy_score, f1_score
    
    print("\n" + "=" * 80)
    print(f"CROSS-VALIDATION ({cv}-FOLD)")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Definizione metriche
    # -------------------------------------------------------------------------
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score)
    }
    
    # -------------------------------------------------------------------------
    # Esecuzione cross-validation
    # -------------------------------------------------------------------------
    print(f"\nEsecuzione {cv}-fold cross-validation...")
    print("(Può richiedere alcuni minuti...)")
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv,                      # numero fold
        scoring=scoring,            # metriche
        return_train_score=True     # calcola anche train score
    )
    
    # -------------------------------------------------------------------------
    # Stampa risultati per fold
    # -------------------------------------------------------------------------
    print(f"\nRisultati per fold:")
    print("-" * 70)
    print(f"{'Fold':<8s} {'Train Acc':<12s} {'Test Acc':<12s} {'Test F1':<12s}")
    print("-" * 70)
    
    for i in range(cv):
        print(f"{i+1:<8d} "
              f"{cv_results['train_accuracy'][i]:<12.4f} "
              f"{cv_results['test_accuracy'][i]:<12.4f} "
              f"{cv_results['test_f1'][i]:<12.4f}")
    
    print("-" * 70)
    
    # -------------------------------------------------------------------------
    # Statistiche aggregate
    # -------------------------------------------------------------------------
    print(f"\nStatistiche aggregate (media ± deviazione standard):")
    print("-" * 70)
    
    train_acc_mean = cv_results['train_accuracy'].mean()
    train_acc_std = cv_results['train_accuracy'].std()
    test_acc_mean = cv_results['test_accuracy'].mean()
    test_acc_std = cv_results['test_accuracy'].std()
    test_f1_mean = cv_results['test_f1'].mean()
    test_f1_std = cv_results['test_f1'].std()
    
    print(f"   Train Accuracy: {train_acc_mean:.4f} ± {train_acc_std:.4f}")
    print(f"   Test Accuracy:  {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    print(f"   Test F1-Score:  {test_f1_mean:.4f} ± {test_f1_std:.4f}")
    
    # -------------------------------------------------------------------------
    # Analisi overfitting
    # -------------------------------------------------------------------------
    gap = train_acc_mean - test_acc_mean
    
    print(f"\nAnalisi overfitting:")
    print(f"   Gap Train-Test: {gap:.4f}")
    
    if gap < 0.05:
        print("   ✓ Nessun overfitting significativo")
        print("     (modello generalizza bene su nuovi dati)")
    elif gap < 0.10:
        print("   ⚠ Leggero overfitting")
        print("     (modello memorizza parzialmente training set)")
    else:
        print("   ✗ Overfitting rilevato")
        print("     (modello memorizza troppo il training set)")
        print("     Raccomandazioni:")
        print("       - Aumenta regolarizzazione")
        print("       - Riduci complessità modello")
        print("       - Aumenta dati training")
    
    print("-" * 70)
    
    return cv_results

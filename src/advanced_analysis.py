"""
Modulo per l'Analisi Avanzata: Diagnostica Modelli e Importanza Variabili.

Questo modulo si occupa di analizzare "sotto il cofano" i dati e i modelli:
- Multicollinearità: Matrice di Correlazione e VIF (Variance Inflation Factor).
- Interpretazione Modello: Feature Importance (per Random Forest).
- Validazione Statistica: Cross-Validation K-Fold per stima robusta dell'accuratezza.
- Qualità Probabilistica: Curve ROC (AUC) e Calibrazione (Brier Score).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score


# ==============================================================================
# 1. MATRICE DI CORRELAZIONE LINEARE
# ==============================================================================

def plot_correlation_matrix(dataset_df, feature_columns):
    """
    Genera la matrice di correlazione di Pearson per visualizzare le 
    relazioni lineari tra le features e il target.
    """
    print("\n" + "=" * 80)
    print("ANALISI CORRELAZIONI (PEARSON)")
    print("=" * 80)
    
    # Preparazione dati: aggiunge il target e riempie eventuali NaN con la mediana
    correlation_df = dataset_df[feature_columns + ['player_1_wins']].fillna(dataset_df[feature_columns].median())
    
    # Calcolo Matrice
    correlation_matrix = correlation_df.corr()
    
    # -------------------------------------------------------------------------
    # Visualizzazione Heatmap
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix,
                annot=True,              # Mostra valori numerici
                fmt='.2f',               # 2 cifre decimali
                cmap='coolwarm',         # Rosso = Positiva, Blu = Negativa
                center=0,                # Punto neutro
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Indice di Correlazione di Pearson', 'shrink': 0.8},
                vmin=-1, vmax=1,
                annot_kws={'size': 8})
    
    plt.title('Matrice di Correlazione: Features e Target', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    path_grafico = os.path.join("plots", "correlation_matrix.png")
    plt.savefig(path_grafico, dpi=200)
    print("\n✓ Heatmap esportata: correlation_matrix.png")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Estrazione Insights
    # -------------------------------------------------------------------------
    print("\nTop 5 Features più correlate con la Vittoria del Player 1:")
    
    # Estrae solo la colonna del target, rimuove il target stesso e ordina per valore assoluto
    target_correlations = (correlation_matrix['player_1_wins']
                           .drop('player_1_wins')
                           .sort_values(ascending=False))
    
    print("-" * 65)
    for index, (feature_name, corr_value) in enumerate(target_correlations.head(5).items(), 1):
        if abs(corr_value) > 0.5:   strength = "Forte"
        elif abs(corr_value) > 0.3: strength = "Moderata"
        else:                       strength = "Debole"
        
        direction = "Positiva" if corr_value > 0 else "Negativa"
        
        print(f"   {index}. {feature_name:30s}: {corr_value:+.4f} ({strength} {direction})")
    print("-" * 65)
    
    return correlation_matrix


# ==============================================================================
# 2. ANALISI MULTICOLLINEARITÀ (VIF)
# ==============================================================================

def calculate_vif(dataset_df, feature_columns):
    """
    Calcola il Variance Inflation Factor (VIF) per rilevare multicollinearità.
    Se una feature ha VIF > 10, significa che è quasi perfettamente predicibile
    dalle altre features, rendendo i modelli lineari instabili.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    print("\n" + "=" * 80)
    print("ANALISI MULTICOLLINEARITÀ (VIF)")
    print("=" * 80)
    
    X_matrix = dataset_df[feature_columns].fillna(dataset_df[feature_columns].median())
    
    vif_results = pd.DataFrame()
    vif_results["Feature"] = feature_columns
    vif_results["VIF_Score"] = [variance_inflation_factor(X_matrix.values, i) for i in range(len(feature_columns))]
    
    # Ordina i risultati dal più critico al meno critico
    vif_results = vif_results.sort_values('VIF_Score', ascending=False)
    
    print("\nInterpretazione Accademica:")
    print("   VIF < 5:      Assenza di multicollinearità (Ottimale)")
    print("   5 < VIF < 10: Multicollinearità moderata (Accettabile)")
    print("   VIF > 10:     Multicollinearità severa (Richiede rimozione feature)")
    
    print("\n" + "-" * 60)
    print(f"{'Nome Feature':<35s} {'VIF':>10s}  {'Diagnosi':<20s}")
    print("-" * 60)
    
    for _, row in vif_results.iterrows():
        feature_name = row['Feature']
        score = row['VIF_Score']
        
        if score < 5:      status = "✓ OK"
        elif score < 10:   status = "⚠ Moderata"
        else:              status = "❌ Severa"
        
        print(f"{feature_name:<35s} {score:>10.2f}  {status:<20s}")
    
    print("-" * 60)
    return vif_results


# ==============================================================================
# 3. CURVE ROC E AUC (Area Under Curve)
# ==============================================================================

def plot_roc_curves(trained_models_dict, X_test_scaled, y_test):
    """
    Genera le curve ROC per valutare il trade-off tra Sensibilità (True Positive Rate) 
    e Specificità (False Positive Rate) per ciascun modello.
    """
    print("\n" + "=" * 80)
    print("ANALISI CURVE ROC (Receiver Operating Characteristic)")
    print("=" * 80)
    
    plt.figure(figsize=(10, 8))
    
    for model_name, model in trained_models_dict.items():
        # Estrazione delle probabilità predette (solo classe 1 = Vittoria P1)
        if hasattr(model, 'predict_proba'):
            y_predicted_probabilities = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_predicted_probabilities = model.decision_function(X_test_scaled)
        
        # Calcolo FPR, TPR e Area Sotto la Curva (AUC)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_predicted_probabilities)
        auc_score = auc(false_positive_rate, true_positive_rate)
        
        # Plot della curva del modello
        plt.plot(false_positive_rate, true_positive_rate, linewidth=2.5, label=f'{model_name} (AUC = {auc_score:.4f})')
        
        # Output testuale
        print(f"\n{model_name}:")
        print(f"   AUC Score: {auc_score:.4f}")
        
        if auc_score > 0.9:   diagnosi = "Eccellente"
        elif auc_score > 0.8: diagnosi = "Buona"
        elif auc_score > 0.7: diagnosi = "Accettabile"
        elif auc_score > 0.6: diagnosi = "Scarsa"
        else:                 diagnosi = "Inadeguata"
        print(f"   Valutazione Capacità Discriminativa: {diagnosi}")
    
    # Linea della Baseline (Modello Casuale)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guessing (AUC = 0.50)')
    
    plt.xlabel('False Positive Rate (1 - Specificità)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensibilità)', fontsize=12)
    plt.title('Curve ROC - Confronto tra Algoritmi', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    path_grafico = os.path.join("plots", "roc_curves.png")
    plt.savefig(path_grafico, dpi=150)
    print("\n✓ Grafico esportato: roc_curves.png")
    plt.show()


# ==============================================================================
# 4. FEATURE IMPORTANCE (Analisi Random Forest)
# ==============================================================================

def analyze_feature_importance(random_forest_model, feature_columns):
    """
    Estrae i pesi decisionali (Impurità di Gini) dal modello Random Forest 
    per capire quali variabili guidano le predizioni.
    """
    print("\n" + "=" * 80)
    print("ANALISI IMPORTANZA VARIABILI (FEATURE IMPORTANCE)")
    print("=" * 80)
    
    if not hasattr(random_forest_model, 'feature_importances_'):
        print("⚠️ Modello non supportato (Richiede modelli Tree-Based).")
        return None
    
    raw_importances = random_forest_model.feature_importances_
    sorted_indices = np.argsort(raw_importances)[::-1]
    
    importance_dataframe = pd.DataFrame({
        'Feature': [feature_columns[i] for i in sorted_indices],
        'Importance_Score': raw_importances[sorted_indices],
        'Impact_Percentage': (raw_importances[sorted_indices] / raw_importances.sum()) * 100
    })
    
    # -------------------------------------------------------------------------
    # Visualizzazione Bar Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(raw_importances)))
    
    plt.bar(range(len(raw_importances)), raw_importances[sorted_indices], 
            color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    plt.xticks(range(len(raw_importances)), [feature_columns[i] for i in sorted_indices], 
               rotation=45, ha='right', fontsize=10)
    
    plt.title('Feature Importance (Random Forest Gini Impurity)', fontsize=14, fontweight='bold')
    plt.ylabel('Punteggio di Importanza', fontsize=11)
    plt.xlabel('Variabili Predittive', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    path_grafico = os.path.join("plots", "feature_importance.png")
    plt.savefig(path_grafico, dpi=150)
    print("✓ Grafico esportato: feature_importance.png")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Aggregazione per Categorie Logiche
    # -------------------------------------------------------------------------
    print("\nImpatto Aggregato per Macro-Categorie:")
    print("-" * 60)
    
    logical_groups = {
        'Ranking Storico (ELO)': ['elo_diff'],
        'Stato di Forma (Win Rate)': ['p1_win_rate_last5', 'p2_win_rate_last5'],
        'Scontri Diretti (H2H)': ['p1_head_to_head_wins', 'p2_head_to_head_wins', 'p1_head_to_head_win_ratio'],
        'Momentum / Inerzia': ['p1_streak', 'p2_streak'],
        'Volatilità / Affidabilità': ['p1_form_volatility', 'p2_form_volatility']
    }
    
    for group_name, variables in logical_groups.items():
        total_group_impact = importance_dataframe[importance_dataframe['Feature'].isin(variables)]['Impact_Percentage'].sum()
        print(f"\n{group_name} ({len(variables)} feature/s): {total_group_impact:.1f}% del peso decisionale")
        
        for feature in variables:
            if feature in importance_dataframe['Feature'].values:
                specific_impact = importance_dataframe[importance_dataframe['Feature'] == feature]['Impact_Percentage'].values[0]
                print(f"   • {feature:30s}: {specific_impact:4.1f}%")
                
    print("-" * 60)
    return importance_dataframe


# ==============================================================================
# 5. K-FOLD CROSS VALIDATION
# ==============================================================================

def perform_cross_validation(model, X_train_scaled, y_train, cv=5):
    """
    Esegue la K-Fold Cross Validation sul set di addestramento per stimare
    le prestazioni del modello senza toccare il Test Set, prevenendo l'Overfitting.
    """
    print("\n" + "=" * 80)
    print(f"K-FOLD CROSS-VALIDATION (K={cv})")
    print("=" * 80)
    print("In esecuzione... (Richiede potenza di calcolo)")
    
    scoring_metrics = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score)
    }
    
    cv_results_dict = cross_validate(
        model, X_train_scaled, y_train,
        cv=cv,
        scoring=scoring_metrics,
        return_train_score=True
    )
    
    # Estrazione medie e deviazioni standard
    mean_train_acc = cv_results_dict['train_accuracy'].mean()
    mean_test_acc = cv_results_dict['test_accuracy'].mean()
    std_test_acc = cv_results_dict['test_accuracy'].std()
    
    print(f"\nReport Finale Cross Validation:")
    print(f"   Accuracy sul Train Fold : {mean_train_acc:.4f}")
    print(f"   Accuracy sul Valid Fold : {mean_test_acc:.4f} (± {std_test_acc:.4f})")
    
    # Diagnosi Overfitting
    train_valid_gap = mean_train_acc - mean_test_acc
    print(f"\nDiagnosi Generalizzazione (Gap Train-Validation: {train_valid_gap:.4f}):")
    
    if train_valid_gap < 0.05:
        print("   ✓ Eccellente: Nessun overfitting rilevato. Il modello generalizza bene.")
    elif train_valid_gap < 0.10:
        print("   ⚠️ Attenzione: Lieve overfitting. Il modello inizia a memorizzare il train set.")
    else:
        print("   ❌ Allarme: Forte overfitting. Il modello non riesce a generalizzare su dati nuovi.")
        
    return cv_results_dict


# ==============================================================================
# 6. CALIBRAZIONE E BRIER SCORE
# ==============================================================================

def plot_calibration_curves(trained_models_dict, X_test_scaled, y_test):
    """
    Analizza la Reliability del modello (Calibrazione).
    Risponde alla domanda: "Se il modello dice che P1 ha il 70% di probabilità 
    di vincere, P1 vince effettivamente il 70% delle volte?"
    """
    print("\n" + "=" * 80)
    print("ANALISI DI CALIBRAZIONE (RELIABILITY) E BRIER SCORE")
    print("=" * 80)
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Calibrazione Perfetta (Ideale)")
    
    for model_name, model in trained_models_dict.items():
        if hasattr(model, 'predict_proba'):
            # Probabilità assegnata alla vittoria del P1
            y_predicted_probabilities = model.predict_proba(X_test_scaled)[:, 1]
            
            # Brier Score (Mean Squared Error delle probabilità: 0 = Perfetto, 1 = Pessimo)
            brier_value = brier_score_loss(y_test, y_predicted_probabilities)
            
            print(f"\n{model_name}:")
            print(f"   Brier Score: {brier_value:.4f}")
            
            if brier_value < 0.15:   giudizio = "Eccellente (Molto Affidabile)"
            elif brier_value < 0.20: giudizio = "Buono"
            elif brier_value < 0.25: giudizio = "Accettabile"
            else:                    giudizio = "Scarso (Inaffidabile)"
            print(f"   Valutazione: {giudizio}")
            
            # Generazione punti della curva
            true_fraction, predicted_mean = calibration_curve(y_test, y_predicted_probabilities, n_bins=10, strategy='uniform')
            plt.plot(predicted_mean, true_fraction, "s-", label=f"{model_name} (Brier: {brier_value:.3f})")
            
    plt.ylabel("Frequenza Reale di Vittoria", fontsize=12)
    plt.xlabel("Probabilità Stimata dal Modello", fontsize=12)
    plt.title("Curve di Calibrazione (Reliability Diagram)", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    path_grafico = os.path.join("plots", "calibration_curves.png")
    plt.savefig(path_grafico, dpi=150)
    print("\n✓ Grafico esportato: calibration_curves.png")
    plt.show()
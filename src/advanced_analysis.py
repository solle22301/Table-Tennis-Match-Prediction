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
    print("\n   -> Generazione Matrice di Correlazione di Pearson...")
    
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
    print(f"      ✓ Heatmap esportata con successo: {path_grafico}")
    
    # Visualizzazione a schermo per la presentazione
    plt.show() 
    
    # -------------------------------------------------------------------------
    # Estrazione Insights
    # -------------------------------------------------------------------------
    print("\n      Top 5 Features più correlate con la Vittoria del Player 1:")
    
    target_correlations = (correlation_matrix['player_1_wins']
                           .drop('player_1_wins')
                           .sort_values(ascending=False))
    
    print("      " + "-" * 60)
    for index, (feature_name, corr_value) in enumerate(target_correlations.head(5).items(), 1):
        if abs(corr_value) > 0.5:   strength = "Forte"
        elif abs(corr_value) > 0.3: strength = "Moderata"
        else:                       strength = "Debole"
        
        direction = "Positiva" if corr_value > 0 else "Negativa"
        print(f"        {index}. {feature_name:30s}: {corr_value:+.4f} ({strength} {direction})")
    print("      " + "-" * 60)
    
    return correlation_matrix


# ==============================================================================
# 2. ANALISI MULTICOLLINEARITÀ (VIF)
# ==============================================================================

def calculate_vif(dataset_df, feature_columns):
    """
    Calcola il Variance Inflation Factor (VIF) per rilevare multicollinearità.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    print("\n   -> Analisi Multicollinearità tramite VIF (Variance Inflation Factor)...")
    
    X_matrix = dataset_df[feature_columns].fillna(dataset_df[feature_columns].median())
    
    vif_results = pd.DataFrame()
    vif_results["Feature"] = feature_columns
    vif_results["VIF_Score"] = [variance_inflation_factor(X_matrix.values, i) for i in range(len(feature_columns))]
    
    vif_results = vif_results.sort_values('VIF_Score', ascending=False)
    
    print("      " + "-" * 60)
    print(f"      {'Nome Feature':<35s} {'VIF':>10s}  {'Diagnosi':<20s}")
    print("      " + "-" * 60)
    
    for _, row in vif_results.iterrows():
        feature_name = row['Feature']
        score = row['VIF_Score']
        status = "✓ OK" if score < 5 else "⚠ Moderata" if score < 10 else "❌ Severa"
        print(f"      {feature_name:<35s} {score:>10.2f}  {status:<20s}")
    
    print("      " + "-" * 60)
    return vif_results


# ==============================================================================
# 3. CURVE ROC E AUC
# ==============================================================================

def plot_roc_curves(trained_models_dict, X_test_scaled, y_test):
    """
    Genera le curve ROC per valutare la capacità discriminativa dei modelli.
    """
    print("\n   -> Generazione Curve ROC e Calcolo AUC...")
    
    plt.figure(figsize=(10, 8))
    
    for model_name, model in trained_models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_probs = model.decision_function(X_test_scaled)
        
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc_score = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC = {auc_score:.4f})')
        print(f"      • {model_name:20s}: AUC = {auc_score:.4f}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Curve ROC - Confronto tra Algoritmi', fontweight='bold')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    
    path_grafico = os.path.join("plots", "roc_curves.png")
    plt.savefig(path_grafico, dpi=150)
    print(f"      ✓ Grafico esportato con successo: {path_grafico}")
    plt.show()


# ==============================================================================
# 4. FEATURE IMPORTANCE
# ==============================================================================

def analyze_feature_importance(random_forest_model, feature_columns):
    """
    Estrae l'importanza delle variabili dal modello Random Forest.
    """
    print("\n   -> Estrazione Feature Importance (solo modelli Tree-Based)...")
    
    if not hasattr(random_forest_model, 'feature_importances_'):
        print("      ⚠️ Modello non supportato per l'importanza diretta.")
        return None
    
    importances = random_forest_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(importances))))
    plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance (Random Forest)', fontweight='bold')
    plt.tight_layout()
    
    path_grafico = os.path.join("plots", "feature_importance.png")
    plt.savefig(path_grafico, dpi=150)
    print(f"      ✓ Grafico esportato con successo: {path_grafico}")
    plt.show()
    return pd.DataFrame({'Feature': [feature_columns[i] for i in indices], 'Score': importances[indices]})


# ==============================================================================
# 5. K-FOLD CROSS VALIDATION (CORRETTA PER PARAMETRO model_name)
# ==============================================================================

def perform_cross_validation(model, X_train_scaled, y_train, cv=5, model_name="Modello"):
    """
    Esegue la K-Fold Cross Validation. Accetta ora model_name per coerenza col main.
    """
    print(f"      - Validazione in corso per: {model_name}...")
    
    scoring_metrics = {'accuracy': make_scorer(accuracy_score), 'f1': make_scorer(f1_score)}
    
    cv_results_dict = cross_validate(
        model, X_train_scaled, y_train,
        cv=cv,
        scoring=scoring_metrics,
        return_train_score=True
    )
    
    mean_train = cv_results_dict['train_accuracy'].mean()
    mean_valid = cv_results_dict['test_accuracy'].mean()
    gap = mean_train - mean_valid
    
    print(f"        [Train Acc: {mean_train:.4f} | Valid Acc: {mean_valid:.4f} | Gap: {gap:.4f}]")
    
    return cv_results_dict


# ==============================================================================
# 6. CALIBRAZIONE
# ==============================================================================

def plot_calibration_curves(trained_models_dict, X_test_scaled, y_test):
    """
    Analizza la calibrazione delle probabilità dei modelli.
    """
    print("\n   -> Analisi di Calibrazione e Brier Score...")
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Calibrazione Perfetta")
    
    for model_name, model in trained_models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_probs = model.predict_proba(X_test_scaled)[:, 1]
            brier = brier_score_loss(y_test, y_probs)
            print(f"      • {model_name:20s}: Brier Score = {brier:.4f}")
            
            prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
            plt.plot(prob_pred, prob_true, "s-", label=f"{model_name} ({brier:.3f})")
            
    plt.title("Curve di Calibrazione", fontweight='bold')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    
    path_grafico = os.path.join("plots", "calibration_curves.png")
    plt.savefig(path_grafico, dpi=150)
    print(f"      ✓ Grafico esportato con successo: {path_grafico}")
    plt.show()
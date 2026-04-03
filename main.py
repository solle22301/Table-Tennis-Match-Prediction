"""
================================================================================
🏓 TABLE TENNIS MATCH PREDICTION
================================================================================
Autore: Alessandro Sollevanti
Data: Gennaio 2026
Corso: Ingegneria Informatica - Machine Learning

## OBIETTIVO DEL PROGETTO:
Sviluppo di una pipeline di ML per la predizione dell'esito di match di tennis tavolo.
Il sistema analizza features dinamiche (forma, ELO, streak) e confronta modelli 
lineari, ad albero e neurali, unificandoli in un Ensemble ottimizzato.
================================================================================
"""

import os
import warnings
import joblib

# Silenzia i fastidiosi FutureWarning di scikit-learn per mantenere l'output pulito
warnings.filterwarnings("ignore", category=FutureWarning)

# Definizione del percorso della cartella destinata ai grafici generati
plots_folder = "plots"
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Import moduli del progetto
from src.data_loader import load_data
from src.preprocessing import normalize_and_merge
from src.feature_engineering import create_features
from src.exploratory_analysis import run_eda, detect_outliers, remove_outliers
from src.modeling import train_test_split_temporal, prepare_data, train_baseline_models, tune_all_models, build_optimized_ensemble
from src.evaluation import evaluate_models, plot_confusion_matrix
from src.advanced_analysis import (plot_correlation_matrix, calculate_vif, 
                                   plot_roc_curves, analyze_feature_importance,
                                   perform_cross_validation, plot_calibration_curves)

def main():
    # Header progetto
    print("=" * 80)
    print(" 🏓 TABLE TENNIS MATCH PREDICTION")
    print("    Progetto Machine Learning - Alessandro Sollevanti")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # FASI 1-2: Caricamento e Preprocessing
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASI 1-2: ACQUISIZIONE DATI E PREPROCESAMENTO")
    print("=" * 80)
    
    matches_file = os.path.join('datasets', 'TT_Elite_Series_Dataset.csv')
    ranking_file = os.path.join('datasets', 'RANKING-TT-ELITE-SERIES.csv')
    
    df_matches, df_ranking = load_data(matches_file, ranking_file)
    df_merged = normalize_and_merge(df_matches, df_ranking)
    
    # -------------------------------------------------------------------------
    # FASE 3: Feature Engineering
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASE 3: FEATURE ENGINEERING (Rolling Window)")
    print("=" * 80)
    
    df_features = create_features(df_merged)
    
    # Pulizia dataset finale post-generazione features
    df_clean = df_features.dropna(subset=['player_1_elo', 'player_2_elo']).reset_index(drop=True)
    print(f"\n   -> Pulizia dataset:")
    print(f"      Righe rimosse (dati mancanti): {len(df_features) - len(df_clean)}")
    print(f"      Dataset pronto per l'analisi: {len(df_clean)} partite")
    
    # -------------------------------------------------------------------------
    # FASE 4: EDA + Outlier Detection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASE 4: EXPLORATORY DATA ANALYSIS & OUTLIER DETECTION")
    print("=" * 80)
    
    # Esecuzione analisi descrittiva
    baseline = run_eda(df_clean)
    
    # Configurazione rilevamento anomalie
    feature_cols_outlier = [
        'player_1_elo', 'elo_diff', 'p1_win_rate_last5', 'p2_win_rate_last5',
        'p1_head_to_head_wins', 'p2_head_to_head_wins', 'p1_head_to_head_win_ratio',
        'p1_streak', 'p2_streak', 'p1_form_volatility', 'p2_form_volatility'
    ]
    
    if 'player_1_wins' not in df_clean.columns:
        df_clean['player_1_wins'] = (df_clean['winner'] == df_clean['player_1']).astype(int)
    
    # Rilevamento e rimozione outliers multivariate
    outlier_result = detect_outliers(df_clean[feature_cols_outlier].fillna(0).values, 
                                     df_clean['player_1_wins'].values, 
                                     feature_names=feature_cols_outlier, verbose=True)
    
    df_clean = remove_outliers(df_clean, outlier_result['outlier_indices'], verbose=True)
    
    # -------------------------------------------------------------------------
    # FASI 5-6-7: Preparazione Finale
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASI 5-6-7: SPLIT TEMPORALE E DIAGNOSTICA FEATURES")
    print("=" * 80)
    
    train, test = train_test_split_temporal(df_clean, test_size=0.2)
    
    feature_cols = ['elo_diff', 'p1_win_rate_last5', 'p2_win_rate_last5',
                    'p1_head_to_head_wins', 'p2_head_to_head_wins', 'p1_head_to_head_win_ratio',
                    'p1_streak', 'p2_streak', 'p1_form_volatility', 'p2_form_volatility']
    
    # Gestione eventuali NaN residui nelle features rolling
    train[feature_cols] = train[feature_cols].fillna(0)
    test[feature_cols] = test[feature_cols].fillna(0)
    
    # Analisi tecnica delle variabili
    plot_correlation_matrix(train, feature_cols)
    calculate_vif(train, feature_cols)
    
    # Trasformazione in matrici e scalatura Z-Score
    X_train, y_train, X_test, y_test, scaler = prepare_data(train, test, feature_cols)
    
    # -------------------------------------------------------------------------
    # FASI 8-9-10: Modeling & Tuning
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASI 8-9-10: ADDESTRAMENTO, TUNING E ENSEMBLE")
    print("=" * 80)
    
    # Training iniziale (Baseline)
    train_baseline_models(X_train, y_train)
    
    # Ottimizzazione iperparametri tramite Grid Search
    tuned_models = tune_all_models(X_train, y_train)

    # Costruzione Ensemble Voting con i modelli ottimizzati
    optimized_ensemble = build_optimized_ensemble(tuned_models, X_train, y_train)

    # Creazione dizionario finale per validazione
    final_models = tuned_models.copy()
    final_models['Ensemble Voting'] = optimized_ensemble

    # Validazione robusta (Cross-Validation)
    print("\n   📊 Valutazione tramite 5-Fold Cross-Validation:")
    cv_results = {}
    for name, model in final_models.items():
        cv_scores = perform_cross_validation(model, X_train, y_train, cv=5, model_name=name)
        cv_results[name] = cv_scores['test_accuracy'].mean()

    best_model_name = max(cv_results, key=cv_results.get)
    best_model = final_models[best_model_name]

    print("\n   " + "-" * 60)
    print(f"   🏆 MIGLIOR MODELLO IN ADDESTRAMENTO: {best_model_name}")
    print(f"      Score medio di validazione: {cv_results[best_model_name]:.4f}")
    print("   " + "-" * 60)
    
    # -------------------------------------------------------------------------
    # FASI 11-12: Valutazione Finale su Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASI 11-12: EVALUATION FINALE E ANALISI AVANZATA")
    print("=" * 80)
    print("   🎯 Utilizzo del Test Set (dati mai visti dal modello)\n")
    
    results, _ = evaluate_models(final_models, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
    
    # Generazione grafici diagnostici (ROC, Calibrazione, Importanza)
    plot_roc_curves(final_models, X_test, y_test)
    plot_calibration_curves(final_models, X_test, y_test)
    
    if 'Random Forest' in best_model_name:
        analyze_feature_importance(best_model, feature_cols)
    else:
        print(f"\n   ℹ️ Grafico Feature Importance saltato (non supportato da {best_model_name})")

    # -------------------------------------------------------------------------
    # RIEPILOGO RISULTATI
    # -------------------------------------------------------------------------
    final_best_accuracy = results[best_model_name]
    
    print("\n" + "=" * 80)
    print(" 📊 RIEPILOGO RISULTATI DEL PROGETTO")
    print("=" * 80)
    print(f"   Incontri analizzati      : {len(df_clean)}")
    print(f"   Baseline (Casuale)       : {baseline:.2f}%")
    print(f"   Modello Campione         : {best_model_name}")
    print(f"   Accuracy Finale (Test)   : {final_best_accuracy:.4f} ({final_best_accuracy*100:.2f}%)")
    print(f"   Miglioramento vs Baseline: +{(final_best_accuracy - baseline/100)*100:.2f}%")
    
    gap = abs(cv_results[best_model_name] - final_best_accuracy)
    if gap < 0.02:
        print(f"\n   ✓ Generalizzazione ottima: gap Train-Test minimo ({gap:.4f})")
    else:
        print(f"\n   ⚠️ Sospetto Overfitting: rilevato gap Train-Test di {gap:.4f}")
    
    # -------------------------------------------------------------------------
    # FASE 13: Salvataggio Modelli
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FASE 13: ESPORTAZIONE MODELLI (DEPLOYMENT)")
    print("=" * 80)

    os.makedirs('models', exist_ok=True)
    print("\n   💾 Salvataggio in corso nella cartella 'models/'...")

    for name, model in final_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, f'models/{safe_name}.pkl')
        print(f"      ✅ {safe_name}.pkl salvato.")

    joblib.dump(scaler, 'models/scaler.pkl')
    print("      ✅ scaler.pkl salvato.")

    print("\n✓ PIPELINE COMPLETATA. Il sistema è pronto per simulazioni live.")
    print("=" * 80)

if __name__ == '__main__':
    main()
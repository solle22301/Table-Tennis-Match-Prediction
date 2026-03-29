"""
================================================================================
🏓 TABLE TENNIS MATCH PREDICTION
================================================================================
Autore: Alessandro Sollevanti
Data: Gennaio 2026
Corso: Ingegneria Informatica - Machine Learning

## OBIETTIVO DEL PROGETTO:
Questo progetto sviluppa un modello di Machine Learning per predire il vincitore di partite di tennis tavolo (circuito TT Elite Series) 
prima dell'inizio del match, utilizzando i dati storici dei giocatori come il divario di punteggio ELO, lo stato di forma recente, 
l'esito degli scontri diretti e il momentum (le serie di vittorie o sconfitte di fila).

Per fare le predizioni, sono stati addestrati e confrontati tre diversi algoritmi di classificazione: 
Logistic Regression, Random Forest e Neural Network (MLP). 
Infine, le valutazioni di questi tre algoritmi sono state unite in un quarto e ultimo modello, un Ensemble (Soft Voting Classifier), 
che ne calcola la media probabilistica per ottenere previsioni più stabili e ridurre il margine di errore.

## PIPELINE COMPLETA DEL PROGETTO:

###  DATA LOADING & PREPROCESSING (Fasi 1-3)
1. Caricamento Dataset
   - Import file partite (raccolta di 9415 match del torneo polacco "TT Elite Series" in una finestra temporale di un mese)
   - Import ranking ELO ufficiale (413 giocatori, rating 216-1520)

2. Preprocessing
   - Normalizzazione nomi (rimozione accenti, caratteri speciali)
   - Inversione formato "Cognome Nome" e risoluzione duplicati
   - Merge LEFT JOIN per associare l'ELO ai rispettivi giocatori

3. Feature Engineering
   - Creazione di 10 variabili predittive calcolate tramite "Rolling Window"
   - Utilizzo esclusivo dello storico antecedente al match (zero data leakage)

###  EXPLORATORY DATA ANALYSIS & OUTLIER DETECTION (Fase 4)
4. Analisi Esplorativa + Outlier Detection
   - Bilanciamento target, distribuzioni ELO e relazione differenza punteggio-vittoria
   - Outlier Detection: Studentized Residuals, Leverage e DFFITS
   - Rimozione sample anomali per aumentare la robustezza del modello

###  TRAIN/TEST SPLIT & FEATURE ANALYSIS (Fasi 5-6)
5. Split Temporale
   - Divisione 80/20 rigorosamente in ordine cronologico (Train sul passato, Test sul futuro)
   - Test set "congelato" fino alla valutazione finale

6. Analisi Correlazioni e Multicollinearità
   - Esecuzione SOLO sul train set per evitare data leakage
   - Matrice di correlazione e VIF (Variance Inflation Factor)
   - Rimozione delle feature ridondanti per pulire il modello predittivo

###  MODELING & VALIDATION (Fasi 7-10)
7. Preparazione Dati
   - Creazione target binario e normalizzazione con StandardScaler

8. Training Modelli
   - Modelli Base: Logistic Regression, Random Forest, Neural Network (MLP)
   - Modello Avanzato: Ensemble (Soft Voting) per mediare le probabilità dei 3 modelli

9. Cross-Validation (5-Fold su Train Set)
   - Valutazione robusta dei modelli senza barare guardando i dati di test
   - Selezione del miglior modello basata sui punteggi medi della CV

10. Hyperparameter Tuning
    - Ottimizzazione dei parametri del miglior modello tramite GridSearchCV

###  FINAL EVALUATION (Fasi 11-12)
11. Evaluation Finale su Test Set
    - Test set utilizzato per la PRIMA e UNICA volta
    - Valutazione tramite Accuracy, Precision, Recall, F1-Score e Confusion Matrix

12. Analisi Avanzata (Visualizzazioni)
    - Curve ROC (AUC) per confrontare la sensibilità/specificità dei modelli
    - Analisi di calibrazione delle probabilità (Brier Score)

## NOTE TECNICHE:
- Zero Data Leakage: Garantito dallo split temporale e dalle rolling window.
- Riproducibilità: Utilizzo di seed fissi (es. random_state=42).
================================================================================
"""

import os
import warnings
import joblib

# Definizione del percorso della cartella destinata ai grafici generati
plots_folder = "plots"

# Controllo dell'esistenza della cartella: se non presente, viene creata automaticamente
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Import moduli del progetto
from src.data_loader import load_data
from src.preprocessing import normalize_and_merge
from src.feature_engineering import create_features
from src.exploratory_analysis import run_eda, detect_outliers, remove_outliers
from src.modeling import train_test_split_temporal, prepare_data, train_models, hyperparameter_tuning
from src.evaluation import evaluate_models, plot_confusion_matrix
from src.advanced_analysis import (plot_correlation_matrix, calculate_vif, 
                                   plot_roc_curves, analyze_feature_importance,
                                   perform_cross_validation, plot_calibration_curves)

def main():
    # Header progetto
    print("=" * 80)
    print("TABLE TENNIS MATCH PREDICTION")
    print("Progetto Machine Learning - Alessandro Sollevanti")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # FASE 1: Caricamento Dataset
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 1: CARICAMENTO DATASET")
    print("=" * 80)
    
    #salvo i percorsi dei due dataset in due variabili per poi passarle alla funzione di caricamento dati
    matches_file = os.path.join('datasets', 'TT_Elite_COMBINED_9415_matches.csv')
    ranking_file = os.path.join('datasets', 'RANKING-TT-ELITE-SERIES.csv')
    
    # Caricamento dati
    df_matches, df_ranking = load_data(matches_file, ranking_file)
    
    # -------------------------------------------------------------------------
    # FASE 2: Preprocessing (Normalizzazione Nomi + Merge)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 2: PREPROCESSING")
    print("=" * 80)
    
    df_merged = normalize_and_merge(df_matches, df_ranking)
    
    # -------------------------------------------------------------------------
    # FASE 3: Feature Engineering
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    df_features = create_features(df_merged)
    
    # Pulizia dataset finale
    df_clean = df_features.dropna(subset=['player_1_elo', 'player_2_elo']).reset_index(drop=True)
    print(f"\nPulizia dataset:")
    print(f"   Righe con ELO mancante rimosse: {len(df_features) - len(df_clean)}")
    print(f"   Dataset pre-outlier removal: {len(df_clean)} partite")
    
    # -------------------------------------------------------------------------
    # FASE 4: Exploratory Data Analysis + Outlier Detection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 4: EXPLORATORY DATA ANALYSIS + OUTLIER DETECTION")
    print("=" * 80)
    
    # EDA standard
    baseline = run_eda(df_clean)
    
    # Outlier Detection (Sezione 4.6 Appunti - parte dell'EDA)
    print("\n" + "-" * 80)
    print("OUTLIER DETECTION (integrato nell'EDA)")
    print("-" * 80)
    
    # Definizione features per outlier detection (Nomi aggiornati)
    feature_cols_outlier = [
        'player_1_elo', 'elo_diff',
        'p1_win_rate_last5', 'p2_win_rate_last5',
        'p1_head_to_head_wins', 'p2_head_to_head_wins', 'p1_head_to_head_win_ratio',
        'p1_streak', 'p2_streak',
        'p1_form_volatility', 'p2_form_volatility'
    ]
    
    # Crea target temporaneo se non esiste
    if 'player_1_wins' not in df_clean.columns:
        df_clean['player_1_wins'] = (df_clean['winner'] == df_clean['player_1']).astype(int)
    
    # Estrai features e target per outlier detection
    X_outlier = df_clean[feature_cols_outlier].fillna(0).values
    y_outlier = df_clean['player_1_wins'].values
    
    # Rileva outliers
    outlier_result = detect_outliers(
        X_outlier, 
        y_outlier, 
        feature_names=feature_cols_outlier,
        verbose=True
    )
    
    # Rimuovi outliers
    df_clean = remove_outliers(df_clean, outlier_result['outlier_indices'], verbose=True)
    
    print(f"\n✓ Dataset finale pulito: {len(df_clean)} partite")
    
    # -------------------------------------------------------------------------
    # FASE 5: Split Temporale
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 5: SPLIT TEMPORALE")
    print("=" * 80)
    
    train, test = train_test_split_temporal(df_clean, test_size=0.2)
    
    # Definizione delle 10 features finali (Nomi aggiornati)
    feature_cols = [
        'elo_diff',
        'p1_win_rate_last5', 'p2_win_rate_last5',
        'p1_head_to_head_wins', 'p2_head_to_head_wins', 'p1_head_to_head_win_ratio',
        'p1_streak', 'p2_streak',
        'p1_form_volatility', 'p2_form_volatility'
    ]
    
    print(f"\nFeatures selezionate: {len(feature_cols)} (Dataset ottimizzato)")
    print("   Elenco esatto delle features rimosse per ottimizzazione (VIF Analysis):")
    print("     1. player_1_elo        (Rimosso per VIF > 10: manteniamo solo elo_diff)")
    print("     2. player_2_elo        (Rimosso per VIF infinito: ridondanza matematica perfetta)")
    print("     3. elo_sum             (Rimosso per correlazione col target quasi nulla: 0.01)")
    print("     4. p1_win_rate_overall (Rimosso per VIF 17: troppo correlato con la forma recente)")
    print("     5. p2_win_rate_overall (Rimosso per VIF 17: troppo correlato con la forma recente)")
    print("     6. p1_recent_form      (Rimosso per VIF 15: ridondante rispetto a win_rate_last5)")
    print("     7. p2_recent_form      (Rimosso per VIF 15: ridondante rispetto a win_rate_last5)")
    
    print("\n   Categorie finali mantenute (10 features):")
    print("     • ELO (1): elo_diff")
    print("     • Win Rate (2): p1_win_rate_last5, p2_win_rate_last5")
    print("     • Head-to-Head (3): p1_head_to_head_wins, p2_head_to_head_wins, p1_head_to_head_win_ratio")
    print("     • Momentum (2): p1_streak, p2_streak")
    print("     • Volatilità (2): p1_form_volatility, p2_form_volatility")
    
    # Verifica e gestione valori mancanti
    print(f"\nVerifica valori mancanti nelle features:")
    nan_found = False
    for col in feature_cols:
        missing = train[col].isna().sum()
        if missing > 0:
            print(f"  ⚠️  {col}: {missing} NaN ({missing/len(train)*100:.1f}%)")
            nan_found = True
    
    if nan_found:
        print(f"\n⚠️  Imputazione NaN con 0 (default per features rolling window)")
        train[feature_cols] = train[feature_cols].fillna(0)
        test[feature_cols] = test[feature_cols].fillna(0)
        print(f"✓ Imputazione completata")
    else:
        print(f"✓ Nessun valore mancante trovato")

    # -------------------------------------------------------------------------
    # FASE 6: Analisi Correlazioni (SOLO SU TRAIN SET)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 6: ANALISI CORRELAZIONI (TRAIN SET)")
    print("=" * 80)
    print("\n⚠️  IMPORTANTE: Analisi eseguita SOLO su train set per evitare data leakage")
    
    # Analisi SOLO su train
    corr_matrix = plot_correlation_matrix(train, feature_cols)
    vif_data = calculate_vif(train, feature_cols)
    
    # -------------------------------------------------------------------------
    # FASE 7: Preparazione Dati
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 7: PREPARAZIONE DATI")
    print("=" * 80)
    
    X_train, y_train, X_test, y_test, scaler = prepare_data(train, test, feature_cols)
    
    # -------------------------------------------------------------------------
    # FASE 8: Training Modelli Base
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 8: TRAINING MODELLI BASE")
    print("=" * 80)
    
    models = train_models(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # FASE 9: Cross-Validation (5-Fold su Train Set)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 9: CROSS-VALIDATION (TRAIN SET)")
    print("=" * 80)
    print("\n📊 Valutazione robusta dei modelli tramite 5-Fold CV")
    print("⚠️  Test set ancora NON utilizzato\n")

    # CV per ogni modello
    cv_results = {}
    for name, model in models.items():
        print(f"\nCross-Validation: {name}...")
        cv_scores = perform_cross_validation(model, X_train, y_train, cv=5)
        cv_results[name] = cv_scores['test_accuracy'].mean()

    # Selezione miglior modello basato su CV
    best_model_name = max(cv_results, key=cv_results.get)
    best_model = models[best_model_name]

    print("\n" + "=" * 80)
    print(f"✓ MIGLIOR MODELLO (basato su CV): {best_model_name}")
    print(f"  CV Score medio: {cv_results[best_model_name]:.4f}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # FASE 10: Hyperparameter Tuning (Opzionale)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 10: HYPERPARAMETER TUNING (OPZIONALE)")
    print("=" * 80)
    
    print("\nVuoi eseguire hyperparameter tuning sul miglior modello?")
    print("Nota: GridSearchCV con inner CV, può richiedere diversi minuti")
    response = input("Premi ENTER per saltare, 'si' per continuare: ").lower()
    
    if response == 'si':
        best_model_tuned = hyperparameter_tuning(X_train, y_train, best_model_name)
        
        if best_model_tuned is not None:
            # Sostituisci modello con versione ottimizzata
            models[best_model_name] = best_model_tuned
            best_model = best_model_tuned
            print(f"\n✓ Modello {best_model_name} ottimizzato e aggiornato")
    else:
        print("\n⏭️  Hyperparameter tuning saltato - uso modello base")
    
    # -------------------------------------------------------------------------
    # FASE 11: Evaluation Finale su Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 11: EVALUATION FINALE SU TEST SET")
    print("=" * 80)
    print("\n🎯 ATTENZIONE: Test set utilizzato per la PRIMA e UNICA volta\n")
    
    # Valuta TUTTI i modelli sul test set
    results, _ = evaluate_models(models, X_test, y_test)
    
    # Confusion matrix del modello finale
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
    
    # -------------------------------------------------------------------------
    # FASE 12: Analisi Avanzata (Visualizzazioni Finali)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 12: ANALISI AVANZATA")
    print("=" * 80)
    
    # ROC Curves per tutti i modelli
    plot_roc_curves(models, X_test, y_test)
    
    # Calibrazione e Brier Score
    plot_calibration_curves(models, X_test, y_test)
    
    # Feature importance (solo Random Forest)
    if 'Random Forest' in best_model_name:
        importance_df = analyze_feature_importance(best_model, feature_cols)
    else:
        print(f"\nFeature importance non disponibile per {best_model_name}")
        print("(disponibile solo per Random Forest)")

    
    # -------------------------------------------------------------------------
    # RIEPILOGO FINALE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PROGETTO COMPLETATO")
    print("=" * 80)
    
    final_best_accuracy = results[best_model_name]
    
    print("\n📊 RIEPILOGO RISULTATI:")
    print("-" * 80)
    print(f"   Dataset finale: {len(df_clean)} partite")
    print(f"   Outliers rimossi: {outlier_result['summary']['n_outliers']}")
    print(f"   Features utilizzate: {len(feature_cols)}")
    print(f"   Modelli testati: {len(models)}")
    print(f"\n   Baseline accuracy: {baseline:.2f}% (predizione casuale)")
    print(f"   Miglior modello: {best_model_name}")
    print(f"   CV Score (train): {cv_results[best_model_name]:.4f}")
    print(f"   Test Accuracy: {final_best_accuracy:.4f} ({final_best_accuracy*100:.2f}%)")
    print(f"   Miglioramento vs baseline: +{(final_best_accuracy - baseline/100)*100:.2f}%")
    
    # Verifica overfitting
    gap = abs(cv_results[best_model_name] - final_best_accuracy)
    if gap < 0.02:
        print(f"\n   ✓ Gap CV-Test: {gap:.4f} (modello generalizza bene)")
    else:
        print(f"\n   ⚠️  Gap CV-Test: {gap:.4f} (possibile overfitting)")
    
    print("-" * 80)
    
    print("\n📊 GRAFICI SALVATI:")
    print("   • target_balance.png - Bilanciamento classi")
    print("   • elo_distributions.png - Distribuzione ELO")
    print("   • elo_diff_vs_win.png - Relazione ELO vs vittoria")
    print("   • correlation_matrix.png - Matrice correlazioni (train set)")
    print("   • confusion_matrix_{}.png - Matrice confusione".format(best_model_name.replace(' ', '_')))
    print("   • roc_curves.png - ROC curves confronto modelli")
    print("   • calibration_curves.png - Calibrazione probabilità e Brier Score")
    if 'Random Forest' in best_model_name:
        print("   • feature_importance.png - Importanza features")
    
    print("\n✓ Pipeline completata con successo!")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # FASE 13: SALVATAGGIO MODELLI
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FASE 13: SALVATAGGIO MODELLI")
    print("=" * 80)

    # Crea cartella models se non esiste
    os.makedirs('models', exist_ok=True)

    # Salva tutti i modelli addestrati
    print("\n💾 Salvataggio modelli in cartella models/...")

    joblib.dump(models['Logistic Regression'], 'models/logistic_regression.pkl')
    print("   ✅ logistic_regression.pkl")

    joblib.dump(models['Random Forest'], 'models/random_forest.pkl')
    print("   ✅ random_forest.pkl")

    joblib.dump(models['Neural Network'], 'models/neural_network.pkl')
    print("   ✅ neural_network.pkl")
    
    joblib.dump(models['Ensemble Voting'], 'models/ensemble_voting.pkl')
    print("   ✅ ensemble_voting.pkl")

    joblib.dump(scaler, 'models/scaler.pkl')
    print("   ✅ scaler.pkl")

    print("\n✓ Modelli salvati con successo!")

if __name__ == '__main__':
    main()
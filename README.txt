# 🏓 Table Tennis Match Prediction (TT Elite Series)

**Autore:** Alessandro Sollevanti  
**Corso:** Ingegneria Informatica - Machine Learning  
**Data:** Gennaio 2026  

---

## OBIETTIVO DEL PROGETTO:
Questo progetto sviluppa un modello di Machine Learning per predire il vincitore di partite di tennis tavolo (circuito TT Elite Series) prima dell'inizio del match, 
utilizzando i dati storici dei giocatori come il divario di punteggio ELO, lo stato di forma recente, l'esito degli scontri diretti e il 
momentum (le serie di vittorie o sconfitte di fila).

Per fare le predizioni, sono stati addestrati e confrontati tre diversi algoritmi di classificazione: Logistic Regression, Random Forest e Neural Network (MLP). 
Infine, le valutazioni di questi tre algoritmi sono state unite in un quarto e ultimo modello, un Ensemble (Soft Voting Classifier), che ne calcola la media 
probabilistica per ottenere previsioni più stabili e ridurre il margine di errore.

---

## PIPELINE COMPLETA DEL PROGETTO:

### DATA LOADING & PREPROCESSING (Fasi 1-3)
1. **Caricamento Dataset**
   - Import file partite (raccolta di 9415 match del torneo polacco "TT Elite Series" in una finestra temporale di un mese).
   - Import ranking ELO ufficiale (413 giocatori, rating 216-1520).

2. **Preprocessing**
   - Normalizzazione nomi (rimozione accenti e caratteri speciali per uniformità).
   - Controllo e risoluzione automatica dei duplicati nel ranking.
   - Merge LEFT JOIN per associare l'ELO corretto ai rispettivi giocatori.

3. **Feature Engineering**
   - Creazione di 10 variabili predittive calcolate tramite "Rolling Window".
   - Utilizzo esclusivo dello storico antecedente al match per garantire **zero data leakage**.

### EXPLORATORY DATA ANALYSIS & OUTLIER DETECTION (Fase 4)
4. **Analisi Esplorativa + Outlier Detection**
   - Studio del bilanciamento target, distribuzioni ELO e relazione tra differenza punteggio e probabilità di vittoria.
   - Outlier Detection avanzata tramite algebra lineare (Studentized Residuals, Leverage e DFFITS).
   - Rimozione dei sample anomali critici per aumentare la robustezza dei modelli.

### TRAIN/TEST SPLIT & FEATURE ANALYSIS (Fasi 5-6)
5. **Split Temporale**
   - Divisione 80/20 eseguita rigorosamente in ordine cronologico (Train sul passato, Test sul futuro).
   - Test set "congelato" in cassaforte fino alla valutazione finale.

6. **Analisi Correlazioni e Multicollinearità**
   - Esecuzione SOLO sul train set per non "inquinare" l'addestramento.
   - Analisi Matrice di Correlazione di Pearson e VIF (Variance Inflation Factor).
   - Rimozione preventiva delle feature ridondanti per pulire il modello predittivo.

### MODELING & VALIDATION (Fasi 7-10)
7. **Preparazione Dati**
   - Creazione target binario e normalizzazione statistica tramite StandardScaler.

8. **Training Modelli Base**
   - Addestramento iniziale di Logistic Regression, Random Forest e Neural Network (MLP) utilizzando parametri standard per creare una baseline algoritmica.

9. **Hyperparameter Tuning**
   - Esecuzione della Grid Search su TUTTI i modelli base per esplorare diverse combinazioni di parametri e trovarne la configurazione ottimale.

10. **Creazione Ensemble e Cross-Validation**
    - Costruzione del Meta-Modello Ensemble (Soft Voting) unendo i 3 algoritmi portati al loro massimo potenziale.
    - Valutazione robusta di tutti i modelli tramite 5-Fold CV sul Train Set per testare la generalizzazione e decretare il modello vincitore prima del test finale.

### FINAL EVALUATION & DEPLOYMENT (Fasi 11-13)
11. **Evaluation Finale su Test Set**
    - Test set sbloccato e utilizzato per la PRIMA e UNICA volta.
    - Valutazione ufficiale tramite Accuracy, Precision, Recall, F1-Score e Matrice di Confusione.

12. **Analisi Avanzata (Visualizzazioni)**
    - Studio delle Curve ROC (AUC) per confrontare la capacità discriminativa dei modelli.
    - Analisi di calibrazione delle probabilità (Brier Score) per misurare l'affidabilità percentuale delle predizioni.
    - Feature Importance per analizzare il peso decisionale delle singole variabili.

13. **Salvataggio Modelli**
    - Esportazione dei modelli addestrati e dello scaler in file .pkl. In questo modo congelo il loro stato e sono pronti per fare predizioni dal vivo usando la console interattiva, senza doverli riaddestrare ogni volta.

---

## ⚙️ Le 10 Variabili Predittive Estratte
Per garantire la robustezza del modello, le features create si dividono in 5 categorie logiche:
* **Ranking (1):** `elo_diff` (Divario tecnico teorico).
* **Forma Recente (2):** `p1_win_rate_last5`, `p2_win_rate_last5`.
* **Scontri Diretti (3):** `p1_head_to_head_wins`, `p2_head_to_head_wins`, `p1_head_to_head_win_ratio`.
* **Momentum Psicologico (2):** `p1_streak`, `p2_streak` (Serie ininterrotte di risultati).
* **Affidabilità/Incostanza (2):** `p1_form_volatility`, `p2_form_volatility` (Deviazione standard dei risultati).

---

## 📈 Risultati Principali

* **Baseline Statistica (Caso):** 50.5%
* **Miglior Modello:** Ensemble Voting (Soft)
* **Accuratezza Finale (Test Set):** **60.5%**
* **Vantaggio Competitivo (Edge):** **+10.0%** rispetto alla baseline. Nelle scommesse sportive, questo margine certifica l'alto valore informativo delle variabili dinamiche introdotte.
* **Generalizzazione:** Gap tra Train (CV) e Test pari a 0.007, certificando l'assoluta assenza di Overfitting.

---

## 📂 Struttura del Repository

* `main.py`: Script principale che esegue l'intera pipeline end-to-end.
* `predict_match.py`: Interfaccia interattiva per simulazioni e predizioni dal vivo.
* `src/`: Cartella contenente i moduli core per la logica del progetto:
  * `data_loader.py`: Gestisce l'acquisizione dei dati grezzi e il caricamento dei CSV in DataFrame Pandas.
  * `preprocessing.py`: Si occupa della pulizia testuale, risoluzione duplicati e del merge (Left Join) tra match e ranking ELO.
  * `feature_engineering.py`: Implementa il calcolo delle 10 variabili dinamiche tramite rolling window, garantendo il paradigma "zero data leakage".
  * `exploratory_analysis.py`: Gestisce l'EDA, il bilanciamento classi, i plot delle distribuzioni ELO e l'outlier detection di base (IQR).
  * `advanced_analysis.py`: Esegue l'outlier detection avanzata tramite algebra lineare (Residui Studentizzati, Leverage, DFFITS).
  * `modeling.py`: Orchestra lo split temporale, lo scaling, il training dei tre modelli base, l'Hyperparameter Tuning (GridSearch) e la costruzione dell'Ensemble Soft Voting.
  * `evaluation.py`: Valuta i modelli sul test set calcolando metriche (Accuracy, F1), generando Matrici di Confusione, Curve ROC e grafici di calibrazione (Brier Score).
* `plots/`: Grafici generati in automatico (ROC Curves, Confusion Matrices, EDA).
* `models/`: Modelli addestrati serializzati in formato `.pkl`.
* `datasets/`: File CSV contenenti storico match e ranking.
* `requirements.txt`: Elenco delle librerie Python necessarie per l'esecuzione.

---

## 🚀 Installazione ed Esecuzione

1. Clonare il repository e spostarsi nella cartella:
   ```bash
   git clone [https://github.com/tuo-username/Table_Tennis_Match_Prediction.git](https://github.com/tuo-username/Table_Tennis_Match_Prediction.git)
   cd Table_Tennis_Match_Prediction

2. Installare le librerie necessarie:
    pip install -r requirements.txt

3. Avviare l'addestramento e la valutazione:
    python main.py

4. Effettuare una predizione testando il modello con giocatori reali:
    python predict_match.py
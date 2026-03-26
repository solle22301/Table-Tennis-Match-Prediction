# 🏓 Table Tennis Match Prediction

**Autore:** Alessandro Sollevanti  
**Corso:** Machine Learning  
**Data:** Gennaio 2026

## Descrizione Progetto

Sistema di Machine Learning per la predizione del vincitore di partite di tennis tavolo utilizzando dati storici e ranking ELO dei giocatori.

## Struttura Dataset

### Dataset Partite
- **File:** `TT_Elite_CLEAN_8178_matches.csv`
- **Righe:** 8,178 partite
- **Periodo:** 19 dicembre 2025 - 15 gennaio 2026
- **Colonne principali:**
  - `player_1`, `player_2`: Nomi giocatori
  - `winner`: Vincitore della partita
  - `date`: Data partita
  - `player_1_sets_won`, `player_2_sets_won`: Set vinti

### Dataset Ranking
- **File:** `RANKING-TT-ELITE-SERIES.csv`
- **Giocatori:** 413
- **Range ELO:** 216 - 1520
- **Colonne:** `Nome Giocatore`, `Rating ELO`

## Pipeline Progetto

### 1. Data Loading & Preprocessing
- Caricamento CSV con validazione colonne obbligatorie
- Normalizzazione nomi (rimozione accenti, pulizia caratteri speciali)
- Risoluzione inconsistenza formato nomi (Cognome Nome → Nome Cognome)
- Merge LEFT JOIN per mantenere tutte le partite

### 2. Feature Engineering
Creazione di 12 features suddivise in 4 categorie:

**Ranking (4 features):**
- `player_1_elo`, `player_2_elo`: Rating ELO individuali
- `elo_diff`: Differenza ELO (P1 - P2)
- `elo_sum`: Somma ELO (indicatore livello match)

**Win Rate (4 features):**
- `p1_win_rate_overall`, `p2_win_rate_overall`: % vittorie totali
- `p1_win_rate_last5`, `p2_win_rate_last5`: % vittorie ultime 5 partite

**Head-to-Head (2 features):**
- `h2h_p1_wins`: Vittorie P1 negli scontri diretti
- `h2h_total_matches`: Totale scontri diretti

**Momentum (2 features):**
- `p1_streak`, `p2_streak`: Streak vittorie/sconfitte consecutive

**Principio anti-leakage:** Rolling window che usa solo dati storici precedenti

### 3. Exploratory Data Analysis
- Verifica bilanciamento classi
- Distribuzione rating ELO (istogrammi + boxplot)
- Identificazione outliers (metodo IQR)
- Analisi relazione ELO difference vs probabilità vittoria

### 4. Modeling
**Split temporale:** 80% train, 20% test (evita data leakage)

**Modelli testati:**
- Logistic Regression
- Random Forest Classifier
- Neural Network (MLP)

### 5. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve e AUC
- Feature Importance (Random Forest)
- Cross-Validation K-Fold

### 6. Advanced Analysis
- Matrice di correlazione
- VIF (Variance Inflation Factor) per multicollinearità
- Analisi feature importance per gruppi

## Esecuzione

```bash
python main.py

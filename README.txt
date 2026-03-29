# 🏓 Analisi Predittiva Match Tennis Tavolo (TT Elite Series)

Studente: Alessandro Sollevanti  
Corso: Machine Learning  
Data: Gennaio 2026

---

## 📝 Panoramica del Progetto
L'obiettivo di questo lavoro è lo sviluppo di un sistema di intelligenza artificiale capace di prevedere l'esito di un incontro di tennis tavolo prima dell'inizio del match.

Il cuore del progetto risiede nella trasformazione di dati storici grezzi in **indicatori statistici dinamici**. A differenza dei modelli basati solo sulla classifica statica, questo sistema analizza lo stato di forma recente, la tenuta psicologica (**momentum**) e la consistenza delle performance degli atleti, garantendo una capacità predittiva superiore alla semplice probabilità calcolata sul Ranking ELO.

---

## I Dataset
Il sistema integra ed elabora due fonti di dati principali:

* **Match History** (`TT_Elite_COMBINED_9415_matches.csv`): Un archivio di oltre 9.000 incontri disputati nel circuito "TT Elite Series".
* **Ranking Ufficiale** (`RANKING-TT-ELITE-SERIES.csv`): La classifica con i punteggi di oltre 400 atleti (Range ELO: 216 - 1520).

**Data Cleaning:** È stata implementata una pipeline di normalizzazione testuale per gestire i caratteri speciali della lingua polacca e uniformare i formati dei nomi, garantendo un'associazione corretta tra i due dataset tramite **Left Join**.

---

## ⚙️ Feature Engineering: Le 10 Variabili Predittive
Per ogni match, il sistema ricostruisce il profilo statistico aggiornato dei due sfidanti utilizzando una tecnica di **Rolling Window (finestra mobile)**. 

Questo approccio garantisce **Zero Data Leakage**: il calcolo delle variabili avviene considerando esclusivamente i match disputati *prima* dell'incontro oggetto di predizione. In particolare, il sistema analizza l'intera storia pregressa per gli scontri diretti e si focalizza sugli ultimi 5 incontri per determinare lo stato di forma recente.

Le 10 variabili estratte e utilizzate per l'addestramento sono:

1.  **Classifica (1 Feature)**
    * `elo_diff`: Differenza tra il Rating ELO del Player 1 e del Player 2. Rappresenta il divario tecnico teorico.
2.  **Stato di Forma Recente (2 Features)**
    * `p1_win_rate_last5`: Percentuale di vittorie del Player 1 negli ultimi 5 match.
    * `p2_win_rate_last5`: Percentuale di vittorie del Player 2 negli ultimi 5 match.
3.  **Scontri Diretti - Head-to-Head (3 Features)**
    * `p1_head_to_head_wins`: Vittorie totali del Player 1 contro il Player 2.
    * `p2_head_to_head_wins`: Vittorie totali del Player 2 contro il Player 1.
    * `p1_head_to_head_win_ratio`: Rapporto di dominanza (Vittorie P1 / Totale scontri diretti).
4.  **Trend e Momentum (2 Features)**
    * `p1_streak`: Serie di risultati consecutivi (positivi per vittorie, negativi per sconfitte). Indica l'inerzia psicologica attuale.
    * `p2_streak`: Serie di risultati consecutivi del Player 2.
5.  **Volatilità e Affidabilità (2 Features)**
    * `p1_form_volatility`: Deviazione standard dei risultati. Indica quanto è "altalenante" il rendimento recente.
    * `p2_form_volatility`: Deviazione standard dei risultati del Player 2.

---

## Modellazione e Strategia Ensemble
Sono stati addestrati e validati tre modelli base con architetture differenti:

1.  **Logistic Regression**: Per modellare le relazioni lineari tra ELO e vittoria.
2.  **Random Forest Classifier**: Per catturare interazioni non lineari e gerarchiche.
3.  **Neural Network (MLP)**: Una rete neurale profonda per l'estrazione di pattern complessi.

**Meta-Modello:** Per massimizzare la robustezza, è stato implementato un quarto modello: **Soft Voting Classifier**. A differenza dell'Hard Voting (basato sulla maggioranza secca), il **Soft Voting** calcola la media ponderata delle probabilità predette dai singoli modelli, dando più peso alle previsioni "più sicure" e mitigando gli errori isolati.

---

## 📈 Risultati e Valutazione
L'efficacia del sistema è stata misurata rispetto a una soglia statistica di riferimento (**Baseline Accuracy**) pari al **50.5%** (frequenza della classe maggioritaria, in questo caso vittoria player 2).

* **Miglior Modello:** **Ensemble Voting (Soft)**.
* **Accuratezza Finale (Test Set):** **60.5%**. Il modello prevede correttamente oltre 6 match su 10.
* **Vantaggio sulla Baseline:** Un incremento di ben **+10%** conferma il valore informativo delle variabili dinamiche introdotte.
* **Diagnosi Overfitting:** Gap Train-Test minimo (**0.007**), certificando un'ottima capacità di generalizzazione.

### Perché l'Ensemble è risultato superiore?
* **Riduzione della Varianza:** Ha mostrato la deviazione standard più bassa nella Cross-Validation (**± 0.0075**), risultando il più solido ai cambi di dataset.
* **Compensazione degli Errori:** La mediazione tra modelli lineari, ad albero e neurali permette di bilanciare i "bias" individuali di ogni algoritmo.
* **Stabilità Statistica:** Ha mitigato il rumore dei singoli modelli, producendo una decisione finale più robusta.

---

## 📂 Struttura del Repository
* `main.py`: Script principale (Preprocessing -> EDA -> Training -> Evaluation).
* `predict_match.py`: Interfaccia interattiva per simulazioni in tempo reale.
* `src/`: Moduli per la logica di caricamento, ingegnerizzazione e analisi.
* `plots/`: Grafici generati (ROC Curves, Confusion Matrices, Feature Importance).
* `models/`: Modelli addestrati serializzati in formato `.pkl`.
* `requirements.txt`: Elenco delle dipendenze necessarie.

---

## 🚀 Installazione ed Esecuzione
1. Clonare il repository.
2. Creare un ambiente virtuale e installare le librerie:
   ```bash
   pip install -r requirements.txt
3. Training modelli e Valutazione:
	python main.py
4. Effettuare una predizione testando il modello con giocatori reali:
	python predict_match.py
"""
Modulo per la valutazione statistica dei modelli di Machine Learning.

Questo modulo si occupa di testare i modelli su dati mai visti prima (Test Set)
per misurarne la reale capacità di generalizzazione.

Metriche implementate:
- Accuracy: Percentuale totale di predizioni corrette.
- Precision: Delle partite in cui il modello ha predetto la vittoria del P1, quante erano corrette?
- Recall: Di tutte le partite realmente vinte dal P1, quante ne ha individuate il modello?
- F1-Score: Media armonica tra Precision e Recall (utile per valutazioni bilanciate).
- Confusion Matrix: Mappa visiva degli errori (Falsi Positivi vs Falsi Negativi).
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# ==============================================================================
# VALUTAZIONE METRICHE SUL TEST SET
# ==============================================================================

def evaluate_models(trained_models_dict, X_test_scaled, y_test):
    """
    Esegue il test finale di tutti i modelli addestrati.
    
    Args:
        trained_models_dict (dict): Dizionario {nome_modello: modello_addestrato}.
        X_test_scaled (array): Matrice delle features di test normalizzate.
        y_test (array): Vettore target reale del test set.
    
    Returns:
        tuple: (accuracy_results_dict, best_performing_model_name)
    """
    
    accuracy_results = {}
    
    # Itera su ogni modello addestrato
    for model_name, model in trained_models_dict.items():
        
        # Generazione delle predizioni
        y_pred = model.predict(X_test_scaled)
        
        # Calcolo dell'Accuracy complessiva
        current_accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[model_name] = current_accuracy
        
        # Formattazione dello stacco visivo del report per renderlo leggibile in console
        print(f"\n   {'='*50}")
        print(f"   🎯 Modello: {model_name.upper()}")
        print(f"   {'='*50}")
        print(f"   Accuracy Globale: {current_accuracy:.4f}\n")
        
        # Stampa del report dettagliato (Precision, Recall, F1)
        report = classification_report(
            y_test, 
            y_pred,
            target_names=['Player 2 Wins (0)', 'Player 1 Wins (1)'],
            digits=4
        )
        
        # Indentiamo il report per allinearlo al layout del terminale
        indented_report = "\n".join(["      " + line for line in report.split('\n')])
        print(indented_report)
    
    # Identificazione del modello con l'Accuracy più alta sul Test Set
    best_model_name = max(accuracy_results, key=accuracy_results.get)
    
    print("\n   " + "-" * 60)
    print(f"   🏆 VINCITORE SUL TEST SET (Dati Nuovi): {best_model_name}")
    print(f"      Accuracy Definitiva: {accuracy_results[best_model_name]:.4f}")
    print("   " + "-" * 60)
    
    return accuracy_results, best_model_name


# ==============================================================================
# ANALISI DEGLI ERRORI: CONFUSION MATRIX
# ==============================================================================

def plot_confusion_matrix(best_model, X_test_scaled, y_test, model_name):
    """
    Genera e salva la matrice di confusione per analizzare il tipo di errori
    commessi dal miglior modello.
    
    Struttura della Matrice (Target 1 = Player 1 Wins):
                 | Predetto P2 (0) | Predetto P1 (1) |
    -------------|-----------------|-----------------|
    Reale P2 (0) | True Negative   | False Positive  |
    -------------|-----------------|-----------------|
    Reale P1 (1) | False Negative  | True Positive   |
    """
    print(f"\n   -> Analisi Errori: Generazione Matrice di Confusione per '{model_name}'...")
    
    # Generazione predizioni
    y_pred = best_model.predict(X_test_scaled)
    
    # Calcolo della matrice
    cm = confusion_matrix(y_test, y_pred)
    
    # Estrazione dei 4 quadranti per l'analisi numerica
    true_negatives, false_positives, false_negatives, true_positives = cm.ravel()
    
    # -------------------------------------------------------------------------
    # Visualizzazione Grafica (Heatmap)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, 
                annot=True,              # Mostra i numeri dentro le celle
                fmt='d',                 # Formato intero (no decimali)
                cmap='Blues',            # Scala di blu
                xticklabels=['Predetto P2 Vince', 'Predetto P1 Vince'],
                yticklabels=['Reale P2 Vince', 'Reale P1 Vince'],
                cbar_kws={'label': 'Numero di Partite'})
    
    plt.title(f'Matrice di Confusione - {model_name}', fontweight='bold', fontsize=13)
    plt.xlabel('Predizione del Modello', fontsize=11)
    plt.ylabel('Risultato Reale', fontsize=11)
    plt.tight_layout()
    
    # Esportazione del file
    safe_filename = model_name.replace(" ", "_").lower()
    # Unisco la cartella "plots" al nome del file dinamico
    export_path = os.path.join("plots", f'confusion_matrix_{safe_filename}.png')
    plt.savefig(export_path, dpi=150)
    
    # Visualizzazione a schermo
    plt.show() 
    
    # -------------------------------------------------------------------------
    # Report Analitico degli Errori
    # -------------------------------------------------------------------------
    print(f"      ✓ Matrice di Confusione esportata: {export_path}")
    print("\n      Dettaglio degli Errori:")
    print(f"        True Negatives (TN):  {true_negatives} -> Predetto P2, ha vinto P2 ✓")
    print(f"        True Positives (TP):  {true_positives} -> Predetto P1, ha vinto P1 ✓")
    print(f"        False Positives (FP): {false_positives} -> Predetto P1, ma ha vinto P2 ❌ (Errore Tipo I)")
    print(f"        False Negatives (FN): {false_negatives} -> Predetto P2, ma ha vinto P1 ❌ (Errore Tipo II)")
    
    # Calcolo dei tassi di errore
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    false_negative_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
    
    print(f"\n      Tassi di Errore Specifici:")
    print(f"        False Positive Rate (FPR): {false_positive_rate*100:.2f}% (Tasso di sovrastima del P1)")
    print(f"        False Negative Rate (FNR): {false_negative_rate*100:.2f}% (Tasso di sottostima del P1)")
"""
Modulo per la valutazione dei modelli di Machine Learning.

Metriche implementate:
- Accuracy: percentuale predizioni corrette
- Precision: dei positivi predetti, quanti sono corretti
- Recall: dei positivi reali, quanti sono catturati
- F1-Score: media armonica di precision e recall
- Confusion Matrix: matrice errori per analisi dettagliata
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ==============================================================================
# VALUTAZIONE MODELLI
# ==============================================================================

def evaluate_models(models, X_test, t_test):
    """
    Valuta tutti i modelli sul test set e stampa metriche.
    
    Per ogni modello calcola:
    - Accuracy: (TP + TN) / Total
    - Precision per classe: TP / (TP + FP)
    - Recall per classe: TP / (TP + FN)
    - F1-Score per classe: 2 * (Precision * Recall) / (Precision + Recall)
    
    Dove:
    - TP = True Positive, TN = True Negative
    - FP = False Positive, FN = False Negative
    
    Args:
        models (dict): Dizionario {nome_modello: modello_addestrato}
        X_test (array): Features test normalizzate
        t_test (array): Target test
    
    Returns:
        tuple: (results, best_model_name)
               results = dict {nome: accuracy}
               best_model_name = nome del modello con accuracy migliore
    """
    print("\n" + "=" * 80)
    print("EVALUATION - TEST SET")
    print("=" * 80)
    
    results = {}
    
    # Valuta ogni modello
    for name, model in models.items():
        # Predizioni sul test set
        t_pred = model.predict(X_test)
        
        # Calcola accuracy
        acc = accuracy_score(t_test, t_pred)
        results[name] = acc
        
        # Stampa risultati
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        
        # Classification report dettagliato
        # Mostra precision, recall, f1-score per ogni classe
        print(classification_report(
            t_test, t_pred,
            target_names=['Player 2 Wins', 'Player 1 Wins'],
            digits=2
        ))
    
    # Identifica miglior modello
    best_model_name = max(results, key=results.get)
    
    print("=" * 80)
    print(f"MIGLIOR MODELLO: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]:.4f}")
    print("=" * 80)
    
    return results, best_model_name


# ==============================================================================
# CONFUSION MATRIX
# ==============================================================================

def plot_confusion_matrix(model, X_test, t_test, model_name):
    """
    Visualizza confusion matrix del modello.
    
    Confusion Matrix:
    
                      Predicted
                   P2 Wins  P1 Wins
    Actual P2 Wins    TN       FP
           P1 Wins    FN       TP
    
    Dove:
    - TN (True Negative): predetto P2 vince, effettivo P2 vince ✓
    - FP (False Positive): predetto P1 vince, effettivo P2 vince ✗
    - FN (False Negative): predetto P2 vince, effettivo P1 vince ✗
    - TP (True Positive): predetto P1 vince, effettivo P1 vince ✓
    
    Utile per:
    - Identificare tipo di errori (FP vs FN)
    - Capire se modello è biased verso una classe
    - Analizzare pattern di errore
    
    Args:
        model: Modello addestrato
        X_test (array): Features test
        t_test (array): Target test
        model_name (str): Nome del modello per titolo e filename
    
    Output:
        Salva grafico 'confusion_matrix_{model_name}.png'
    """
    # Predizioni
    t_pred = model.predict(X_test)
    
    # Calcola confusion matrix
    cm = confusion_matrix(t_test, t_pred)
    
    # -------------------------------------------------------------------------
    # Visualizzazione con heatmap
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # Heatmap con annotazioni
    sns.heatmap(cm, 
                annot=True,              # mostra numeri nelle celle
                fmt='d',                 # formato integer
                cmap='Blues',            # colormap blu
                xticklabels=['Player 2 Wins', 'Player 1 Wins'],
                yticklabels=['Player 2 Wins', 'Player 1 Wins'],
                cbar_kws={'label': 'Numero Partite'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=13)
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    
    # Salva grafico
    filename = f'confusion_matrix_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150)
    print(f"\n✓ Confusion matrix salvata: {filename}")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Analisi numerica confusion matrix
    # -------------------------------------------------------------------------
    tn, fp, fn, tp = cm.ravel()
    
    print("\nAnalisi Confusion Matrix:")
    print(f"   True Negatives (TN):  {tn} - Predetto P2, effettivo P2 ✓")
    print(f"   False Positives (FP): {fp} - Predetto P1, effettivo P2 ✗")
    print(f"   False Negatives (FN): {fn} - Predetto P2, effettivo P1 ✗")
    print(f"   True Positives (TP):  {tp} - Predetto P1, effettivo P1 ✓")
    
    # Calcola tassi di errore
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    print(f"\nTassi di errore:")
    print(f"   False Positive Rate: {fpr*100:.2f}%")
    print(f"   False Negative Rate: {fnr*100:.2f}%")

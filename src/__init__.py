"""
Table Tennis Match Prediction
Progetto Machine Learning - Alessandro Sollevanti

Moduli disponibili:
- data_loader: Caricamento e validazione dataset
- preprocessing: Normalizzazione nomi e merge
- feature_engineering: Creazione features con rolling window
- exploratory_analysis: Analisi esplorativa dati
- modeling: Training e preparazione modelli
- evaluation: Valutazione performance
- advanced_analysis: Analisi avanzate (correlazioni, VIF, ROC, feature importance)
"""

__version__ = "1.0.0"
__author__ = "Alessandro Sollevanti"
__email__ = "alessandro.sollevanti@studenti.unipg.it"

# Lista moduli esportati
__all__ = [
    'data_loader',
    'preprocessing',
    'feature_engineering',
    'exploratory_analysis',
    'modeling',
    'evaluation',
    'advanced_analysis'
]

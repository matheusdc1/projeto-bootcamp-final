"""
Módulo para avaliação de modelos
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def avaliar_modelo_completo(y_true, y_pred, y_pred_proba, failure_names):
    """Avaliação completa do modelo multirrótulo"""
    resultados = {}
    
    f1_scores = []
    for i, name in enumerate(failure_names):
        f1 = f1_score(y_true.iloc[:, i], y_pred[:, i], zero_division=0)
        precision = precision_score(y_true.iloc[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_true.iloc[:, i], y_pred[:, i], zero_division=0)
        
        resultados[name] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'casos_teste': y_true.iloc[:, i].sum()
        }
        
        f1_scores.append(f1)
    
    resultados['f1_medio'] = np.mean(f1_scores)
    
    return resultados

def calcular_roi(performance_metrics, params_financeiros):
    """Calcula ROI baseado na performance do modelo"""
    recall_medio = np.mean([m['recall'] for m in performance_metrics.values() if isinstance(m, dict)])
    
    # Cálculos conservadores
    falhas_detectadas = params_financeiros['falhas_ano'] * recall_medio
    economia_paradas = falhas_detectadas * params_financeiros['custo_parada']
    
    economia_anual = economia_paradas - params_financeiros['custo_sistema']
    roi = economia_anual / params_financeiros['investimento_inicial'] * 100
    
    return {
        'economia_anual': economia_anual,
        'roi_anual': roi,
        'falhas_detectadas': falhas_detectadas
    }
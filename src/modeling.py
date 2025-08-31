"""
Módulo para modelagem - Random Forest vs XGBoost
"""
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

def calc_scale_pos_weight(y_col):
    """Calcula scale_pos_weight para XGBoost"""
    neg = (y_col == 0).sum()
    pos = (y_col == 1).sum()
    return neg / max(pos, 1)

def criar_modelo_rf():
    """Cria modelo Random Forest otimizado para dados desbalanceados"""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    return MultiOutputClassifier(rf, n_jobs=-1)

def criar_modelos_xgb(y_train, failure_names):
    """Cria modelos XGBoost individuais para cada tipo de falha"""
    xgb_models = {}
    
    for i, name in enumerate(failure_names):
        scale_weight = calc_scale_pos_weight(y_train.iloc[:, i])
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_models[name] = xgb_model
    
    return xgb_models

def treinar_xgb_models(xgb_models, X_train, y_train, failure_names):
    """Treina todos os modelos XGBoost"""
    for i, name in enumerate(failure_names):
        xgb_models[name].fit(X_train, y_train.iloc[:, i])
    
    return xgb_models

def fazer_predicoes_xgb(xgb_models, X_test, failure_names):
    """Faz predições com modelos XGBoost"""
    predictions = np.zeros((len(X_test), len(failure_names)))
    probabilities = []
    
    for i, name in enumerate(failure_names):
        predictions[:, i] = xgb_models[name].predict(X_test)
        probabilities.append(xgb_models[name].predict_proba(X_test)[:, 1])
    
    probabilities = np.array(probabilities).T
    
    return predictions, probabilities

def comparar_modelos(y_test, rf_pred, xgb_pred, failure_names):
    """Compara performance de RF vs XGBoost"""
    rf_f1_scores = []
    xgb_f1_scores = []
    
    for i in range(len(failure_names)):
        rf_f1 = f1_score(y_test.iloc[:, i], rf_pred[:, i], zero_division=0)
        xgb_f1 = f1_score(y_test.iloc[:, i], xgb_pred[:, i], zero_division=0)
        
        rf_f1_scores.append(rf_f1)
        xgb_f1_scores.append(xgb_f1)
    
    rf_f1_avg = np.mean(rf_f1_scores)
    xgb_f1_avg = np.mean(xgb_f1_scores)
    
    melhor_modelo = "XGBoost" if xgb_f1_avg > rf_f1_avg else "Random Forest"
    
    return {
        'rf_f1_scores': rf_f1_scores,
        'xgb_f1_scores': xgb_f1_scores,
        'rf_f1_avg': rf_f1_avg,
        'xgb_f1_avg': xgb_f1_avg,
        'melhor_modelo': melhor_modelo
    }

def otimizar_thresholds(y_true, y_pred_proba, failure_names):
    """Otimiza thresholds para cada tipo de falha"""
    thresholds_otimizados = []
    f1_scores_otimizados = []
    
    for i, name in enumerate(failure_names):
        # Definir range de thresholds
        if name in ['FDF', 'FA']:  # Tipos raros
            thresholds = np.arange(0.05, 0.5, 0.05)
        else:  # Tipos mais comuns
            thresholds = np.arange(0.1, 0.8, 0.05)
        
        melhor_f1 = 0
        melhor_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba[:, i] >= threshold).astype(int)
            f1 = f1_score(y_true.iloc[:, i], y_pred_thresh, zero_division=0)
            
            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_threshold = threshold
        
        thresholds_otimizados.append(melhor_threshold)
        f1_scores_otimizados.append(melhor_f1)
    
    return thresholds_otimizados, f1_scores_otimizados

def fazer_predicao_final(modelo, modelo_tipo, X, thresholds=None):
    """Faz predição final com thresholds otimizados"""
    if modelo_tipo == "Random Forest":
        pred_proba = modelo.predict_proba(X)
        # Converter formato das probabilidades
        pred_proba_clean = np.zeros((len(X), len(pred_proba)))
        for i in range(len(pred_proba)):
            pred_proba_clean[:, i] = pred_proba[i][:, 1]
    else:  # XGBoost
        pred_proba_clean = []
        for name, xgb_model in modelo.items():
            pred_proba_clean.append(xgb_model.predict_proba(X)[:, 1])
        pred_proba_clean = np.array(pred_proba_clean).T
    
    if thresholds is None:
        thresholds = [0.5] * pred_proba_clean.shape[1]
    
    # Aplicar thresholds
    pred_final = np.zeros_like(pred_proba_clean)
    for i, threshold in enumerate(thresholds):
        pred_final[:, i] = (pred_proba_clean[:, i] >= threshold).astype(int)
    
    return pred_final, pred_proba_clean
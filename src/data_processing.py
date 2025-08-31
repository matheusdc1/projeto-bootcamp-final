"""
Módulo para processamento de dados
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def limpar_rotulos(df, failure_cols):
    """Limpa e padroniza rótulos de falha"""
    def limpar_rotulo(valor):
        valor_str = str(valor).lower().strip()
        if valor_str in ['não', 'n', 'false', '0', 'nan']:
            return 0
        else:
            return 1
    
    for col in failure_cols:
        df[col] = df[col].apply(limpar_rotulo)
    
    return df

def criar_features_engineered(df):
    """Cria features engineered baseadas no conhecimento do domínio"""
    # Tratar missing values
    if df['desgaste_da_ferramenta'].isnull().sum() > 0:
        df['desgaste_da_ferramenta'] = df.groupby('tipo')['desgaste_da_ferramenta'].transform(
            lambda x: x.fillna(x.median())
        )
        df['desgaste_da_ferramenta'].fillna(df['desgaste_da_ferramenta'].median(), inplace=True)
    
    # Features engineered
    df['stress_termico'] = df['temperatura_processo'] - df['temperatura_ar']
    df['potencia_mecanica'] = df['torque'] * df['velocidade_rotacional'] / 1000
    df['taxa_desgaste'] = df['desgaste_da_ferramenta'] / (df['velocidade_rotacional'] + 1)
    df['stress_mecanico'] = (df['torque'] * df['desgaste_da_ferramenta']) / (df['velocidade_rotacional'] + 1)
    
    return df

def preprocessar_dados(df, feature_cols_final, scaler=None, fit_scaler=True):
    """Preprocessa dados finais"""
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['tipo'], prefix='tipo')
    
    # Selecionar features
    X = df_encoded[feature_cols_final]
    
    # Normalizar
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    return X_scaled, scaler
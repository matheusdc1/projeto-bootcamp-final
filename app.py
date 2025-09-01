import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# Carregar modelo treinado

modelo_data = joblib.load("models/modelo_final_bootcamp.pkl")
modelo_nome = modelo_data['modelo_tipo']
modelo = modelo_data['modelo']
scaler = modelo_data['scaler']
feature_names = modelo_data['feature_names']
failure_names = modelo_data['failure_names']
thresholds = modelo_data['thresholds_otimizados']


# Sidebar - Navegação

st.sidebar.title("🔧 Manutenção Preditiva")
page = st.sidebar.radio("Navegação", [
    "📊 Distribuição das Falhas",
    "🤖 Comparação de Modelos",
    "🚨 Predições em Novos Dados",
    "💰 Análise de Negócio",
    "🚀 Próximos Passos"
])

# Página 1 - Distribuição das Falhas
if page == "📊 Distribuição das Falhas":
    st.title("Distribuição das Falhas")
    st.write("Aqui mostramos como as falhas estão distribuídas no dataset.")

    df = pd.read_csv("data/X_processed.csv")  # use seu arquivo real
    y = pd.read_csv("data/y_processed.csv")   # rótulos das falhas

    failure_cols = y.columns.tolist()  # ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    counts = y.sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(counts.index, counts.values, color="red", alpha=0.7)
    ax.set_title("Número de falhas por tipo", fontsize=14)
    ax.set_ylabel("Quantidade", fontsize=12)

    # Rotacionar rótulos do eixo x para não sobrepor
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=10)

    # Adicionar números e percentuais nas barras
    total = len(y)
    for bar, col in zip(bars, counts.index):
        value = counts[col]
        pct = value / total * 100
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            value + max(counts.values)*0.01, 
            f"{value} ({pct:.1f}%)", 
            ha='center', va='bottom', fontweight='bold'
        )

    st.pyplot(fig)


# Página 2 - Comparação de Modelos

elif page == "🤖 Comparação de Modelos":
    st.title("Comparação de Modelos")
    st.write("Resultados obtidos com Random Forest e XGBoost (F1 final com thresholds otimizados).")

    modelos = ["Random Forest", "XGBoost"]
    f1_final = [0.193, 0.222]  # RF e XGBoost com threshold otimizado

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(modelos, f1_final, color=["skyblue", "orange"], alpha=0.8)
    ax.set_ylabel("F1-score final")
    ax.set_ylim(0, 0.25)
    ax.set_title("Comparação de Modelos - F1 final com threshold otimizado")

    # Adicionar valores acima das barras
    for bar, valor in zip(bars, f1_final):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            valor + 0.005,
            f"{valor:.3f}",
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    st.pyplot(fig)


# Página 3 - Predições em Novos Dados

elif page == "🚨 Predições em Novos Dados":
    st.title("Predições em Novos Dados")
    st.write("Faça upload de um arquivo CSV para prever falhas e identificar casos críticos.")

    uploaded_file = st.file_uploader("Carregar CSV", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)

        # Pré-processamento: criar features usadas no treino
        if 'desgaste_da_ferramenta' in new_data.columns:
            new_data['desgaste_da_ferramenta'] = new_data.groupby('tipo')['desgaste_da_ferramenta'].transform(
                lambda x: x.fillna(x.median())
            )
            new_data['desgaste_da_ferramenta'].fillna(new_data['desgaste_da_ferramenta'].median(), inplace=True)

        new_data['stress_termico'] = new_data['temperatura_processo'] - new_data['temperatura_ar']
        new_data['potencia_mecanica'] = new_data['torque'] * new_data['velocidade_rotacional'] / 1000
        new_data['taxa_desgaste'] = new_data['desgaste_da_ferramenta'] / (new_data['velocidade_rotacional'] + 1)
        new_data['stress_mecanico'] = (new_data['torque'] * new_data['desgaste_da_ferramenta']) / (new_data['velocidade_rotacional'] + 1)

        # One-hot encoding para coluna tipo
        new_data_encoded = pd.get_dummies(new_data, columns=['tipo'], prefix='tipo')

        # Adicionar colunas faltantes
        for col in feature_names:
            if col not in new_data_encoded.columns:
                new_data_encoded[col] = 0

        X_new_scaled = scaler.transform(new_data_encoded[feature_names])

        # Predição
        if modelo_nome == "Random Forest":
            pred_proba_raw = modelo.predict_proba(X_new_scaled)
            pred_proba = np.zeros((len(X_new_scaled), len(failure_names)))
            for i in range(len(failure_names)):
                pred_proba[:, i] = pred_proba_raw[i][:, 1]
        else:
            pred_proba_list = [modelo[name].predict_proba(X_new_scaled)[:, 1] for name in failure_names]
            pred_proba = np.array(pred_proba_list).T

        # Aplicar thresholds
        pred_bin = np.zeros_like(pred_proba, dtype=int)
        for i, t in enumerate(thresholds):
            pred_bin[:, i] = (pred_proba[:, i] >= t).astype(int)

        # Adicionar coluna de risco máximo
        new_data['Risco (%)'] = (pred_proba.max(axis=1) * 100).round(2)
        new_data['Falhas Previstas'] = pred_bin.sum(axis=1)

        st.write("### Casos críticos (risco > 70%)")
        st.dataframe(new_data[new_data['Risco (%)'] > 70])


# Página 4 - Análise de Negócio

elif page == "💰 Análise de Negócio":
    st.title("Análise de Negócio")
    st.write("""
    Simulação de impacto financeiro usando dados fictícios:
    - Economia anual estimada: **R$ 250.000**  
    - ROI no primeiro ano: **40%**  
    - Payback estimado: **2 anos**  
    """)

    df_roi = pd.DataFrame({
        "Ano": [1, 2, 3, 4, 5],
        "ROI acumulado (%)": [40, 85, 120, 160, 200]
    })
    st.line_chart(df_roi.set_index("Ano"))


# Página 5 - Próximos Passos

elif page == "🚀 Próximos Passos":
    st.title("Próximos Passos")
    st.markdown("""
    - Balanceamento das classes raras  
    - Testar novos algoritmos (LightGBM, redes neurais)  
    - Refinar thresholds para reduzir falsos positivos  
    - Integrar o modelo em API para uso em produção  
    - Coletar mais dados históricos para melhorar desempenho  
    """)

# 🏭 Manutenção Preditiva Industrial – Projeto Bootcamp

## 📄 Descrição do Projeto
Este projeto implementa um **sistema de manutenção preditiva**.
O objetivo é detectar falhas em máquinas antes que ocorram, reduzindo paradas não planejadas e custos com manutenção corretiva.  

O projeto está em fase inicial, mas já gera **insights valiosos** sobre os tipos de falhas, performance dos modelos e impacto financeiro simulado.

---

## 🎯 Objetivos
- 🔧 Detectar falhas em equipamentos com antecedência  
- 📊 Priorizar manutenção preventiva em casos críticos  
- 🧠 Avaliar performance de modelos de classificação multirrótulo  
- 💰 Simular impacto financeiro da manutenção preditiva  
- 🚀 Criar uma base para evolução contínua do projeto  

---

## ⚠️ Características do Problema
- **Multirrótulo:** uma mesma peça/máquina pode apresentar múltiplas falhas simultaneamente  
- **Classes desbalanceadas:** algumas falhas ocorrem raramente, exigindo atenção especial no treino  
- **Métrica escolhida:** F1-Score, que equilibra precisão e recall, priorizando a detecção de falhas críticas  

---

## 📁 Estrutura do Repositório

- `/data`  
  - `/raw`  
    - `bootcamp_train.csv`  
    - `bootcamp_test.csv`  
  - `/processed`  
    - `X_processed.csv`  
    - `y_processed.csv`  
- `/models`  
  - `modelo_final_bootcamp.pkl`  
- `/notebooks`  
  - `01_data_processing.ipynb`  
  - `02_feature_engineering.ipynb`  
  - `03_analise_negocio.ipynb`  
  - `04_predicoes_finais.ipynb`  
- `/results`  
  - `resumo_negocio.json`  
  - `predicoes_bootcamp_test.csv`  
  - `casos_criticos.csv`  
  - `predicoes_para_api.csv`  
- `README.md`  
- `requirements.txt`  

---

## 📝 Notebook Overview

### 1️⃣ `01_data_processing.ipynb`
- Limpeza e normalização dos dados  
- Criação de colunas alvo (falhas)  
- Salva arquivos processados (`X_processed.csv` e `y_processed.csv`)  

### 2️⃣ `02_feature_engineering.ipynb`
- Criação de novas features:  
  - `stress_termico` 🌡️  
  - `potencia_mecanica` ⚡  
  - `taxa_desgaste` 🔧  
  - `stress_mecanico` 🏋️‍♂️  
- Normalização e preparação dos dados para treino  

### 3️⃣ `03_analise_negocio.ipynb`
- Análise de distribuição das falhas 📊  
- Comparação de modelos (Random Forest vs XGBoost) 🌳⚡  
- Simulação de impacto financeiro 💰  
- Identificação de limitações e próximos passos  
- Exporta resumo em JSON (`resumo_negocio.json`)  

### 4️⃣ `04_predicoes_finais.ipynb`
- Aplicação do modelo final em dados de teste 🧪  
- Geração de predições binárias e probabilísticas  
- Identificação de casos críticos (alto risco 🚨)  
- Exporta resultados para CSV para análise e API  

---

## 📊 Resultados
- **F1-score médio (cross-validation):** 0.230  
- **F1-score final XGBoost (threshold otimizado):** 0.222  
- **Comparação RF vs XGBoost:** RF 0.193 | XGBoost 0.195  
- **Casos críticos (>70% risco):** 10 amostras  
- **Impacto financeiro simulado:**  
  - Economia anual: R$ 200.000 💵  
  - ROI 1º ano: 20% 📈  
  - Payback estimado: 2,5 anos ⏳  

---

## 📈 Visualizações Incluídas
- Distribuição de falhas por tipo  
- Performance por modelo e por tipo de falha  
- Distribuição de risco das amostras de teste  
- Comparação de custos anuais e ROI acumulado  

---

## 🚀 Como Usar

1. Clone este repositório:
```bash
git clone https://github.com/matheusdc1/projeto-bootcamp-final.git
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Para usar a dashboard execute no terminal:
```bash
streamlit run app.py
```

Execute os notebooks na seguinte ordem:

- 01_data_processing.ipynb      # Limpeza e preparação dos dados
- 02_feature_engineering.ipynb  # Criação de features e normalização
- 03_analise_negocio.ipynb      # Análise de negócio e simulação financeira
- 04_predicoes_finais.ipynb     # Predições finais e identificação de casos críticos

Os resultados serão gerados na pasta results/:

- predicoes_bootcamp_test.csv   # Predições completas das falhas
- casos_criticos.csv            # Amostras com alto risco (>30%)
- predicoes_para_api.csv         # Predições prontas para API
- resumo_negocio.json           # Resumo do modelo, métricas e impacto financeiro

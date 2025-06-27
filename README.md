# Forecasting de Investimento em Marketing com Séries Temporais e MLOps no Azure

## 1. Sumário do Projeto

Este projeto implementa uma solução de ponta a ponta para a previsão de gastos diários em campanhas de marketing digital. Utilizando dados históricos extraídos via API da Meta, foi construído um pipeline automatizado no Azure Machine Learning para processar os dados, realizar engenharia de features e treinar um modelo de Machine Learning para prever investimentos futuros.

O objetivo principal é fornecer uma ferramenta de forecasting que permita um planejamento orçamentário mais inteligente, otimização de campanhas e uma tomada de decisão orientada por dados.

---

## 2. Arquitetura da Solução e Tecnologias

A solução foi desenvolvida como um pipeline de MLOps modular e reutilizável, orquestrado inteiramente no **Azure Machine Learning**.

* **Plataforma Cloud:** Microsoft Azure
* **Orquestração:** Azure ML Pipelines & Components (SDK v2)
* **Segurança:** Azure Key Vault para gerenciamento de chaves de API.
* **Computação:** Azure ML Compute Cluster.
* **Modelo:** XGBoost (`XGBRegressor`).
* **Bibliotecas Principais:** Python, Pandas, Scikit-learn, MLflow, Requests.
* **Fonte de Dados:** API de Marketing da Meta (Facebook Ads).

---

## 3. Metodologia

O projeto foi dividido em três componentes principais, formando um pipeline automatizado:

![arquitetura-pipeline](/artefacts/pipeline.png)

### a) Componente 1: Extração de Dados (`get_data`)

* Conecta-se de forma segura ao **Azure Key Vault** para recuperar o token de acesso da API.
* Executa chamadas paginadas à API de Insights da Meta para extrair dados diários de desempenho (gasto, impressões, cliques, alcance) no período de um ano.
* Salva os dados brutos como um artefato no Azure ML, garantindo o versionamento e a rastreabilidade da fonte de dados.

### b) Componente 2: Engenharia de Features (`process_data`)

* Recebe os dados brutos do componente anterior.
* Realiza a limpeza e transformação dos dados, convertendo as datas para um formato apropriado.
* Cria um conjunto robusto de features de séries temporais para enriquecer o modelo, incluindo:
    * **Features de Calendário:** Dia da semana, dia do mês, mês, semana do ano.
    * **Features de Lag:** Valores de gasto de dias anteriores (ex: D-1, D-7).
    * **Features de Janela Móvel:** Média de gastos em uma janela de 7 dias para suavizar tendências.

### c) Componente 3: Treinamento e Avaliação (`train_model`)

* Utiliza os dados processados para treinar um modelo `XGBRegressor`.
* A divisão dos dados é feita temporalmente, utilizando os 14 dias mais recentes como conjunto de teste para simular um cenário de previsão real.
* Utiliza a técnica de *early stopping* para otimizar o treinamento e evitar overfitting.
* Avalia o modelo, registra as métricas de performance, um gráfico de resultados e o modelo treinado no **MLflow**, garantindo total reprodutibilidade.

---

## 4. Resultados do Modelo XGBoost

O modelo treinado alcançou **primeiramente** os seguintes resultados no conjunto de teste (últimos 14 dias):

* **Mean Absolute Error (MAE):** 16.8
* **Root Mean Squared Error (RMSE):** 21.5

_Interpretação: Em média, as previsões de gasto diário do modelo erraram por R$ 16,80. O RMSE, por penalizar mais os erros maiores, também apresenta um valor controlado, indicando que o modelo não cometeu erros excessivamente grandes._

### Análise Visual da Previsão

O gráfico abaixo compara os gastos reais (linha azul) com as previsões do modelo (linha tracejada laranja) para o período de teste. Observa-se que o modelo foi capaz de capturar bem um certo padrão dos dados, mais falhou no dia 25 de dezembro, não percebendo o feriado.

![grafico-previsao-real](/artefacts/previsao_vs_real.png)


### Engenharia de Features para Sazonalidade

Após identificar problemas de sazonalidade, trabalhei na construção de novas features para melhorar nosso modelo, utilizando a biblioteca `holidays` para identificar feriados brasileiros durante o ano gerando uma nova feature. Posteriormente, novas features cíclicas com `sen` e `cos` para o modelo entender que o dia 6 (Domingo) está perto do dia 0 (Segunda), ou que o mês 12 (Dezembro) está perto do mês 1 (Janeiro). Segue os novos resultados:

* **Mean Absolute Error (MAE):** 10.5
* **Root Mean Squared Error (RMSE):** 13.3

![grafico-previsao-real](/artefacts/previsao_vs_real_2.png)

_Interpretação: Em média, as previsões de gasto diário do modelo erraram por R$ 10,50. Isso indica uma grande melhoria e precisão do nosso modelo durante as novas previsões._

--- 

## 5. Resultados do Modelo SARIMAX

Para otimizar o desempenho do SARIMAX, foi incorporado novas features: `cpc` (custo por clique) e `ctr` (taxa de cliques). Além disso, utilizamos a técnica de Grid Search para sistematicamente explorar e identificar os melhores parâmetros para o modelo SARIMAX, garantindo sua calibração ideal para os dados em questão.

* **Mean Absolute Error (MAE):** 14.7
* **Root Mean Squared Error (RMSE):** 17.8

![grafico-previsao-sarimax](/artefacts/previsao_sarimax.png)


---

## Conclusão da Análise Comparativa

As análises e experimentos demonstraram que o modelo **XGBoost** se manteve consistentemente **superior e mais eficaz** na previsão do **gasto (`spend`)** dos anúncios quando comparado ao modelo SARIMAX.

Apesar da aplicação de novas *features* e da otimização de parâmetros via Grid Search no SARIMAX, o XGBoost entregou um desempenho preditivo mais robusto e preciso para a variável `spend`. Isso indica que, para este cenário específico de previsão de gastos em anúncios, o **XGBoost é a escolha mais recomendada** devido à sua maior acurácia e eficácia comprovada.

---

## 6. Contato

*   [LinkedIn](https://www.linkedin.com/in/thiago-mustasheep/)

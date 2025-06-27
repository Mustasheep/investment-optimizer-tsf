import argparse
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow.statsmodels
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os

# Configuração básica do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argumentos ---
parser = argparse.ArgumentParser()
parser.add_argument("--processed_data", type=str, help="Pasta com dados processados")
args = parser.parse_args()

# --- CARREGAR E PREPARAR OS DADOS ---
logging.info("Iniciando o treinamento do modelo SARIMAX...")
input_file_path = os.path.join(args.processed_data, 'dados_processados.csv')
df = pd.read_csv(input_file_path, parse_dates=True, index_col='date')

# Garante uma frequência diária para os dados e interpola caso faltar dias
df = df.asfrec('D')
df.interpolate(inplace=True)

# --- PREPARAR VARIÁVEIS ---
endog = df['spend'] # endog = y

exog_features = ['clicks', 'impressions', 'eh_feriado', 'dia_semana_sin', 'dia_semana_cos']
exog = df[exog_features] # exog = X

# --- DIVIDIR EM TREINO E TESTE ---
train_size = len(df) - 14
endog_train, endog_test = endog.iloc[:train_size], endog.iloc[train_size:]
exog_train, exog_test = exog.iloc[:train_size], exog.iloc[train_size:]

# --- TREINAR O MODELO SARIMAX ---
logging.info("Treinando o modelo SARIMAX...")
sarimax_order = (1,1,1)
seasonal_order = (1,1,0,7)

model = sm.tsa.SARIMAX(
    endog=endog_train,
    exog=exog_train,
    order=sarimax_order,
    seasonal_order=seasonal_order
)
results = model.fit(disp=False)
logging.info("Treinamento concluído.")
logging.info(results.summary())

# --- AVALIAR E REGISTRAR ---
logging.info("Avaliando o modelo e registrando métricas...")
preds = results.predict(start=endog_test.index[0], end=endog_test.index[-1], exog=exog_test)

mae = mean_absolute_error(endog_test, preds)
rmse = np.sqrt(mean_squared_error(endog_test, preds))

mlflow.log_metric("mae_sarimax", mae)
mlflow.log_metric("rmse_sarimax", rmse)
logging.info(f"SARIMAX MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# --- GERAR E REGISTRAR GRÁFICO ---
logging.info("Gerando e registrando o gráfico de previsões do Sarimax...")
fig = plt.figure(figsize=(15, 7))
plt.plot(endog_test.index, endog_test, label='Valores Reais', marker='o')
plt.plot(preds.index, preds, label='Valores Previstos', linestyle='--')
plt.title('Previsão de Gastos (Spend) - SARIMAX - Últimos 14 dias')
plt.legend()
plt.grid(True)
mlflow.log_figure(fig, "sarimax_forecast_plot.png")
logging.info("Gráfico SARIMAX registrado como artefato.")

# Registrar modelo
logging.info("Registrando o modelo SARIMAX...")
mlflow.statsmodels.log_model(
    sm_model=results,
    artifact_path="modelo_sarimax_spend",
    registered_model_name="modelo-sarimax-spend-dp100"
)
logging.info("Modelo SARIMAX registrado com sucesso.")
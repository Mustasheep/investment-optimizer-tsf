import argparse
import pandas as pd
import os
import logging
import mlflow
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argumento de entrada ---
parser = argparse.ArgumentParser()
parser.add_argument("--processed_data", type=str, help="Pasta de entrada com os dados processados")
args = parser.parse_args()

# --- Lógica de Treinamento ---
logging.info("Iniciando o treinamento do modelo de forecast...")
input_file_path = os.path.join(args.processed_data, 'dados_processados.csv')
df = pd.read_csv(input_file_path, parse_dates=True, index_col='date')
logging.info(f"Lidas {len(df)} linhas de dados processados.")

df.drop(columns=['date_start', 'date_stop'], inplace=True, errors='ignore')

features = [
    'impressions', 'clicks', 'reach',
    'dia_semana_cos','dia_semana_sin', 'dia_do_mes', 
    'mes_cos', 'mes_sin', 'semana_do_ano',
    'spend_lag_1', 'spend_lag_7',
    'spend_media_movel_7d',
    'eh_feriado', 
]
target = 'spend'

X = df[features]
y = df[target]

train_size = int(len(df) - 14)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
logging.info(f"Dados divididos em {len(X_train)} para treino e {len(X_test)} para teste.")

logging.info("Treinando o modelo XGBRegressor...")
params = {'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05, 'early_stopping_rounds': 50}
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
logging.info("Treinamento concluído.")

logging.info("Avaliando o modelo e registrando métricas...")
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("rmse", rmse)
logging.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

logging.info("Gerando e registrando o gráfico de previsões...")
fig = plt.figure(figsize=(15, 7))
plt.plot(y_test.index, y_test, label='Valores Reais')
plt.plot(y_test.index, preds, label='Valores Previstos', linestyle='--')
plt.title('Previsão de Gastos (Spend) - XGBoost - Últimos 14 dias')
plt.legend()
plt.grid(True)
mlflow.log_figure(fig, "previsao_vs_real.png")

# --- Registro do Modelo ---
logging.info("Registrando o modelo no workspace...")
mlflow.xgboost.log_model(
    xgb_model=model,
    artifact_path="modelo_xgboost_forecast",
    registered_model_name="modelo-forecast-spend-dp100"
)
logging.info("Modelo registrado com sucesso.")
logging.info("Pipeline de treinamento finalizado com sucesso.")
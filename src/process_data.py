import argparse
import pandas as pd
import os
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argumentos do Componente ---
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, help="Caminho de entrada para os dados brutos")
parser.add_argument("--output_data", type=str, help="Caminho para os dados de saída processados")
args = parser.parse_args()

# --- Carregar os Dados ---
logging.info(f"Iniciando o processamento de dados...")
input_file_path = os.path.join(args.input_data, "dados_brutos.csv")
df = pd.read_csv(input_file_path)
logging.info(f"Dados carregados com sucesso. Total de registros: {len(df)} linhas.")

# --- Engenharia de Features ---
# Etapa 1: Converter coluna de data e definir como índice
df['date'] = pd.to_datetime(df['date_start'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)
logging.info("Coluna de data processada e definida como índice.")

# Etapa 2: Criar features baseadas na data
df['dia_da_semana'] = df.index.dayofweek
df['dia_do_mes'] = df.index.day
df['mes'] = df.index.month
df['semana_do_ano'] = df.index.isocalendar().week
logging.info("Features de data (dia da semana, mês, etc.) criadas.")

# Etapa 3: Criar features de Lag
df['spend_lag_1'] = df['spend'].shift(1)    # Gasto ontem
df['spend_lag_7'] = df['spend'].shift(7)    # Gasto semana passada
logging.info("Features de lag criadas.")

# Etapa 4: Criar features de Janela Móvel
df['spend_media_movel_7d'] = df['spend'].rolling(window=7).mean().shift(1)
logging.info("Features de média móvel criadas.")

# Lidando com valores NaN caso forem gerados
df.dropna(inplace=True)
logging.info(f"Linhas com NaN removidas. Total de {len(df)} linhas prontas para treinamento.")

# --- Salvar os Dados Processados ---
os.makedirs(args.output_data, exist_ok=True)
output_path = os.path.join(args.output_data, 'dados_processados.csv')
df.to_csv(output_path)
logging.info(f"Dados processados salvos em: {output_path}")

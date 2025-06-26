import argparse
import pandas as pd
import requests
import logging
import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ARGUMENTOS DO COMPONENTE ---
parser = argparse.ArgumentParser()
parser.add_argument("--key_vault_name", type=str, help="Nome do Azure Key Vault")
parser.add_argument("--secret_name", type=str, help="Nome do segredo da chave da API")
parser.add_argument("--output_data", type=str, help="Pasta de saída para os dados")
args = parser.parse_args()

# --- RECUPERAR A CHAVE DA API DO KEY VAULT ---
try:
    key_vault_uri = f"https://{args.key_vault_name}.vault.azure.net/"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=key_vault_uri, credential=credential)
    api_key = client.get_secret(args.secret_name).value
    logging.info("Chave da API recuperada com sucesso do Key Vault.")
except Exception as e:
    logging.critical(f"Falha ao recuperar a chave do Key Vault: {e}")
    raise

# --- FAZER CHAMADA PARA A API DA META ---
logging.info("Fazendo a chamada para a API da Meta...")

ad_account_id = "act_10204275819078901" 
logging.info(f"Buscando insights diários para a conta {ad_account_id}...")

# Estrutura de URL recomendada para o nosso projeto
try:
    url = (
        f"https://graph.facebook.com/v20.0/{ad_account_id}/insights?"
        f"date_preset=last_90d&"   # Período: últimos 90 dias
        f"time_increment=1&"       # Agrupar dados por dia
        f"level=account&"          # Nível de agregação: na conta toda
        f"fields=spend,impressions,clicks,reach&"
        f"access_token={api_key}"
    )
    response = requests.get(url)
    response.raise_for_status()
    api_data = response.json()
    logging.info("Dados recebidos da API com sucesso.")
except requests.exceptions.RequestException as e:
    logging.error(f"Erro na chamada da API da Meta: {e}")
    raise

# --- PROCESSAR OS DADOS E SALVAR ---
if 'data' in api_data and api_data['data']:
    df = pd.DataFrame(api_data['data'])

    os.makedirs(args.output_data, exist_ok=True)
    output_path = os.path.join(args.output_data, 'dados_brutos.csv')

    df.to_csv(output_path, index=False)
    logging.info(f"Dados processados e salvos em: {output_path}. Total de {len(df)} linhas.")
else:
    logging.warning("Nenhum dado encontrado na resposta da API. O arquivo de saída estará vazio.")
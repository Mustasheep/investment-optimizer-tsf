import argparse
import pandas as pd
import requests
import os
import logging
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argumentos do Componente ---
parser = argparse.ArgumentParser()
parser.add_argument("--key_vault_name", type=str, help="Nome do Azure Key Vault")
parser.add_argument("--secret_name", type=str, help="Nome do segredo da chave da API")
parser.add_argument("--output_data", type=str, help="Pasta de saída para os dados")
args = parser.parse_args()

# --- Recuperar a Chave da API do Key Vault ---
try:
    key_vault_uri = f"https://{args.key_vault_name}.vault.azure.net/"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=key_vault_uri, credential=credential)
    api_key = client.get_secret(args.secret_name).value
    logging.info("Chave da API recuperada com sucesso do Key Vault.")
except Exception as e:
    logging.critical(f"Falha ao recuperar a chave do Key Vault: {e}")
    raise

# --- Fazer a Chamada para a API da Meta ---
ad_account_id = "act_10204275819078901"
all_data = []

url = (
    f"https://graph.facebook.com/v20.0/{ad_account_id}/insights?"
    f"date_preset=last_90d&"
    f"time_increment=1&"
    f"level=account&"
    f"fields=spend,impressions,clicks,reach&"
    f"limit=100&"
    f"access_token={api_key}"
)

# Loop para percorrer todas as páginas de resultados
while url:
    try:
        logging.info(f"Buscando dados da URL: {url[:150]}...")
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        
        page_data = json_data.get('data', [])
        if page_data:
            all_data.extend(page_data)
            logging.info(f"Recebidos {len(page_data)} registros. Total até agora: {len(all_data)}.")
        
        url = json_data.get('paging', {}).get('next')

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro na chamada da API da Meta: {e}")
        url = None

# --- Processar os Dados e Salvar ---
if all_data:
    df = pd.DataFrame(all_data)
    
    os.makedirs(args.output_data, exist_ok=True)
    output_path = os.path.join(args.output_data, 'dados_brutos.csv')
    df.to_csv(output_path, index=False)
    
    logging.info(f"Processo concluído. Total de {len(df)} linhas salvas em: {output_path}")
else:
    logging.warning("Nenhum dado foi extraído da API.")
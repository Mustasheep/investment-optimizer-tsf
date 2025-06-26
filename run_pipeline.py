from azure.ai.ml import MLClient, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# Conectar ao workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# --- CARREGAR COMPONENTE DO ARQUIVO YML ---
get_data_component = load_component(source="./components/get_data_component.yml")

# --- DEFINIÇÃO DO PIPELINE ---
@pipeline(
    compute="cluster-cpu-dp100",
    description="Pipeline para buscar e processar dados de marketing.",
)

def marketing_data_pipeline(
    key_vault_name: str,
    secret_name: str,
):
    get_data_step = get_data_component(
        key_vault_name=key_vault_name,
        secret_name=secret_name
    )

    return {
        "pipeline_output_data": get_data_step.outputs.output_data
    }

# --- EXECUÇÃO DO PIPELINE ---
pipeline_job = marketing_data_pipeline(
    key_vault_name="dp1000524355966",
    secret_name="meta-api-token",
)

print("Submetendo o pipeline ao Azure Machine Learning...")
returned_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="pipeline_de_marketing")
print(f"Pipeline submetido. Acompanhe em: {returned_job.studio_url}")
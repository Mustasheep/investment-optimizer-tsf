from azure.ai.ml import MLClient, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

get_data_component = load_component(source="./components/get_data_component.yml")
process_data_component = load_component(source="./components/process_data_component.yml")
train_component = load_component(source="./components/train_forecast_component.yml")

@pipeline(
    compute="cluster-cpu-dp100",
    description="Pipeline E2E final: busca, processa e treina um modelo de forecast.",
)
def full_marketing_pipeline(
    key_vault_name: str,
    secret_name: str,
):
    get_data_step = get_data_component(
        key_vault_name=key_vault_name,
        secret_name=secret_name,
    )
    
    process_data_step = process_data_component(
        input_data=get_data_step.outputs.output_data
    )

    train_model_step = train_component(
        processed_data=process_data_step.outputs.output_data
    )

pipeline_job = full_marketing_pipeline(
    key_vault_name="dp1000524355966",
    secret_name="meta-api-token",
)

print("Submetendo o pipeline final e sincronizado...")
returned_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="pipeline_de_marketing")
print(f"Pipeline submetido. Acompanhe em: {returned_job.studio_url}")
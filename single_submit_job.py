from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv
import os

load_dotenv()

# Create MLClient using Azure CLI authentication
credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
)
print("Connected to workspace:", ml_client.workspace_name)

# Set up the compute cluster (create if it doesn't exist)
cluster_name = "cpu-cluster"
vm_size = "Standard_D2s_v3"

try:
    # Check if compute exists
    compute = ml_client.compute.get(cluster_name)
    print("Found existing compute target:", cluster_name)
except Exception:
    print("Creating new compute target:", cluster_name)
    # Define compute
    compute = AmlCompute(
        name=cluster_name,
        size=vm_size,
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=1200
    )
    ml_client.compute.begin_create_or_update(compute).result()

# Define custom environment
custom_env = Environment(
    name="training-env",
    version="6",
    description="Custom environment for PyTorch training",
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

# Register the environment
env = ml_client.environments.create_or_update(custom_env)

# Create job specification
job = command(
    experiment_name="single-node-pytorch-benchmark",
    display_name="pytorch-training-job",
    description="PyTorch training job",
    compute=cluster_name,
    environment=f"{env.name}:{env.version}",
    code=".",
    command="python single_training.py --data-dir ./data --output-dir ${{outputs.output_dir}} --epochs 10 --batch-size 64 --learning-rate 0.001",
    outputs={
        "output_dir": Output(type=AssetTypes.URI_FOLDER)
    }
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print("Run submitted:", returned_job.name)
print(f"Run URL: {returned_job.studio_url}")
# Wait for the job to complete
ml_client.jobs.stream(returned_job.name)
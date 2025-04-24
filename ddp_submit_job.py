import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Output, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv

load_dotenv()

credential = DefaultAzureCredential()

ml_client = MLClient(
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    credential=credential
)
print("Connected to workspace:", ml_client.workspace_name)

# Set up the compute cluster (if it doesn't exist)
cluster_name = "cpu-cluster"
vm_size = "Standard_D2s_v3"

try:
    compute = ml_client.compute.get(cluster_name)
    print("Found existing compute target:", cluster_name)
except Exception:
    print("Creating new compute target:", cluster_name)
    compute = AmlCompute(
        name=cluster_name,
        vm_size=vm_size,
        min_instances=0,
        max_instances=2,
        idle_seconds_before_scaledown=1200
    )
    ml_client.compute.begin_create_or_update(compute).result()

# Define training environment
custom_env = Environment(
    name="training-env",
    version="7",
    description="Custom environment for PyTorch training",
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

env = ml_client.environments.create_or_update(custom_env)

job = command(
    name="distributed-pytorch-training",
    description="Distributed training with PyTorch and MPI",
    display_name="pytorch-distributed-job",
    experiment_name="distributed-pytorch-training",
    code=".",
    command="python distributed_training.py --data-dir ./data --output-dir ${{outputs.output_dir}} --epochs 10 --batch-size 64 --learning-rate 0.001",
    environment=env,
    outputs={
        "output_dir": Output(type=AssetTypes.URI_FOLDER)
    },
    compute=cluster_name,
    instance_count=2,
    distribution={
        "type": "mpi"
    }
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print("Run submitted:", returned_job.name)
print(f"Run URL: {returned_job.studio_url}")
# Wait for the job to complete
ml_client.jobs.stream(returned_job.name)
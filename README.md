# fashion-mnist-distributed-training

## Overview
This project is a distributed deep learning system to classify fashion items using the Fashion-MNIST dataset and training a neural network in parallel.

## Team Members
- Ryan Pham
- Eric Hoang

## Running the Training
### Prerequisites:
- Azure ML workspace
- Minimum free tier Azure subscription

### Step 1: Environment Setup
Clone the repository and open it:
```bash
$ git clone https://github.com/ryangpham/fashion-mnist-distributed-training
$ cd fashion-mnist-distributed-training
```
Create a python virtual environment and install dependencies:
```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```
Login to Azure through the CLI:
```bash
az login
```
After logging in, create a `.env` file in the root directory of the project, and fill out your credential info following this format:
```env
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group-name
AZURE_WORKSPACE_NAME=your-workspace-name
```

### Step 2: Training on the cloud
Run single-node training:
```bash
python single_submit_job.py
```
Run distributed training:
```bash
python ddp_submit_job.py
```
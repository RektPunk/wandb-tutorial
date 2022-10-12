from module import WandbExperimentManager

# init
wandb_run_manager = WandbExperimentManager(
    project="my-test-project",
    name="test-code",
    notes="test flight",
    tags=["test-tag1", "test-tag2"],
    config={"learning_rate": 0.001, "epochs": 1000, "batch_size": 128},
)

# log config
wandb_run_manager.log_config(
    {
        "train/size": 1000,
        "test/size": 100,
    }
)

# log metrics
for epoch in range(1000):
    wandb_run_manager.log(
        {
            "train/loss": 1 / (epoch + 1),
            "train/accuracy": 1 - 1 / (epoch + 1),
        }
    )

# log summary
wandb_run_manager.log_summary(
    {
        "train/accuracy": 0.99,
        "test/accuracy": 0.97,
    }
)

# get info of run
print(wandb_run_manager.get_infos())

# alert
wandb_run_manager.alert(
    title="alert test title",
    text="alert test text",
    level="ERROR",
)

# log artifact
wandb_run_manager.log_artifact(
    name="train",
    path="artifacts/dataset.txt",
    type="dataset",
)

wandb_run_manager.log_artifact(
    name="baseline",
    path="artifacts/model_v0.txt",
    type="model",
)

wandb_run_manager.log_artifact(
    name="baseline",
    path="artifacts/model_v1.txt",
    type="model",
)


# download artifact
v0_artifact_path = wandb_run_manager.download_artifact(
    name="baseline",
    version="v0",
)
print(v0_artifact_path)

latest_artifact_path = wandb_run_manager.download_artifact(
    name="baseline",
)
print(latest_artifact_path)


wandb_run_manager.finish()

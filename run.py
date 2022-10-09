from module import WandbExperimentManager


wandb_run_manager = WandbExperimentManager(
    project="my-test-project",
    name="test-code",
    notes="test flight",
    tags=["test-tag1", "test-tag2"],
    config={"learning_rate": 0.001, "epochs": 1000, "batch_size": 128},
)

wandb_run_manager.log_config(
    {
        "train/size": 1000,
        "test/size": 100,
    }
)

for epoch in range(1000):
    wandb_run_manager.log(
        {
            "train/loss": 1 / (epoch + 1),
            "train/accuracy": 1 - 1 / (epoch + 1),
        }
    )

wandb_run_manager.log_summary(
    {
        "train/accuracy": 0.99,
        "test/accuracy": 0.97,
    }
)

print(wandb_run_manager.run_infos)

wandb_run_manager.alert(
    title="alert test title",
    text="alert test text",
    level="ERROR",
)

wandb_run_manager.finish()

import random
from typing import Tuple
from module import WandbExperimentManager, WandbSweepManager


SWEEP_CONFIGURATION = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32]},
        "epochs": {"values": [5, 10]},
        "learning_rate": {"max": 0.1, "min": 0.0001},
    },
}


def train_eval(epoch: int, lr: float, bs: int) -> Tuple[float, float]:
    """
    #FIXME; example train loss
    Args:
        epoch (int)
        lr (float)
        bs (int)
    Returns:
        Tuple[float, float]
    """
    acc = 0.25 + ((epoch / 30) + (random.random() / lr))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / bs))
    return acc, loss


def validation_eval(epoch: int) -> Tuple[float, float]:
    """
    #FIXME; example validation loss
    Args:
        epoch (int)
    Returns:
        Tuple[float, float]
    """
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def train(project: str, **kwargs):
    """
    #FIXME: example train
    Args:
        project (str): project name
        **kwargs: input of WandbExperimentManager or wandb.init
    """
    wandb_experiment_manager = WandbExperimentManager(project=project, **kwargs)
    _config = wandb_experiment_manager.get_infos()["config"]
    learning_rate = _config["lr"]
    batch_size = _config["batch_size"]
    epochs = _config["epochs"]

    for epoch in range(epochs):
        train_acc, train_loss = train_eval(epoch, learning_rate, batch_size)
        val_acc, val_loss = validation_eval(epoch)
        wandb_experiment_manager.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )


WandbSweepManager(
    project="my-test-sweep-project",
    sweep_configuration=SWEEP_CONFIGURATION,
    func=train,
    count=3,
)

WandbSweepManager.sweep()

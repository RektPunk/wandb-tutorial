import os
from dotenv import load_dotenv


class WandbToken:
    """
    Set wandb token
    """

    def __init__(self):
        load_dotenv()
        self.wandb_api_token = os.environ["WANDB_API_TOKEN"]

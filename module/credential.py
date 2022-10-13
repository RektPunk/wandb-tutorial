import os
from enum import Enum
from dotenv import load_dotenv
import wandb


class Credential(str, Enum):
    load_dotenv()
    WANDB_API_TOKEN: str = os.environ["WANDB_API_TOKEN"]

    def __repr__(self) -> str:
        return f"{self.value}"


def login_check(api_token: str):
    _is_login = wandb.login(key=api_token)
    assert _is_login == True

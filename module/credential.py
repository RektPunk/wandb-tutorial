import os
from enum import Enum
from dotenv import load_dotenv


class Credential(str, Enum):
    load_dotenv()
    WANDB_API_TOKEN: str = os.environ["WANDB_API_TOKEN"]

    def __repr__(self) -> str:
        return f"{self.value}"

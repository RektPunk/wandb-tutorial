from typing import Any, Callable, Dict
import inspect
from functools import partial
import wandb
from module.credential import Credential, login_check


class WandbSweepManager:
    """
    Wandb sweep manager\\
    Args:
        project (str): project name
        sweep_config ([Dict[str, Any]]): sweep configuration
        func (Callable): main train function, 'project' must be in the input
        count (int): sweep count
    Methods:
        sweep()
    """

    def __init__(
        self,
        project: str,
        sweep_config: Dict[str, Any],
        func: Callable,
        count: int,
        **kwargs
    ):
        """
        Args:
            project (str): project name
            sweep_config ([Dict[str, Any]]): sweep configuration
            func (Callable): main train function, 'project' must be in the input
            count (int): sweep count
        Raises:
            Exception: project input not in function
        """
        self.wandb_api_token: str = Credential.WANDB_API_TOKEN
        login_check(api_token=self.wandb_api_token)
        self.sweep_config = sweep_config
        self.project = project
        if "project" not in inspect.getfullargspec(func).args:
            raise Exception("Project not in target function")
        self.main_func = partial(func, project=self.project, **kwargs)
        self.count = count

    def sweep(self) -> str:
        """
        Sweep run
        Returns:
            str: sweep id
        """
        sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.project)
        wandb.agent(sweep_id, self.main_func, count=self.count)
        return sweep_id

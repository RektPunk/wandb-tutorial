import wandb
from typing import Any, Optional, Union, List, Dict
from module.credential import WandbToken
from wandb import AlertLevel


class WandbExperimentManager(WandbToken):
    """
    Wandb experiment manager\\

    Args:
        project (str)
        name (Optional[str], optional)  Defaults to None.
        notes (Optional[str], optional)  Defaults to None.
        tags (Optional[Union[str, List[str]]], optional)  Defaults to None.
        config (Optional[Dict[str, Any]], optional)  Defaults to None.

    Methods:
        log_config()
        log_summary()
        log()
        alert()
        finish()

    Properties:
        run_infos
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[Union[str, List[str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            project (str)
            name (Optional[str], optional)  Defaults to None.
            notes (Optional[str], optional)  Defaults to None.
            tags (Optional[Union[str, List[str]]], optional)  Defaults to None.
            config (Optional[Dict[str, Any]], optional)  Defaults to None.
        """
        super().__init__()
        wandb.login(key=self.wandb_api_token)
        self.wandb_run = wandb.init(
            project=project,
            name=name,
            tags=tags,
            notes=notes,
            config=config,
        )
        self._run_url = self.wandb_run.get_project_url()
        self._config = self.wandb_run.config
        self._summary = {}

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log config
        Args:
            config (Dict[str, Any])
        """
        self.wandb_run.config.update(config)
        self._config = self.wandb_run.config

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log summary
        Args:
            summary (Dict[str, Any])
        """
        self.wandb_run.summary.update(summary)
        self._summary = self.wandb_run.summary._as_dict()

    def log(self, metric: Dict[str, Any]) -> None:
        """
        Log metrics
        Args:
            metric (Dict[str, Any])
        """
        self.wandb_run.log(metric)

    def alert(
        self, title: str, text: str, level: str = AlertLevel.WARN.value, **kwargs
    ) -> None:
        """
        alert
        Args:
            title (str)
            text (str)
            level (str)

        """
        _level = level.upper()
        if _level in AlertLevel._member_names_:
            self.wandb_run.alert(
                title=title,
                text=text,
                level=AlertLevel[_level].value,
                **kwargs,
            )
        else:
            raise Exception(f"Level should be one of {AlertLevel._member_names_}")

    def finish(self) -> None:
        """
        Finish logging
        """
        self.wandb_run.finish()

    @property
    def run_infos(self):
        """
        Run infos
        Returns:
            {
                "run_url": ...,
                "config" : ...,
                "summary": ...,
            }
        """
        return {
            "run_url": self._run_url,
            "config": self._config,
            "summary": self._summary,
        }

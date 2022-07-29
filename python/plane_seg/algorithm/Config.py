from pathlib import Path
from typing import Sequence, Union

import configparser

__all__ = ["Config"]


class ConfigError(BaseException):
    def __init__(self, msg: str):
        self.msg = msg


class Config:
    def __init__(self, cfg_path: Path, parameter_list: Sequence[str]):
        self.cfg_path = cfg_path
        self.parameter_list = parameter_list

        self.config = Config.__init_config(cfg_path, parameter_list)

    @staticmethod
    def __init_config(
        cfg_path: Path, parameter_list: Sequence[str]
    ) -> configparser.ConfigParser():
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg.read(cfg_path)
        if cfg.sections() != ["Parameters"]:
            raise ConfigError(f".ini should contain only 'Parameters' section")

        if sorted(cfg["Parameters"]) != sorted(parameter_list):
            raise ConfigError(
                f"Expected parameters: {sorted(parameter_list)}, got {sorted(cfg['Parameters'])}"
            )

        return cfg

    def change_value(self, param_name: str, value: Union[int, float]):
        if param_name not in self.parameter_list:
            raise ConfigError(f"Got unexpected parameter '{param_name}'")
        self.config["Parameters"][param_name] = str(value)

    def write(self, cfg_path: Path) -> Path:
        with open(cfg_path, "w") as output:
            self.config.write(output, space_around_delimiters=False)

        return cfg_path

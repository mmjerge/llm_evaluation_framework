#!/usr/bin/env python

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Any, Dict, Callable
from dataclasses import dataclasses
import importlib.util
from datetime import datetime
import yaml
from omegaconf import OmegaConf, DictConfig

#logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """_summary_
    """
    benchmark_name: str
    benchmark_path: str
    model_names: List[str]
    methods_to_test: List[str]
    output_dir: str
    batch_size: int = 10
    instruction: str = ""

def get_config_dir() -> Path:
    """_summary_

    Returns:
        Path: _description_
    """
    return Path)__file__).parent / "config"

def load_method_config() -> DictConfig:
    method_path = Path(__file__).parent / "evaluated_methods"
    method_name = Path(method_path).name
    method_config = get_config_dir / "method_config.yaml"

class MethodLoader:
    """_summary_
    """
    @staticmethod
    def load_method(project_path: str) -> Callable:
        """
        Load a method from an external project directory

        Args:
            project_path (str): Path to the project containing the method

        Returns:
            Callable: The loaded method function
        """
        project_path = os.path.abspath(project_path)
        config_path = os.path.join(project_path, "method_config.yaml")
        


from dataclasses import dataclass
from pathlib import Path
import yaml
import rootutils

root_path = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)  

CONFIG_PATH = root_path / "config" / "parameters.yaml"


@dataclass(frozen=True)
class DataProcessingConfig:
    """Setup of data processing parameters to lfw"""
    lfw_min_faces_per_person: int
    lfw_resize_factor: float
    target_image_size: list[int]
    sharpness_threshold: float
    detector_model_name: str
    detector_ctx_id: int
    detector_det_size: list[int]



@dataclass
class Config:
    """Principal configuration dataclass."""
    data_processing: DataProcessingConfig

   
def load_config() -> Config:
    """
    Load configuration from a YAML file and return a Config dataclass instance.
    """
 
    with open(CONFIG_PATH, "r") as f:
        yaml_data = yaml.safe_load(f)

    config = Config(
        data_processing=DataProcessingConfig(**yaml_data["data_processing"]),
    )
    
    return config
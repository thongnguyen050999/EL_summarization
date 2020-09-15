from dataclasses import dataclass
from pathlib import Path

@dataclass
class Project:

    base_dir: Path = Path(__file__).parents[0]
    data_dir = base_dir/'dataset'
    checkpoint_dir = base_dir/'checkpoint'

    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=1)
        self.checkpoint_dir.mkdir(exist_ok=1)
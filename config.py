# src/config.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

# Optional YAML support (falls back to JSON if unavailable)
try:
    from ruamel.yaml import YAML  # type: ignore
    _HAVE_YAML = True
    _yaml = YAML(typ="safe")
except Exception:
    _HAVE_YAML = False
    _yaml = None  # type: ignore


# ----------------------------
# Dataclass configs
# ----------------------------

@dataclass
class DataConfig:
    data: str = ""                # path to dataset JSON
    batch_size: int = 256
    val_split: float = 0.1        # 10% validation split


@dataclass
class ModelConfig:
    node_hidden: int = 64
    edge_hidden: int = 64
    score_hidden: int = 64
    use_edges: bool = True
    gnn_layers: int = 0           # 0 = DeepSets+edges; >0 enables message passing
    dropout: float = 0.0
    device: str = "auto"          # "auto" -> cuda if available else cpu


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 1337
    out_dir: str = "outputs"      # base output directory


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------ I/O ------------

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not _HAVE_YAML:
            raise RuntimeError("YAML support not available. Install ruamel.yaml or use JSON.")
        with path.open("r") as f:
            return _yaml.load(f) or {}

    @staticmethod
    def _dump_yaml(obj: Dict[str, Any], path: Path) -> None:
        if not _HAVE_YAML:
            raise RuntimeError("YAML support not available. Install ruamel.yaml or use JSON.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            _yaml.dump(obj, f)

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """
        Load configuration from a YAML or JSON file.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in {".yml", ".yaml"}:
            raw = cls._load_yaml(p)
        else:
            raw = json.loads(p.read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Config:
        """
        Build from a nested dict like:
          {"data": {...}, "model": {...}, "train": {...}}
        Missing sections/keys will use defaults.
        """
        def merge(dc_cls, defaults, payload):
            if payload is None:
                return defaults
            out = {**asdict(defaults)}
            out.update({k: v for k, v in payload.items() if k in out})
            return dc_cls(**out)

        base = cls()  # defaults
        data = merge(DataConfig, base.data, d.get("data"))
        model = merge(ModelConfig, base.model, d.get("model"))
        train = merge(TrainConfig, base.train, d.get("train"))
        return cls(data=data, model=model, train=train)

    def save(self, path: str | Path) -> None:
        """
        Save configuration to YAML (if extension .yaml/.yml and ruamel available) or JSON.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        if p.suffix.lower() in {".yml", ".yaml"} and _HAVE_YAML:
            self._dump_yaml(d, p)
        else:
            p.write_text(json.dumps(d, indent=2))

    # ------------ Merge helpers ------------

    def with_overrides(
        self,
        data: Optional[Dict[str, Any]] = None,
        model: Optional[Dict[str, Any]] = None,
        train: Optional[Dict[str, Any]] = None,
    ) -> Config:
        """
        Return a new Config with shallow overrides applied to each section.
        """
        new_data = replace(self.data, **{k: v for k, v in (data or {}).items() if hasattr(self.data, k)})
        new_model = replace(self.model, **{k: v for k, v in (model or {}).items() if hasattr(self.model, k)})
        new_train = replace(self.train, **{k: v for k, v in (train or {}).items() if hasattr(self.train, k)})
        return Config(data=new_data, model=new_model, train=new_train)


# ----------------------------
# Argparse bridge (optional)
# ----------------------------

def add_args_to_parser(parser) -> None:
    """
    Add a minimal, flat set of CLI args that mirror the dataclasses.
    This keeps train.py simple while still allowing config file + overrides.
    """
    # General / files
    parser.add_argument("--config", type=str, default=None, help="Optional path to YAML/JSON config file.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override: outputs directory (train.out_dir)")

    # Data
    parser.add_argument("--data", type=str, default=None, help="Path to dataset JSON (data.data)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (data.batch_size)")
    parser.add_argument("--val_split", type=float, default=None, help="Validation split fraction (data.val_split)")

    # Model
    parser.add_argument("--node_hidden", type=int, default=None)
    parser.add_argument("--edge_hidden", type=int, default=None)
    parser.add_argument("--score_hidden", type=int, default=None)
    parser.add_argument("--use_edges", action="store_true", help="If passed, sets model.use_edges=True")
    parser.add_argument("--no_use_edges", action="store_true", help="If passed, sets model.use_edges=False")
    parser.add_argument("--gnn_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, help='"auto", "cpu", "cuda", or "cuda:N"')

    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)


def build_config_from_args(args) -> Config:
    """
    Construct Config by (1) loading file if provided, then (2) applying CLI overrides.
    """
    # 1) Load base config (file or defaults)
    if getattr(args, "config", None):
        cfg = Config.from_file(args.config)
    else:
        cfg = Config()

    # 2) Sectioned overrides
    data_over = {}
    if args.data is not None:        data_over["data"] = args.data
    if args.batch_size is not None:  data_over["batch_size"] = args.batch_size
    if args.val_split is not None:   data_over["val_split"] = args.val_split

    model_over = {}
    if args.node_hidden is not None:   model_over["node_hidden"] = args.node_hidden
    if args.edge_hidden is not None:   model_over["edge_hidden"] = args.edge_hidden
    if args.score_hidden is not None:  model_over["score_hidden"] = args.score_hidden
    if args.gnn_layers is not None:    model_over["gnn_layers"] = args.gnn_layers
    if args.dropout is not None:       model_over["dropout"] = args.dropout
    if args.device is not None:        model_over["device"] = args.device
    if getattr(args, "use_edges", False):     model_over["use_edges"] = True
    if getattr(args, "no_use_edges", False):  model_over["use_edges"] = False

    train_over = {}
    if args.epochs is not None:        train_over["epochs"] = args.epochs
    if args.lr is not None:            train_over["lr"] = args.lr
    if args.weight_decay is not None:  train_over["weight_decay"] = args.weight_decay
    if args.seed is not None:          train_over["seed"] = args.seed
    if args.out_dir is not None:       train_over["out_dir"] = args.out_dir

    return cfg.with_overrides(data=data_over, model=model_over, train=train_over)

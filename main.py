# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT

"""Unified CLI: generation intentionally does not construct a DataModule."""

import argparse
import collections
import os
import pathlib
import typing
import warnings
from pathlib import Path

import omegaconf
import pytorch_lightning as pl
import torch
from einops._torch_specific import allow_ops_in_compiled_graph
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info

from utils import instantiate_from_config

warnings.filterwarnings("ignore")
allow_ops_in_compiled_graph()
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata,
    typing.Any, dict, collections.defaultdict, omegaconf.nodes.AnyNode,
    omegaconf.base.Metadata, omegaconf.listconfig.ListConfig,
    pathlib.PosixPath,
])


class SetupCallback(Callback):
    def __init__(self, basedir):
        super().__init__()
        self.logdir = Path(basedir) / "log"
        self.ckptdir = Path(basedir) / "ckpt"

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        if trainer.global_rank == 0:
            self.logdir.mkdir(parents=True, exist_ok=True)
            self.ckptdir.mkdir(parents=True, exist_ok=True)


def training_resources(config):
    """Create callbacks and logging only for training."""
    setup = SetupCallback(config.training.output_dir)
    checkpoint = ModelCheckpoint(
        dirpath=setup.ckptdir,
        filename="ckpt-{epoch}-{val_total_loss:.4f}",
        monitor=config.training.monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    logger = TensorBoardLogger(str(setup.logdir), name="tensorboard")
    return [setup, checkpoint], logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "test", "generate"))
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE", help="Override any YAML value; repeat as needed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_type", choices=("bf16", "16", "32"), default="16")
    parser.add_argument("--gradient_clip_val", type=float)
    parser.add_argument("--gradient_clip_algorithm")
    parser.add_argument("--every_n_train_epochs", type=int, default=5)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--val_every_n_epoch", type=int, default=1)
    parser.add_argument("--limit_val_batches", type=int, default=64)
    parser.add_argument("--monitor", default="val_total_loss")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--encoder_ckpt", default="")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sample", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_mode(args):
    legacy = [name for name in ("test", "sample") if getattr(args, name)]
    if len(legacy) > 1:
        raise ValueError("--test and --sample cannot be used together.")
    old_mode = {"test": "test", "sample": "generate"}.get(
        legacy[0] if legacy else ""
    )
    if args.mode and old_mode and args.mode != old_mode:
        raise ValueError("--mode conflicts with the legacy mode flag.")
    return args.mode or old_mode or "train"


def load_config(args, mode):
    config = OmegaConf.load(args.config)
    keys = (
        "seed", "num_nodes", "num_gpus", "update_every", "epochs",
        "base_lr", "use_amp", "amp_type", "gradient_clip_val",
        "gradient_clip_algorithm", "every_n_train_epochs",
        "log_every_n_steps", "val_every_n_epoch", "limit_val_batches",
        "monitor", "output_dir", "ckpt_path", "encoder_ckpt", "fast",
        "deepspeed",
    )
    runtime = {key: getattr(args, key) for key in keys}
    config.training = OmegaConf.merge(
        config.get("training", {}), runtime, {"mode": mode}
    )
    # Apply dot-list values last so scripts can override runtime defaults too.
    if args.overrides:
        config = OmegaConf.merge(
            config, OmegaConf.from_dotlist(args.overrides)
        )
    return config


def resolve_checkpoint(value):
    """Resolve a file, or last.ckpt in a checkpoint directory."""
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    last = path / "last.ckpt"
    if last.is_file():
        return last
    checkpoints = sorted(path.glob("*.ckpt"))
    if len(checkpoints) == 1:
        return checkpoints[0]
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in: {path}")
    raise ValueError(
        f"Multiple checkpoints found in {path}; pass an exact file."
    )


def load_weights(model, path):
    """Load a checkpoint without a Trainer or DataModule."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    if hasattr(model, "on_load_checkpoint") and isinstance(checkpoint, dict):
        model.on_load_checkpoint(checkpoint)


def build_trainer(config, training):
    callbacks, logger = ([], False)
    if training:
        callbacks, logger = training_resources(config)
    multi_device = (
        config.training.num_nodes > 1 or config.training.num_gpus > 1
    )
    strategy = "auto"
    if multi_device:
        strategy = (
            DeepSpeedStrategy(stage=1) if config.training.deepspeed
            else DDPStrategy(find_unused_parameters=True, bucket_cap_mb=1500)
        )
    precision = (
        config.training.amp_type if config.training.use_amp else 32
    )
    return pl.Trainer(
        max_epochs=config.training.epochs,
        precision=precision,
        callbacks=callbacks,
        accelerator="gpu",
        devices=config.training.num_gpus,
        num_nodes=config.training.num_nodes,
        strategy=strategy,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm=config.training.gradient_clip_algorithm,
        accumulate_grad_batches=config.training.update_every,
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        check_val_every_n_epoch=config.training.val_every_n_epoch,
        limit_val_batches=config.training.limit_val_batches,
    )


def generate(model, config):
    checkpoint = resolve_checkpoint(config.training.ckpt_path)
    if checkpoint is None:
        raise ValueError("Generation requires --ckpt_path.")
    load_weights(model, checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires CUDA.")
    model.to("cuda:0").eval()
    options = OmegaConf.to_container(
        config.get("generation", {}), resolve=True
    )
    if hasattr(model, "generate"):
        model.generate(**options)
    elif hasattr(model, "sample"):
        batch_size = int(options.pop("batch_size", 1))
        if options:
            raise ValueError(f"Unsupported sample options: {sorted(options)}")
        model.sample(batch_size=batch_size, device="cuda:0")
    else:
        raise AttributeError(
            f"{type(model).__name__} has no generate() or sample()."
        )


def main():
    args = get_args()
    mode = resolve_mode(args)
    config = load_config(args, mode)
    if config.training.fast:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    if "NNODES" in os.environ:
        config.training.num_nodes = int(os.environ["NNODES"])
    pl.seed_everything(config.training.seed, workers=True)
    rank_zero_info("Resolved configuration:\n" + OmegaConf.to_yaml(config))

    model = instantiate_from_config(config.model)
    model.learning_rate = config.training.base_lr
    model.max_epochs = config.training.epochs
    if config.training.encoder_ckpt:
        model.load_encoder(config.training.encoder_ckpt)

    if mode == "generate":
        generate(model, config)
        return

    if "dataset" not in config:
        raise KeyError(f"Mode '{mode}' requires config.dataset.")
    data = instantiate_from_config(config.dataset)
    trainer = build_trainer(config, training=(mode == "train"))
    checkpoint = resolve_checkpoint(config.training.ckpt_path)
    checkpoint = str(checkpoint) if checkpoint else None
    if mode == "test":
        trainer.test(model, datamodule=data, ckpt_path=checkpoint)
    else:
        trainer.fit(model, datamodule=data, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()

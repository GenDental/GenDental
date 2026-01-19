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

from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.distributed as dist
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()
import pathlib
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info
from utils import instantiate_from_config
import omegaconf
import typing
import collections
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([typing.Any])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.safe_globals([pathlib.PosixPath])


class SetupCallback(Callback):
    def __init__(self, config: DictConfig, basedir: Path, logdir: str = "log", ckptdir: str = "ckpt") -> None:
        super().__init__()
        self.logdir = basedir / logdir
        self.ckptdir = basedir / ckptdir
        self.config = config

    def on_fit_start(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)


def setup_callbacks(config: DictConfig) -> Tuple[List[Callback], Logger]:
    basedir = Path(config.training.output_dir)
    os.makedirs(basedir, exist_ok=True)
    all_callbacks = []

    setup_callback = SetupCallback(config, basedir)
    all_callbacks.append(setup_callback)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="ckpt-{epoch}-{val_total_loss:.4f}",
        monitor=config.training.monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=False,)
        # every_n_epochs=config.training.every_n_train_epochs)
    all_callbacks.append(checkpoint_callback)

    logger = TensorBoardLogger(save_dir=str(setup_callback.logdir), name="tensorboard")

    return all_callbacks, logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--use_amp", default=False, action="store_true")
    parser.add_argument("--amp_type", type=str, default="16")
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--gradient_clip_algorithm", type=str, default=None)
    parser.add_argument("--every_n_train_epochs", type=int, default=5)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--val_every_n_epoch", type=int, default=1)
    parser.add_argument("--limit_val_batches", type=int, default=64)
    parser.add_argument("--monitor", type=str, default="val_total_loss")
    parser.add_argument("--output_dir", type=str, help="the output directory to save everything.")
    parser.add_argument("--ckpt_path", type=str, default="", help="the restore checkpoints.")
    parser.add_argument("--encoder_ckpt", type=str, default="", help="the restore checkpoints.")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--sample", default=False, action="store_true")
    parser.add_argument("--deepspeed", default=False, action="store_true")
    return parser.parse_args()
    

if __name__ == "__main__":
    
    args = get_args()
    
    if args.fast:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.05

    # Set random seed
    pl.seed_everything(args.seed, workers=True)

    # Load configuration
    config = OmegaConf.load(args.config)
    config.training = DictConfig(vars(args))

    # print config
    rank_zero_info("Begin to print configuration ...")
    rank_zero_info(OmegaConf.to_yaml(config))
    rank_zero_info("Finish print ...")

    # Setup callbacks
    callbacks, loggers = setup_callbacks(config)

    # Build model
    model: pl.LightningModule = instantiate_from_config(config.model)

    if config.training.encoder_ckpt != '':
        model.load_encoder(config.training.encoder_ckpt)

    # Build data modules
    data: pl.LightningDataModule = instantiate_from_config(config.dataset)
    
    nodes = args.num_nodes
    ngpus = args.num_gpus
    base_lr = config.training.base_lr
    accumulate_grad_batches = config.training.update_every
    batch_size = config.dataset.params.batch_size

    if 'NNODES' in os.environ:
        nodes = int(os.environ['NNODES'])
        config.training.num_nodes = nodes
        args.num_nodes = nodes

    model.learning_rate = base_lr
    model.max_epochs = config.training.epochs

    # Build trainer
    if args.num_nodes > 1 or args.num_gpus > 1:
        if args.deepspeed:
            ddp_strategy = DeepSpeedStrategy(stage=1)
        else:
            ddp_strategy = DDPStrategy(find_unused_parameters=True, bucket_cap_mb=1500)
            # ddp_strategy = DP
    else:
        ddp_strategy = None  # 'auto'

    rank_zero_info(f'*' * 100)
    if config.training.use_amp:
        amp_type = config.training.amp_type
        assert amp_type in ['bf16', '16', '32'], f"Invalid amp_type: {amp_type}"
        rank_zero_info(f'Using {amp_type} precision')
    else:
        amp_type = 32
        rank_zero_info(f'Using 32 bit precision')
    rank_zero_info(f'*' * 100)

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        precision=amp_type,
        callbacks=callbacks,
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=config.training.num_nodes,
        strategy=ddp_strategy,
        gradient_clip_val=config.training.get('gradient_clip_val'),
        gradient_clip_algorithm=config.training.get('gradient_clip_algorithm'),
        accumulate_grad_batches=args.update_every,
        logger=loggers,
        log_every_n_steps=config.training.log_every_n_steps,
        check_val_every_n_epoch=config.training.val_every_n_epoch,
        limit_val_batches=config.training.limit_val_batches,
    )

    # Train
    if config.training.ckpt_path == '':
        config.training.ckpt_path = None
    if args.test:
        trainer.test(model, datamodule=data, ckpt_path=config.training.ckpt_path)
    elif args.sample:
        device = torch.device('cuda')
        import pathlib
        with torch.serialization.safe_globals([pathlib.PosixPath]):
            # model = model.load_from_checkpoint(config.training.ckpt_path).to(device)
            # model.learning_rate = base_lr
            # model.max_epochs = config.training.epochs
            ckpt = torch.load(config.training.ckpt_path)
            print(ckpt.keys())
            # exit()
        model.sample(batch_size, device='cuda')
        # trainer.test(model, datamodule=data)
    else:
        trainer.fit(model, datamodule=data, ckpt_path=config.training.ckpt_path)

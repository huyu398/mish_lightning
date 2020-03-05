from argparse import ArgumentParser
from pathlib import Path
import yaml

from pytorch_lightning import Trainer
import torch
import numpy as np
import random

#  from src.model import Model

def main(hparams):
    # initialize random seed
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    #  model = Model(hparams)

    trainer = Trainer(
        max_epochs          = hparams.max_epochs,
        gpus                = hparams.gpus,
        checkpoint_callback = hparams.checkpoint_callback,
        early_stop_callback = hparams.early_stop_callback,
        use_amp             = hparams.use_amp,
        row_log_interval    = hparams.row_log_interval,
        log_save_interval   = hparams.log_save_interval,
        val_check_interval  = hparams.val_check_interval
    )
    trainer.configure_checkpoint_callback()
    trainer.checkpoint_callback.save_top_k = -1
    # trainer.fit(model)

def parse_arg():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--config', type=Path, default='./configs/model_config.yml')
    args, _ = parser.parse_known_args()

    if not args.config.exists():
        raise FileNotFoundError('Not found model config file.')

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    main(args)

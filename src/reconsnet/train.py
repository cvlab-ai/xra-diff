import torch
import argparse

from clearml import Task
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--save_every_n_epochs', type=int, default=20, help='Saving frequency')

    return parser.parse_args()


def train_with_clearml(task_name, *args):
    Task.init(task_name=task_name, project_name="reconstruction")
    train(task_name, *args)


def train(task_name, Model, dm):
    torch.set_float32_matmul_precision('medium')

    args = parse_args()
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=task_name + '-epoch-{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
    )
    model = Model(args.lr)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true"
    )
    trainer.fit(model, dm)

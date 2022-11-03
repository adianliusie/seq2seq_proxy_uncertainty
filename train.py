import os
import argparse
import logging
from statistics import mode

from handlers.trainer import Trainer
from utils.general import save_script_args

# Load logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    # Model arguments
    model_parser = argparse.ArgumentParser(description='Arguments for system and model configuration')
    model_parser.add_argument('--path', type=str, help='path to experiment')
    model_parser.add_argument('--transformer', default='t5-base',type=str, help='[bert, roberta, electra ...]')
    model_parser.add_argument('--maxlen', default=512, type=int, help='max length of transformer inputs')
    model_args, moargs = model_parser.parse_known_args()


    ### Training arguments
    train_parser = argparse.ArgumentParser(description='Arguments for training the system')
    train_parser.add_argument('--dataset', default='newscommentary-en-de', type=str, help='dataset to train the system on')
    train_parser.add_argument('--datasubset', default=None, type=int, help='size of data subset to use for debugging')
    train_parser.add_argument('--log-every', default=50, type=int, help='logging training metrics every number of steps')
    train_parser.add_argument('--val-every', default=200, type=int, help='logging validation metrics every number of steps')

    train_parser.add_argument('--num-gradient-accum', default=8, type=int, help='number of gradient accumululations')
    train_parser.add_argument('--num-steps', default=10_000, type=int, help='number of updates to train for')
    train_parser.add_argument('--num-warmup-steps', default=2000, type=int, help='number of warmup updates linearly increasing learning rate to train for')
    train_parser.add_argument('--num-tokens', default=1024, type=int, help='max number of tokens in a batch')
    train_parser.add_argument('--num-sequences', default=30, type=int, help='max number of sequences in a batch')
    train_parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')

    train_parser.add_argument('--loss', default='cross_entropy', type=str, help='loss function to use to train system')
    train_parser.add_argument('--wandb', action='store_true',    help='if set, will log to wandb')
    train_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')

    train_args, toargs = train_parser.parse_known_args()

    # Making sure no unkown arguments are given
    assert set(moargs).isdisjoint(toargs), f"{set(moargs) & set(toargs)}"

    logger.info(model_args.__dict__)
    logger.info(train_args.__dict__)
    
    trainer = Trainer(model_args.path, model_args)
    trainer.train(train_args)
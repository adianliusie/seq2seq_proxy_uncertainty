import os
import argparse
import logging
from statistics import mode

from trainer import Trainer
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
    model_parser.add_argument('--seed', default=None, type=int, help='random seed')
    model_args, other_args_1 = model_parser.parse_known_args()


    ### Training arguments
    train_parser = argparse.ArgumentParser(description='Arguments for training the system')
    train_parser.add_argument('--dataset', default='newscommentary-en-de', type=str, help='dataset to train the system on')
    train_parser.add_argument('--datasubset', default=None, type=int, help='size of data subset to use for debugging')
    train_parser.add_argument('--log-every', default=50, type=int, help='logging training metrics every number of steps')
    train_parser.add_argument('--val-every', default=200, type=int, help='logging validation metrics every number of steps')

    train_parser.add_argument('--num-steps', default=10000, type=int, help='number of updates to train for')
    train_parser.add_argument('--num-warmup-steps', default=1000, type=int, help='number of warmup updates linearly increasing learning rate to train for')
    train_parser.add_argument('--num-tokens', default=1024, type=int, help='number of tokens in a batch')
    train_parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')

    train_parser.add_argument('--loss', default='cross_entropy', type=str, help='loss function to use to train system')
    train_parser.add_argument('--wandb', action='store_true',    help='if set, will log to wandb')
    train_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')

    train_args, other_args_2 = train_parser.parse_known_args()

    # make sure no unkown arguments are given
    # TODO: Do we need this?
    assert set(other_args_1).isdisjoint(other_args_2), f"{set(other_args_1) & set(other_args_2)}"

    # Save arguments
    #save_script_args(os.path.join(model_args.path, 'CMD.log'))

    logger.info(model_args.__dict__), print()
    logger.info(train_args.__dict__), print()
    
    trainer = Trainer(model_args.path, model_args)
    trainer.train(train_args)
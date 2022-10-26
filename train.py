import os
import argparse
import pprint
from statistics import mode

from trainer import Trainer
from utils.general import save_script_args


if __name__ == '__main__':


    # Model arguments
    model_parser = argparse.ArgumentParser(description='Arguments for system and model configuration')
    model_parser.add_argument('--path', type=str, help='path to experiment')
    model_parser.add_argument('--transformer', default='t5-base',type=str, help='[bert, roberta, electra ...]')
    model_parser.add_argument('--max-len', default=512, type=int, help='max length of transformer inputs')
    model_parser.add_argument('--seed', default=None, type=int, help='random seed')
    model_args, _ = model_parser.parse_known_args()


    ### Training arguments
    train_parser = argparse.ArgumentParser(description='Arguments for training the system')
    train_parser.add_argument('--dataset', default='wmt16-en-de', type=str, help='dataset to train the system on')
    train_parser.add_argument('--datasubset', default=None, type=int, help='size of data subset to use for debugging')
    train_parser.add_argument('--log-every', default=50, type=int, help='logging training metrics every number of steps')
    train_parser.add_argument('--val-every', default=200, type=int, help='logging validation metrics every number of steps')

    train_parser.add_argument('--num-updates', default=10000, type=int, help='number of updates to train for')
    train_parser.add_argument('--num-tokens', default=4096, type=int, help='number of tokens in a batch')
    train_parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')

    train_parser.add_argument('--loss', default='cross_entropy', type=str, help='loss function to use to train system')
    train_parser.add_argument('--wandb', default=None, type=str, help='experiment name to use for wandb (and to enable)')
    train_args, _ = train_parser.parse_known_args()

    # Save arguments
    save_script_args(os.path.join(
        model_args.path, 'CMD.log'
    ))
    # pprint.pprint(model_args.__dict__), print()
    # pprint.pprint(train_args.__dict__), print()
    
    # seed_path = model_args.exp_path+'/'+str(model_args.seed_num)
    # assert(not os.path.isdir(seed_path))
    # trainer = Trainer(seed_path, model_args)
    # trainer.train(train_args)
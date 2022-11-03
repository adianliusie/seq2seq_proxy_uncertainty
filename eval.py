import os
import argparse
import logging
from statistics import mode

from handlers.evaluater import Evaluator
from utils.general import save_script_args

# Load logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    ### Decoding arguments
    decode_parser = argparse.ArgumentParser(description='Arguments for training the system')
    decode_parser.add_argument('--path', type=str, help='path to experiment')
    decode_parser.add_argument('--dataset', default='newscommentary-en-de-test', type=str, help='dataset to train the system on')
    decode_parser.add_argument('--datasubset', default=None, type=int, help='size of data subset to use for debugging')

    decode_parser.add_argument('--num-tokens', default=512, type=int, help='max number of tokens in a batch')
    decode_parser.add_argument('--num-sequences', default=10, type=int, help='max number of sequences in a batch')
    decode_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')

    decode_parser.add_argument('--decode-max-length', default=256, type=int, help='maximum output sequence length for decoding')
    decode_parser.add_argument('--num-beams', default=12, type=int, help='number of beams')
    # decode_parser.add_argument('--num-beam-groups', default=3, type=int, help='number of groups of beams')
    # decode_parser.add_argument('--diversity-penalty', default=0.0, type=float, help='penalizing similarity between groups')
    decode_parser.add_argument('--length-penalty', default=0.6, type=float, help='penalizing shorter sequences for larger values')
    decode_parser.add_argument('--no-repeat-ngram-size', default=5, type=int, help='no repeating n gram')
    decode_args = decode_parser.parse_args()

    logger.info(decode_args.__dict__)
    
    evaluator = Evaluator(decode_args.path)
    evaluator.decode(decode_args)

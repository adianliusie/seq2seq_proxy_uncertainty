import logging
import sacrebleu
from collections import namedtuple

from handlers.trainer import Trainer

from data.handler import DataHandler
from handlers.batcher import Batcher
from models.models import load_model 


# Create Logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def eosindex(decoding):
    try:
        return decoding.index('</s>')
    except ValueError:
        return len(decoding)


class Evaluator(Trainer):
    def __init__(self, path):
        self.exp_path = path
        args = self.load_args('model_args.json')
        self.setup_helpers(args)

    def decode(self, args: namedtuple):
        # Get dataset
        data = self.data_handler.prep_data_single(args.dataset)

        # Load model
        self.load_model()

        # Device management
        self.to(args.device)
    
        # Setup model for translation
        self.model.eval()

        # Set batcher to eval (no maxlen)
        self.batcher.eval()

        # Print number of model parameters
        self.log_num_params()

        # Create batched dataset
        dataloader = self.batcher(
            data = data, 
            numtokens = args.num_tokens, 
            numsequences = args.num_sequences, 
            return_last = True,
            shuffle = False
        )

        # Save all predictions
        pred = []

        logger.info("Starting Decode")
        for batch in dataloader:
            
            # Generate prediction
            output = self.model.generate(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_mask, 
                max_length = args.decode_max_length,
                num_beams = args.num_beams,
                length_penalty = args.length_penalty,
                no_repeat_ngram_size = args.no_repeat_ngram_size,
            )

            for i, out, ref in zip(batch.ex_id, output, batch.label_text):
                # Tokenzier decode one sample at a time
                out = self.data_handler.tokenizer.decode(out)
                out = out[:eosindex(out)]

                # Save all decoded outputs
                pred.append({
                    'id': i, 
                    'ref': ref, 
                    'out': out
                })

        score = sacrebleu.corpus_bleu(
            [p['out'] for p in pred],
            [[p['ref'] for p in pred]],
        ).score
        logger.info("Finished Decode")
        logger.info(f"Sacrebleu performance: {score}")

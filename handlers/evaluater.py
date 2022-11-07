import logging

import torch
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


def create_lengths_ref(label_id):
    # Get the index of eos for a single example
    return (label_id == 1).nonzero().item() + 1


def create_temporary_labels(output):
    # Get the index of eos
    eos = (output == 1).nonzero(as_tuple = True)[1]

    # Shift the output by a step
    label_ids = output[:, 1:]

    # Now ensure all tokens past eos are -100
    for i, idx in enumerate(eos):
        label_ids[i, idx:] = -100
    
    # Prune the label_ids to the shortest length
    label_ids = label_ids[:, :eos.max().item()]

    return label_ids.contiguous(), eos


def compute_log_confidence_unc(log_probs):
    confidence = log_probs.max(dim = -1).values
    confidence = -confidence
    return confidence


def compute_entropy(log_probs):
    entropy = log_probs * torch.exp(log_probs)
    entropy = entropy.sum(-1)
    return -entropy


class Evaluator(Trainer):
    def __init__(self, path):
        self.exp_path = path
        args = self.load_args('model_args.json')
        self.setup_helpers(args)

        # Need this for the proxy model, not the best solution...
        self.train_args = self.load_args('train-args.json')

    @torch.no_grad()
    def decode(self, args: namedtuple):
        # Get dataset
        data = self.data_handler.prep_data_single(args.dataset)

        # Load model
        self.load_model()

        # Set arguments
        if hasattr(self.model, 'set_arguments'):
            self.model.set_arguments(self.train_args)

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
        all_preds = []

        logger.info("Starting Decode")
        for (i, batch) in enumerate(dataloader):
            
            # Batch size will be repeatedly used
            batch_size = batch.input_ids.size(0)

            # Generate prediction
            output = self.model.generate(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_mask, 
                max_length = args.decode_max_length,
                num_beams = args.num_beams,
                length_penalty = args.length_penalty,
                no_repeat_ngram_size = args.no_repeat_ngram_size,
                num_return_sequences = args.num_beams,
                output_scores = True,
                return_dict_in_generate = True,
            )

            # Generate reference prediction
            reference_output = self.model(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_mask, 
                labels = batch.label_ids,
            )

            # Save all predictions within batch for logit generation
            batch_preds = []

            for batch_id in range(batch_size):

                # Generate teacher-forcing prediction
                reference_pred = reference_output.logits[batch_id].argmax(dim = -1)
                reference_pred_text = self.data_handler.tokenizer.decode(reference_pred, skip_special_tokens=True)
                proxies = reference_output.proxies[batch_id].item() if hasattr(reference_output, 'proxies') else None
                length = create_lengths_ref(batch.label_ids[batch_id])

                # Compute confidence and entropies:
                log_probs = torch.log_softmax(reference_output.logits[batch_id, :length], dim = -1)

                # Individual prediction
                pred = {
                    'id': batch.ex_id[batch_id],
                    'ref': batch.label_text[batch_id],
                    'ref-pred': reference_pred.cpu().tolist(),
                    'ref-pred-dec': reference_pred_text,
                    'ref-len': length,
                    'ref-conf': compute_log_confidence_unc(log_probs).cpu().tolist(),
                    'ref-ent': compute_entropy(log_probs).cpu().tolist(),
                    'proxies': proxies,
                }

                # Iterate over all beams
                for beam_id in range(args.num_beams):

                    # Get index of sample 
                    idx = beam_id + batch_id * args.num_beams

                    # Get the beam and decode
                    beam = output.sequences[idx]
                    out = self.data_handler.tokenizer.decode(beam, skip_special_tokens=True)

                    # Save all decoded outputs
                    pred[f'beam-{beam_id}'] = beam
                    pred[f'beam-{beam_id}-dec'] = out
                    pred[f'beam-{beam_id}-score'] = output.sequences_scores[idx].item()
                
                # Save all beams within a single list
                batch_preds.append(pred)

            # Iterate over all beams and generate logits for uncertainties
            for beam_id in range(args.num_beams):
                
                # Create a batch of labels
                temp_label_ids = [batch_preds[i][f'beam-{beam_id}'] for i in range(batch_size)]
                temp_label_ids = torch.stack(temp_label_ids, dim = 0)
                temp_label_ids, lengths = create_temporary_labels(temp_label_ids)

                # Feed through the model again to generate logits
                output = self.model(
                    input_ids = batch.input_ids, 
                    attention_mask = batch.attention_mask, 
                    labels = temp_label_ids,
                )

                # Get the log-probs and compute both confidence and entropy
                log_probs = torch.log_softmax(output.logits, dim = -1)

                # Computing confidence and entropy
                confidence = compute_log_confidence_unc(log_probs)
                entropy = compute_entropy(log_probs)

                # Adding example specific information
                for batch_id, length in enumerate(lengths):
                    
                    batch_preds[batch_id][f'beam-{beam_id}-len'] = length.item()
                    batch_preds[batch_id][f'beam-{beam_id}-conf'] = confidence[batch_id][:length].cpu().tolist()
                    batch_preds[batch_id][f'beam-{beam_id}-ent'] = entropy[batch_id][:length].cpu().tolist()
                    batch_preds[batch_id][f'beam-{beam_id}'] = batch_preds[batch_id][f'beam-{beam_id}'].cpu().tolist()

            # Save all predicitons in a master list
            all_preds.extend(batch_preds)

        # Corpus level scoring teacher-forcing
        refscore = sacrebleu.corpus_bleu(
            [p['ref-pred-dec'] for p in all_preds],
            [[p['ref'] for p in all_preds]],
        ).score

        # Corpus level scoring free-running
        freescore = sacrebleu.corpus_bleu(
            [p['beam-0-dec'] for p in all_preds],
            [[p['ref'] for p in all_preds]],
        ).score

        logger.info("Finished Decode")
        logger.info(f"TF Sacrebleu performance: {refscore}")
        logger.info(f"FR Sacrebleu performance: {freescore}")

        return [refscore, freescore] + all_preds
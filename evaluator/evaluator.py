from __future__ import print_function, division

import torch
import torchtext

from loss import NLLLoss

import numpy as np


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data, writer=None, n_iter=0):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
            writer (tensorboardX.SummaryWriter): logger for tensorboard
            n_iter (int): index of tensorboard logging
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        src_vocab = data.fields["src"].vocab
        ctx_vocab = data.fields["ctx"].vocab
        tgt_vocab = data.fields["tgt"].vocab
        pad = tgt_vocab.stoi[data.fields["tgt"].pad_token]

        with torch.no_grad():
            input_variables = None
            target_variables = None
            other = None

            for batch in batch_iterator:
                input_variables, input_lengths = getattr(batch, "src")
                context_variables, context_lengths = getattr(batch, "ctx")
                target_variables = getattr(batch, "tgt")

                decoder_outputs, decoder_hidden, other = model(
                    input_variables,
                    input_lengths.tolist(),
                    context_variables,
                    context_lengths,
                    target_variables)

                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

            # log text
            if writer:
                sample = np.random.randint(0, input_variables.shape[0])
                input_str = " ".join(list(map(lambda index: src_vocab.itos[index], input_variables[sample].cpu().numpy())))
                ctx_str = " ".join(list(map(lambda index: ctx_vocab.itos[index], context_variables[sample].cpu().numpy())))

                seq = torch.stack(other['sequence'], dim=1).squeeze(2)
                output_str = " ".join(list(map(lambda index: tgt_vocab.itos[index], seq[sample].cpu().numpy())))
                target_str = " ".join(list(map(lambda index: tgt_vocab.itos[index], target_variables[sample].cpu().numpy())))

                writer.add_text("input", input_str, n_iter)
                writer.add_text("context", ctx_str, n_iter)
                writer.add_text("output", output_str, n_iter)
                writer.add_text("target", target_str, n_iter)

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        if writer:
            writer.add_scalar("dev_loss", loss.get_loss(), n_iter)
            writer.add_scalar("dev_accuracy", accuracy, n_iter)

        return loss.get_loss(), accuracy

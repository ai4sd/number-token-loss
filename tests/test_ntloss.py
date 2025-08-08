import math

import numpy as np
import pytest
import torch
from torch import Tensor
from transformers import AutoTokenizer

from ntloss import NTLoss, NTLossDotProduct

TOKENIZER = AutoTokenizer.from_pretrained("t5-small")
VOCAB_SIZE = TOKENIZER.vocab_size


def make_logits(token_logit_value_dicts):
    """
    Build a (1 x T x V) tensor filled with -inf,
    then set the logits specified in token_logit_value_dicts.
    """
    seq_len = len(token_logit_value_dicts)
    logits = torch.full((1, seq_len, VOCAB_SIZE), -np.inf, dtype=torch.float32)
    for i, tok_dict in enumerate(token_logit_value_dicts):
        for tok_id, logit in tok_dict.items():
            logits[0, i, tok_id] = logit
    return logits


def dirac_logits(ids, peak_id, peak_value):
    """Perfectly confident distribution (all mass on one token)."""
    logits = torch.full((1, 1, VOCAB_SIZE), -1e6, dtype=torch.float32)
    logits[0, 0, peak_id] = 0.0
    return logits


def gaussian_logits(ids, peak_id, peak_value, sigma=1e-1):
    """
    Smooth bell curve over the reference tokens.
    """
    logits = torch.full((1, 1, VOCAB_SIZE), -1e-3, dtype=torch.float32)
    # Assumes that ids are ordered by their numerical value
    assert len(ids) == 11
    for idx, tok_id in enumerate(ids):
        logits[0, 0, tok_id] = -1 * np.abs((idx - peak_value) ** 2 / (2 * sigma**2))
    return logits


@pytest.mark.parametrize("loss_class", [NTLoss, NTLossDotProduct])
@pytest.mark.parametrize(
    "logits_dicts,label_tokens",
    [
        # positive logits scenario
        (
            [
                {"1": 1.0, "2": 1.2, "0": 0.5, "3": 1.5},
                {"1": 1.5, "2": 1.2, "0": 0.5, "3": 1.5},
                {"1": 1.0, "2": 1.2, "0": 0.5, "3": 1.5},
            ],
            ["1", "1", "a"],
        ),
        # mixed logits scenario
        (
            [
                {"0": -4.0, "1": 2.0, "2": -1.0},
                {"0": 1.5, "1": 0.5, "2": 1.2},
                {"3": -2.0, "4": 1.0, "5": -2.5},
            ],
            ["2", "1", "3"],
        ),
    ],
)
def test_ntloss_variants(loss_class, logits_dicts, label_tokens):
    # convert token strings to IDs
    token_logit_value_dicts = {
        # map token strings to token IDs upfront
        i: {
            TOKENIZER.convert_tokens_to_ids(tok): val
            for tok, val in logits_dicts[i].items()
        }
        for i in range(len(logits_dicts))
    }
    # build logits tensor
    # reorder into a list for our helper
    logits_list = [token_logit_value_dicts[i] for i in range(len(logits_dicts))]
    logits = make_logits(logits_list)

    # build labels tensor shape (1 x T)
    label_ids = [TOKENIZER.convert_tokens_to_ids(tok) for tok in label_tokens]
    labels = torch.tensor([label_ids], dtype=torch.long)

    # instantiate and run
    loss_fn = loss_class(tokenizer=TOKENIZER)
    loss = loss_fn(logits, labels)

    # assertions
    assert isinstance(loss, Tensor), "Loss should be a Python float"
    assert isinstance(loss.item(), float), "Loss should be a Python float"
    assert not math.isnan(loss), "Loss must not be NaN"


@pytest.mark.parametrize("loss_class", [NTLoss, NTLossDotProduct])
@pytest.mark.parametrize("logit_builder", [dirac_logits, gaussian_logits])
def test_correct_minimum(loss_class, logit_builder):
    loss_fn = loss_class(TOKENIZER)
    ref_tokens = [str(i) for i in range(10)] + ["A"]
    ref_ids = [TOKENIZER.convert_tokens_to_ids(t) for t in ref_tokens]

    # Guard: make sure all required tokens exist in the vocab
    assert all(i is not None and i >= 0 for i in ref_ids), "Missing token id"

    losses = torch.zeros(len(ref_ids), len(ref_ids), dtype=torch.float32)
    for i, (gt_token, gt_token_id) in enumerate(zip(ref_tokens, ref_ids)):
        labels = torch.tensor([[gt_token_id]], dtype=torch.long)
        for peak_idx, peak_id in enumerate(ref_ids):
            logits = logit_builder(ref_ids, peak_id, peak_idx)
            loss = loss_fn(logits, labels)
            losses[i, peak_idx] = loss.item()
            if logit_builder == gaussian_logits:
                print(logits[:, :, loss_fn.is_number_token])

        # TODO: Ensure that if GT is number and mass is on text, loss is at least as
        # high as for worst number prediction. This is not there yet and the reason
        # why we exclude the last token
        mins = torch.argsort(losses[i, :-1], dim=0)
        expected = torch.Tensor(
            sorted(range(10), key=lambda j: (abs(j - i), j)),
        ).long()

        if i == 10:
            assert torch.allclose(
                losses[i, :], torch.zeros_like(losses[i, :]), atol=1e-8
            ), "Loss should be zero when the ground-truth token is non-numeric."
        else:
            assert torch.equal(mins, expected), (
                "For a digit ground truth, loss must be minimal when the distribution "
                "peaks over the same digit."
            )

    assert not torch.isnan(losses).any(), "Encountered NaN in loss matrix"

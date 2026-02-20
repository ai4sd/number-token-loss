"""Test equivalence between NumberLevelLoss and NumberLevelLossLooped implementations."""

import pytest
import torch
from transformers import AutoTokenizer

from ntloss import NumberLevelLoss, NumberLevelLossLooped

# Use a simple tokenizer for testing
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

# Test on different devices if available
DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")


def make_logits(logits_dicts, vocab_size=None):
    """Helper to create logits tensor from list of dicts."""
    if vocab_size is None:
        vocab_size = len(TOKENIZER)

    logits = []
    for logit_dict in logits_dicts:
        logit_vec = torch.full((vocab_size,), -50.0)
        for tok_id, val in logit_dict.items():
            logit_vec[tok_id] = val
        logits.append(logit_vec)

    return torch.stack(logits).unsqueeze(0)  # Add batch dimension


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("float_level", [False, True])
@pytest.mark.parametrize("reweigh", [False, True])
def test_equivalence_simple(device, float_level, reweigh):
    """Test that both implementations produce identical results on simple inputs."""

    # Test sequence: "A 123 B 45"
    seq_tokens = ["A", "1", "2", "3", "B", "4", "5"]
    label_ids = TOKENIZER.convert_tokens_to_ids(seq_tokens)
    labels = torch.tensor([label_ids], dtype=torch.long)

    # Perfect predictions
    logits_dicts = []
    for tok in seq_tokens:
        tid = TOKENIZER.convert_tokens_to_ids(tok)
        logits_dicts.append({tid: 50.0})

    logits = make_logits(logits_dicts).to(device)
    labels = labels.to(device)

    # Create both loss functions
    loss_fn = NumberLevelLoss(TOKENIZER, float_level=float_level, reweigh=reweigh)
    loss_fn_looped = NumberLevelLossLooped(
        TOKENIZER, float_level=float_level, reweigh=reweigh
    )

    # Compute losses
    loss = loss_fn(logits, labels, reduction="none")
    loss_looped = loss_fn_looped(logits, labels, reduction="none")

    # Check shapes match
    assert loss.shape == loss_looped.shape, (
        f"Shape mismatch: {loss.shape} vs {loss_looped.shape}"
    )

    # Check values match (allowing for small numerical differences)
    torch.testing.assert_close(loss, loss_looped, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_equivalence_with_floats(device):
    """Test equivalence with decimal numbers."""

    # Test sequence: "A 12.34 B"
    seq_tokens = ["A", "1", "2", ".", "3", "4", "B"]
    label_ids = TOKENIZER.convert_tokens_to_ids(seq_tokens)
    labels = torch.tensor([label_ids], dtype=torch.long)

    # Perfect predictions
    logits_dicts = []
    for tok in seq_tokens:
        tid = TOKENIZER.convert_tokens_to_ids(tok)
        logits_dicts.append({tid: 50.0})

    logits = make_logits(logits_dicts).to(device)
    labels = labels.to(device)

    # Test with float_level=True
    loss_fn = NumberLevelLoss(TOKENIZER, float_level=True, reweigh=False)
    loss_fn_looped = NumberLevelLossLooped(TOKENIZER, float_level=True, reweigh=False)

    loss = loss_fn(logits, labels, reduction="none")
    loss_looped = loss_fn_looped(logits, labels, reduction="none")

    torch.testing.assert_close(loss, loss_looped, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_equivalence_multiple_numbers(device):
    """Test equivalence with multiple numbers in sequence."""

    # Test sequence: "123 456 789"
    seq_tokens = ["1", "2", "3", " ", "4", "5", "6", " ", "7", "8", "9"]
    label_ids = TOKENIZER.convert_tokens_to_ids(seq_tokens)
    labels = torch.tensor([label_ids], dtype=torch.long)

    # Perfect predictions
    logits_dicts = []
    for tok in seq_tokens:
        tid = TOKENIZER.convert_tokens_to_ids(tok)
        logits_dicts.append({tid: 50.0})

    logits = make_logits(logits_dicts).to(device)
    labels = labels.to(device)

    loss_fn = NumberLevelLoss(TOKENIZER, reweigh=False)
    loss_fn_looped = NumberLevelLossLooped(TOKENIZER, reweigh=False)

    loss = loss_fn(logits, labels, reduction="none")
    loss_looped = loss_fn_looped(logits, labels, reduction="none")

    torch.testing.assert_close(loss, loss_looped, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_equivalence_reductions(device, reduction):
    """Test equivalence with different reduction modes."""

    seq_tokens = ["A", "1", "2", "3", "B"]
    label_ids = TOKENIZER.convert_tokens_to_ids(seq_tokens)
    labels = torch.tensor([label_ids], dtype=torch.long)

    logits_dicts = []
    for tok in seq_tokens:
        tid = TOKENIZER.convert_tokens_to_ids(tok)
        logits_dicts.append({tid: 50.0})

    logits = make_logits(logits_dicts).to(device)
    labels = labels.to(device)

    loss_fn = NumberLevelLoss(TOKENIZER, reweigh=False)
    loss_fn_looped = NumberLevelLossLooped(TOKENIZER, reweigh=False)

    loss = loss_fn(logits, labels, reduction=reduction)
    loss_looped = loss_fn_looped(logits, labels, reduction=reduction)

    torch.testing.assert_close(loss, loss_looped, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob

import random
import time

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from ntloss import NTLoss, NTLossDotProduct, NumberLevelLoss, NumberLevelLossLooped


def get_available_devices():
    """Get all available devices for testing."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("seq_length", [1000, 5000])
@pytest.mark.parametrize("num_token_fraction", [0.01, 0.50])
def test_loss_calculation_speed(seq_length, num_token_fraction):
    """Compare speed of NTLoss, NTLossDotProduct, and NumberLevelLoss across devices.

    Parameters:
    - vocab_size: 10K with 10 digit tokens (0-9)
    - batch_size: 16
    - sequence_lengths: 100, 1K, 10K (parametrized)
    - fraction of number tokens: 1%, 10%, 50% (parametrized)
    - numbers occur as floats like 12.345 (digit-level tokenization)
    - tests on all available devices (CPU, CUDA, MPS)
    """
    # Create custom tokenizer with 10K vocab and digit-level tokenization
    vocab_size = 10000
    vocab = {}

    # Add digit tokens (0-9)
    for i in range(10):
        vocab[str(i)] = i

    # Add decimal point
    vocab["."] = 10

    # Add other tokens to reach 10K vocab
    for i in range(11, vocab_size):
        vocab[f"token_{i}"] = i

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="token_11"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok)

    # Generate sequence with floats
    def generate_float_tokens():
        """Generate tokens for a float like 12.345 (fixed 6 tokens)"""
        # Fixed: 2 digits before decimal, 3 digits after = 6 tokens total
        tokens = []
        for _ in range(2):
            tokens.append(str(random.randint(0, 9)))
        tokens.append(".")
        for _ in range(3):
            tokens.append(str(random.randint(0, 9)))

        return tokens

    # Calculate number of floats needed based on fraction
    # Each float is exactly 6 tokens (2 digits, dot, 3 digits)
    tokens_per_float = 6
    num_number_tokens = int(seq_length * num_token_fraction)
    # Round to nearest multiple of 6 to ensure complete floats
    num_number_tokens = (num_number_tokens // tokens_per_float) * tokens_per_float
    num_floats = num_number_tokens // tokens_per_float
    num_other_tokens = seq_length - num_number_tokens

    # Build label sequence
    label_tokens = []
    for _ in range(num_floats):
        label_tokens.extend(generate_float_tokens())

    # Pad with non-number tokens to reach exact sequence length
    for i in range(num_other_tokens):
        label_tokens.append(f"token_{random.randint(11, 100)}")

    # Shuffle to distribute numbers throughout sequence
    random.shuffle(label_tokens)

    # Ensure we have exactly seq_length tokens
    assert len(label_tokens) == seq_length, (
        f"Expected {seq_length} tokens, got {len(label_tokens)}"
    )

    # Convert to IDs
    label_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in label_tokens]

    # Test configuration
    batch_size = 16
    # Test both NumberLevelLoss implementations
    loss_classes = [NTLoss, NTLossDotProduct, NumberLevelLoss, NumberLevelLossLooped]
    devices = get_available_devices()

    # Store results for comparison
    results = {}

    print(f"\n{'=' * 80}")
    print(
        f"Benchmark: seq_len={seq_length}, num_frac={num_token_fraction:.0%}, batch_size={batch_size}"
    )
    print(f"{'=' * 80}")

    for device in devices:
        print(f"\nDevice: {device.upper()}")
        print(f"{'-' * 80}")

        device_results = {}

        for loss_class in loss_classes:
            # Create loss function
            if loss_class in (NumberLevelLoss, NumberLevelLossLooped):
                # NumberLevelLoss and NumberLevelLossLooped don't accept digit_level parameter (always True internally)
                loss_fn = loss_class(
                    tokenizer=tokenizer, vocab_size=vocab_size, reweigh=True
                )
            else:
                loss_fn = loss_class(
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                    digit_level=True,
                    reweigh=True,
                )

            # Create tensors on device
            labels = torch.tensor(
                [label_ids] * batch_size, dtype=torch.long, device=device
            )
            logits = torch.randn(batch_size, seq_length, vocab_size, device=device)

            # Warm-up runs (2 iterations)
            for _ in range(2):
                _ = loss_fn(logits, labels)

            # Synchronize before timing
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            # Timed runs (5 iterations)
            times = []
            loss = None  # Initialize for type checker
            for _ in range(5):
                start = time.perf_counter()
                loss = loss_fn(logits, labels)

                # Synchronize after computation
                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()

                end = time.perf_counter()
                times.append(end - start)

            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            device_results[loss_class.__name__] = avg_time

            # Verify loss is valid (loss is guaranteed to be set after loop)
            assert loss is not None
            assert torch.is_tensor(loss)
            assert not torch.isnan(loss)
            assert loss.item() >= 0

            print(
                f"  {loss_class.__name__:20s}: {avg_time * 1000:7.2f}ms (min: {min_time * 1000:6.2f}ms, max: {max_time * 1000:6.2f}ms)"
            )

        results[device] = device_results

        # Print relative comparison for this device
        print(f"\n  Relative performance (vs NTLoss):")
        ntloss_time = device_results["NTLoss"]
        for loss_name, loss_time in device_results.items():
            if loss_name != "NTLoss":
                ratio = loss_time / ntloss_time
                slower_faster = "slower" if ratio > 1 else "faster"
                print(f"    {loss_name:20s}: {ratio:.2f}x {slower_faster}")

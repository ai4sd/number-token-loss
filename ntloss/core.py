from abc import ABC, abstractmethod
from numbers import Number
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch._tensor import Tensor
from transformers import PreTrainedTokenizer

from .utils import is_number


class AbstractNTLoss(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        add_nt_to_vocab: bool = True,
        digit_nt_only: bool = True,
        weight_using_logits: bool = False,
    ):
        """
        NTL constructor.

        Args:
            tokenizer: Standard HF tokenizer
            add_nt_to_vocab: Whether to ensure at least all digits are in the vocab.
                Defaults to True
            digit_nt_only: Whether to ensure only digit tokens are considered number tokens,
                stabalizing training with NTL. Defaults to True.
            weight_using_logits: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to False.

        """
        super().__init__()
        self.tokenizer = tokenizer
        self.add_nt_to_vocab = add_nt_to_vocab
        self.digit_nt_only = digit_nt_only
        self.weight_using_logits = weight_using_logits

        self.setup_number_tokens()

        self.max_dist = torch.tensor(0.0)

    def setup_number_tokens(self):
        """Setting up attributes needed by NT loss"""

        # Add digits to vocab if not there yet.
        vocab_size = len(self.tokenizer)
        if self.add_nt_to_vocab:
            new_tokens = self.tokenizer.add_tokens(list(map(str, range(10))))
        if vocab_size < len(self.tokenizer) and new_tokens > 0:
            logger.warning(f"Added {new_tokens} new tokens for number token loss")
        vocab = self.tokenizer.get_vocab()
        self.number_values = torch.full((len(vocab),), float("nan"))

        # Try to convert each token to a float after stripping the space prefix
        for token, id in vocab.items():
            if is_number(token, finite=True):
                if self.digit_nt_only:
                    # NOTE: This check ensures number token value only occurs for digits, not for multi-digit numbers (123)
                    # This stabilizes training with NTL. Can be altered though, see paper experiments.
                    if -1 <= float(token) <= 9 and len(token.lstrip(" ")) == 1:
                        self.number_values[id] = float(token)
                else:
                    self.number_values[id] = float(token)

        self.is_number_token = ~torch.isnan(self.number_values)
        self.number_values_dense = self.number_values[self.is_number_token]

    @abstractmethod
    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> Tensor: ...

    def __call__(self, *args, **kwargs):
        """Alias to self.forward"""
        return self.forward(*args, **kwargs)

    def apply_weight_nt_logits(
        self,
        logits: Tensor,
        loss: Tensor,
        valid_positions: Tensor,
    ) -> Tensor:
        """
        Scale the NT loss element-wise using the logit weight on number tokens.

        Args:
            logits: Tensor of shape BS x T x V
            loss: 1D Tensor of size BS*NT with the computed NT losses
            valid_positions: Tensor of shape BS x T indicating for which tokens
                the NT loss should be computed

        Returns:
            A 1D Tensor of size BS*NT with the scaled NT losses

        """

        # Take softmax over logits of all tokens in vocab and compute NT logit weight
        softmax_probs_all = F.softmax(logits, dim=-1)
        self.nt_logit_weight = torch.sum(
            softmax_probs_all[:, :, self.is_number_token], dim=-1
        )[valid_positions]

        # Apply weights for NTL element-wise
        loss *= self.nt_logit_weight

        # Apply regularization
        loss += (
            1.01
            * self.max_dist.to(dtype=loss.dtype, device=loss.device)
            * (1 - self.nt_logit_weight)
        )

        return loss


class NTLossDotProduct(AbstractNTLoss):
    """Class for NT losses that produce a token-wise numerical output"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        add_nt_to_vocab: bool = True,
        digit_nt_only: bool = True,
        weight_using_logits: bool = False,
        loss_function: Callable = F.mse_loss,
    ):
        """
        NTL constructor.

        Args:
            tokenizer: NTLTokenizer with necessary attributes like is_number_token etc.
            add_nt_to_vocab: Whether to ensure at least all digits are in the vocab.
                Defaults to True
            digit_nt_only: Whether to ensure only digit tokens are considered number tokens,
                stabalizing training with NTL. Defaults to True.
            weight_using_logits: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to False.
            loss_function: Function to apply on the delta between the ground truth number
                and the obtained dot product (nt-probs * token-values).

        """
        super().__init__(
            tokenizer=tokenizer,
            add_nt_to_vocab=add_nt_to_vocab,
            digit_nt_only=digit_nt_only,
            weight_using_logits=weight_using_logits,
        )
        self.loss_function = loss_function

        self.setup_max_dist()

    def setup_max_dist(self):
        """
        Set up the maximum distance between the number tokens based on the selected loss function.
        """

        # Extract the number token values and get the minimum and maximum
        vals = self.number_values_dense.unsqueeze(0)
        max_val = vals.max()
        min_val = vals.min()

        # Compute the largest value the loss function used in NT loss computation can get
        # Make sure to account for possibility of assymetrical loss function
        self.max_dist = torch.maximum(
            torch.abs(self.loss_function(min_val, max_val)),
            torch.abs(self.loss_function(max_val, min_val)),
        )

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> Tensor:
        """
        Computes the NTL based on the dot product between token values and their probs.

        Args:
            logits: Tensor of shape BS x T x V
            labels: Tensor of shape BS x T
            loss_mask: Optional tensor of BS x T
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".

        Returns:
            Loss tensor
                0-D if reduction=="mean"|"sum"
                BS x T if reduction=="none"
        """
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NTLossDotProduct are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NTLossDotProduct are empty!")

        labels = labels.masked_fill(labels == -100, 0)

        # Create a mask to filter out non-digit tokens
        y = self.number_values[labels]
        valid_positions = ~torch.isnan(y)

        # Apply the loss_mask to lower importance of number tokens before the final answer
        label_mask = (
            loss_mask[valid_positions]
            if loss_mask is not None
            else torch.ones_like(labels, dtype=logits.dtype)[valid_positions]
        )

        # If no digit tokens in batch, or total of the relevant loss_mask is zero, no need for upcoming calculations
        if (torch.count_nonzero(valid_positions) == 0) or (
            torch.count_nonzero(label_mask) == 0
        ):
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=logits.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(labels, dtype=logits.dtype)
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        # apply softmax and get number labels
        bs, seq_len, _ = logits.size()
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)

        # compute the weighted average of number tokens
        yhat = torch.sum(
            softmax_probs[valid_positions] * self.number_values_dense, dim=-1
        )

        # Apply specified loss function to y and yhat
        loss = self.loss_function(yhat, y[valid_positions], reduction="none")

        # If weight_using_logits: compute weights for NTL based on logits
        if self.weight_using_logits:
            loss = self.apply_weight_nt_logits(
                logits=logits, loss=loss, valid_positions=valid_positions
            )

        if reduction == "mean":
            # Mean pooling (weighted by loss mask)
            loss = torch.dot(
                loss.flatten(), label_mask.flatten()
            ) / torch.count_nonzero(label_mask)
        elif reduction == "sum":
            loss = torch.dot(loss.flatten(), label_mask.flatten())
        elif reduction == "none":
            # Cast loss for number tokens back to Tensor of size BS x T
            loss_ = torch.zeros(valid_positions.view(-1).size()).to(loss.device)
            loss_[valid_positions.view(-1)] = loss * label_mask
            loss = loss_.view(bs, seq_len)

            assert torch.sum(loss[~valid_positions]) == 0, (
                "NTLossDotProduct computed for non-digit tokens!"
            )

        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        return loss


class NTLoss(AbstractNTLoss):
    """Class for Wasserstein-based NTLoss. This is the default as per our paper."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        add_nt_to_vocab: bool = True,
        digit_nt_only: bool = True,
        weight_using_logits: bool = False,
        squash_factor: Optional[float] = None,
    ):
        """
        NTL constructor.

        Args:
            tokenizer: NTLTokenizer with necessary attributes like is_number_token etc.
            add_nt_to_vocab: Whether to ensure at least all digits are in the vocab.
                Defaults to True
            digit_nt_only: Whether to ensure only digit tokens are considered number tokens,
                stabalizing training with NTL. Defaults to True.
            weight_using_logits: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to False.
            squash_factor: The optional squashing factor for the NTL.
        """
        super().__init__(
            tokenizer=tokenizer,
            add_nt_to_vocab=add_nt_to_vocab,
            digit_nt_only=digit_nt_only,
            weight_using_logits=weight_using_logits,
        )

        self.squash_factor = squash_factor
        self.setup_distance_lookup(squash_factor)

    def setup_distance_lookup(
        self,
        squash_factor: Optional[float] = None,
    ):
        """
        Set up a lookup table for the distances between the number tokens.
        Use squash_factor to control by what factor the farthest number token is worse than the closest, incorrect number token.
        If not squash_factor is not set: with 10 number tokens (0-9), the squashing factor is 9.

        Args:
            squash_factor: The optional squashing factor used.

        """

        # Get token ids for number tokens
        num_ids = torch.nonzero(self.is_number_token, as_tuple=True)[0]
        # Create mapping from number token ids to their index in order of appearance in vocab
        vocab_to_dist_idx = torch.full((len(self.tokenizer),), -1, dtype=torch.long)
        vocab_to_dist_idx[num_ids] = torch.arange(num_ids.size(0), dtype=torch.long)

        # Build NxN abs-diff matrix
        vals = self.number_values_dense.unsqueeze(0)  # (1 x N)
        diff = torch.abs(vals - vals.t())  # (N x N)

        if isinstance(squash_factor, Number):
            assert squash_factor > 1, (
                f"The squash factor can't be equal to or smaller than 1, please use a different squashing factor than {squash_factor}"
            )

            # Mask out zeros to find the smallest nonzero diff
            inf = torch.finfo(diff.dtype).max
            diff_nonzero = diff.masked_fill(diff == 0, inf)
            global_min_nz = diff_nonzero.min()
            # Find largest diff
            global_max = diff.max()

            # Compute scaling factor based on indicated squash factor
            scale = (squash_factor - 1) / (global_max - global_min_nz)
            # Scale the absolute differences using scaling factor
            lookup = 1 + (diff - global_min_nz) * scale
            lookup[diff == 0] = 0.0

            additional_log_info = f", used a squashing factor of {squash_factor}."
        else:
            lookup = diff
            additional_log_info = ""

        self.vocab_to_dist_idx = vocab_to_dist_idx
        self.dist_lookup = lookup
        self.max_dist = lookup.max()

        logger.info(f"Done setting up the distance lookup table{additional_log_info}")

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> Tensor:
        """
        Computes the NTL.

        Args:
            logits: Tensor of shape BS x T x V
            labels: Tensor of shape BS x T
            loss_mask: Optional tensor of BS x T
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".

        Returns:
            Loss tensor
                0-D if reduction=="mean"|"sum"
                BS x T if reduction=="none"

        """

        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        labels = labels.clone().masked_fill(labels == ignore_index, 0)

        # Create a mask to filter out non-digit tokens
        y = self.number_values[labels]
        valid_positions = ~torch.isnan(y)

        # Apply the loss_mask to lower importance of number tokens before the final answer
        label_mask = (
            loss_mask[valid_positions]
            if loss_mask is not None
            else torch.ones_like(labels, dtype=logits.dtype)[valid_positions]
        )

        # If no digit tokens in batch, or total of the relevant loss_mask is zero, no need for upcoming calculations
        if (torch.count_nonzero(valid_positions) == 0) or (
            torch.count_nonzero(label_mask) == 0
        ):
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=logits.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(labels, dtype=logits.dtype)
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        # apply softmax and get number labels
        bs, seq_len, _ = logits.size()
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)

        # get distance between the true numbers and all possible number values from lookup table
        abs_diff = self.dist_lookup.to(device=labels.device)[
            self.vocab_to_dist_idx.to(device=labels.device)[labels[valid_positions]]
        ]

        # loss is the absolute difference weighted by the softmax probs
        loss = (abs_diff * softmax_probs[valid_positions]).sum(dim=-1)

        # If weight_using_logits: compute weights for NTL based on logits
        if self.weight_using_logits:
            loss = self.apply_weight_nt_logits(
                logits=logits, loss=loss, valid_positions=valid_positions
            )

        if reduction == "mean":
            # Mean pooling (weighted by loss mask)
            loss = torch.dot(
                loss.flatten(), label_mask.flatten()
            ) / torch.count_nonzero(label_mask)
        elif reduction == "sum":
            loss = torch.dot(loss.flatten(), label_mask.flatten())
        elif reduction == "none":
            # Cast loss for number tokens back to Tensor of size BS x T
            loss_ = torch.zeros(valid_positions.view(-1).size()).to(loss.device)
            loss_[valid_positions.view(-1)] = loss * label_mask
            loss = loss_.view(bs, seq_len)

            assert torch.sum(loss[~valid_positions]) == 0, (
                "NumberTokenLoss computed for non-digit tokens!"
            )

        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        return loss

from abc import ABC, abstractmethod
from numbers import Number
from typing import Callable, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from loguru import logger
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from transformers import PreTrainedTokenizer

from .utils import is_number


class AbstractNTLoss(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        vocab_size: Optional[int] = None,
        digit_level: bool = True,
        reweigh: bool = True,
    ):
        """
        NTL constructor.

        Args:
            tokenizer: Standard HF tokenizer.
            vocab_size: Optional user-provided vocab size. If not provided, the
                tokenizer's vocab size is used.
            digit_level: Whether to ensure only digits are considered number tokens,
                stabilizing training with NTL. Defaults to True. Used for most
                experiments in the ICML paper.
            reweigh: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to True.
                NOTE: The ICML paper does *not* use this option which can lead to
                incorrect loss if most mass is placed outside of the number tokens.

        """
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size if vocab_size is not None else len(self.tokenizer)
        self._vocab_size_validated = False
        self.digit_level = digit_level
        self.reweigh = reweigh

        self.setup_number_tokens()

        self.max_dist = torch.tensor(0.0)

    def setup_number_tokens(self):
        """Setting up attributes needed by NT loss"""

        # Add digits to vocab if not there yet.
        vocab_size = len(self.tokenizer)
        if self.digit_level:
            new_tokens = self.tokenizer.add_tokens(list(map(str, range(10))))
        if vocab_size < len(self.tokenizer) and new_tokens > 0:
            logger.warning(f"Added {new_tokens} new tokens for number token loss")
        vocab = self.tokenizer.get_vocab()
        self.number_values: FloatTensor = torch.full((self.vocab_size,), float("nan"))

        # Try to convert each token to a float after stripping the space prefix
        for token, id in vocab.items():
            if is_number(token, finite=True):
                if self.digit_level:
                    # NOTE: This check ensures number token value only occurs for digits, not for multi-digit numbers (123)
                    # This stabilizes training with NTL. Can be altered though, see paper experiments.
                    # Excludes tokens that are numbers in other languages like ႘ and tokens with space pre-/postfix like ` 2`.
                    if token.isascii() and -1 <= float(token) <= 9 and len(token) == 1:
                        self.number_values[id] = float(token)
                else:
                    self.number_values[id] = float(token)

        self.is_number_token = ~torch.isnan(self.number_values)
        if self.is_number_token.sum() == len(self.is_number_token):
            raise ValueError(
                "At least one token needs to be not a number, otherwise `ignore_index` cannot be set up safely"
            )
        self.nan_id = torch.where(~self.is_number_token)[0][0].item()
        self.number_values_dense = self.number_values[self.is_number_token]

        if self.digit_level and (num_nts := len(self.number_values_dense)) != 10:
            logger.error(
                f"You requested digit-level but {num_nts} number tokens were identified: {self.number_values_dense}"
            )
        self.number_token_ids = torch.nonzero(
            self.is_number_token, as_tuple=False
        ).squeeze(1)
        self._nt_ids_cache: dict[torch.device, torch.Tensor] = {}

    @abstractmethod
    def forward(
        self,
        logits: FloatTensor,
        labels: LongTensor,
        loss_weights: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> Tensor: ...

    def __call__(self, *args, **kwargs):
        """Alias to self.forward"""
        return self.forward(*args, **kwargs)

    def reweigh_fn(
        self,
        logits: Tensor,
        loss: Tensor,
        number_token_positions: Tensor,
    ) -> Tensor:
        """
        Scale the NT loss element-wise using the logit weight on number tokens.
        NOTE: This reweighing ensures that if ground truth is a number token
            but most probability mass is on text tokens, the loss will be *higher*
            than the worst possible number token. Mostly to accelerate early training.
        NOTE: Since NT mass is only calculated at loss positions, the overhead is tiny.

        Args:
            logits: 3D Tensor of shape BS x T x V.
            loss: 1D Tensor over all number tokens in batch.
            number_token_positions: 2D Tensor of shape BS x T indicating for which tokens
                the NT loss was computed.

        Returns:
            A 1D Tensor over all number tokens in batch with the scaled NT losses.
        """

        nt_logits = logits[number_token_positions]
        nt_ids = self._nt_ids_cache.get(nt_logits.device)
        if nt_ids is None:
            nt_ids = self.number_token_ids.to(nt_logits.device)
            self._nt_ids_cache[nt_logits.device] = nt_ids

        # Softmax and mass only for relevant positions
        nt_probs = torch.softmax(nt_logits, dim=-1)  # (K, V)
        nt_mass = nt_probs.index_select(dim=-1, index=nt_ids).sum(dim=-1)

        # Apply regularization (in place is faster)
        loss.mul_(nt_mass)
        # NOTE: We could consider reweighing here with the max for that label token
        # rather than the global max
        loss.add_(
            1.01
            * self.max_dist.to(dtype=loss.dtype, device=loss.device)
            * (1 - nt_mass)
        )
        return loss

    def _validate_inputs(
        self,
        logits: FloatTensor,
        labels: Optional[LongTensor],
        loss_weights: Optional[Tensor],
    ):
        """Private method to perform size and type checks."""
        if (td := len(logits.shape)) != 3 or logits.numel() == 0:
            raise ValueError(
                f"Logits have to be non-empty 3D Tensor, not {td}D with {logits.numel()} elements"
            )
        if not torch.is_floating_point(logits):
            raise TypeError("Logits have to be FloatTensor.")
        if labels is None:
            return
        if not labels.dtype == torch.long:
            raise TypeError(f"Labels have to be LongTensor, not {type(labels)}")
        if (b := labels.shape) != (a := logits.shape[:-1]):
            raise ValueError(
                f"Logit and label sizes of first 2 dims have to match: {a} vs {b}"
            )

        if (td := len(labels.shape)) != 2 or labels.numel() == 0:
            raise ValueError(
                f"Labels have to be non-empty 2D Tensor, not {td}D with {labels.numel()} elements"
            )
        if loss_weights is not None:
            if loss_weights.shape != labels.shape:
                raise ValueError(
                    "Loss mask has to be 2D Tensor of same shape as labels."
                )
            if torch.any(loss_weights < 0):
                raise ValueError("loss_mask must be ≥ 0.")

        if not self._vocab_size_validated:
            logits_vocab_size = logits.shape[-1]
            if logits_vocab_size != self.vocab_size:
                raise ValueError(
                    f"The current `vocab_size` ({self.vocab_size}) does not match the model's vocab size"
                    f"logit dimension ({logits_vocab_size}). Please check the value."
                )
            self._vocab_size_validated = True

    def _prepare_number_token_targets(
        self, labels: LongTensor, loss_weights: Optional[Tensor], ignore_index: int
    ) -> Tuple[FloatTensor, Tensor]:
        """
        Prepare number-token targets and masks.

        Args:
            labels: 2D Tensor of shape BS x T.
            loss_weights: Optional 2D Tensor of shape BS x T with loss weight for each token.
            ignore_index: Label ID to ignore. Defaults to -100.

        Returns:
            y: 2D Float Tensor of shape BS x T with target numeric values (NaN for non-number tokens).
            loss_weight: 1D Tensor with a potentially individual loss weight for each number token position.
        """
        labels = cast(
            LongTensor, labels.masked_fill(labels == ignore_index, self.nan_id)
        )
        # Create a mask to filter out non-digit tokens
        y = self.number_values.to(device=labels.device)[labels]
        number_token_positions = ~torch.isnan(y)
        loss_weights = (
            loss_weights[number_token_positions]
            if loss_weights is not None
            else torch.ones_like(labels, device=labels.device)[number_token_positions]
        )
        return cast(FloatTensor, y), loss_weights

    @staticmethod
    def _apply_reduction(
        loss: Tensor,
        reduction: str,
        loss_weights: Tensor,
        number_token_positions: Tensor,
        logits: Tensor,
    ) -> Tensor:
        """
        Applies the specified reduction type to the calculated loss.

        This method handles 3 types of reduction: "mean", "sum", and "none".
        For "mean" and "sum", it applies weighting using `loss_weights`.
        For "none", it reshapes the loss back to the original batch and sequence
        dimensions.

        Args:
            loss: 1D Tensor containing the loss for each number token in the batch.
            reduction: The reduction method ("mean", "sum", or "none").
            loss_weights: 1D Tensor with a loss weight for each number token.
            number_token_positions: 2D boolean tensor of shape BS x T indicating
                the positions of number tokens.
            logits: 3D Tensor of shape BS x T x V, used to get the original shape
                for the "none" reduction.

        Returns:
            A Tensor representing the reduced loss:
                - 0D tensor if `reduction` is "mean" or "sum".
                - 2D Tensor of shape BS x T if `reduction` is "none".
        """
        if reduction == "mean":
            # Mean pooling (weighted by loss mask)
            loss = torch.dot(
                loss.flatten(), loss_weights.flatten()
            ) / loss_weights.sum().clamp_min(torch.finfo(loss.dtype).eps)
        elif reduction == "sum":
            loss = torch.dot(loss.flatten(), loss_weights.flatten())
        elif reduction == "none":
            # Cast loss for number tokens back to Tensor of size BS x T
            loss_ = torch.zeros(number_token_positions.numel()).to(loss.device)
            loss_[number_token_positions.view(-1)] = loss * loss_weights
            bs, seq_len, _ = logits.size()
            loss = loss_.view(bs, seq_len)

            assert torch.sum(loss[~number_token_positions]) == 0, (
                "NumberTokenLoss computed for non-digit tokens!"
            )

        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        return loss


class NTLossDotProduct(AbstractNTLoss):
    """Class for NT losses that produce a token-wise numerical output."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        vocab_size: Optional[int] = None,
        digit_level: bool = True,
        reweigh: bool = True,
        loss_function: Callable = F.mse_loss,
    ):
        """
        Referred to as NTL-L_p in the paper.

        Args:
            tokenizer: NTLTokenizer with necessary attributes like is_number_token etc.
            vocab_size: Optional user-provided vocab size. If not provided, the
                tokenizer's vocab size is used.
            digit_level: Whether to ensure only digits are considered number tokens,
                stabilizing training with NTL. Defaults to True. Used for most
                experiments in the ICML paper.
            reweigh: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to True.
                NOTE: The ICML paper does *not* use this option which can lead to
                incorrect loss if most mass is placed outside of the number tokens.
            loss_function: Function to apply on the delta between the ground truth number
                and the obtained dot product (nt-probs * token-values). Defaults to
                MSE, but MAE, Huber etc are also compatible.
        """
        super().__init__(
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            digit_level=digit_level,
            reweigh=reweigh,
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
        # Make sure to account for possibility of asymmetrical loss function
        self.max_dist = torch.maximum(
            torch.abs(self.loss_function(min_val, max_val)),
            torch.abs(self.loss_function(max_val, min_val)),
        )

    def predict_numbers(self, logits: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        Calculates token-level numerical prediction.
        NOTE: This calculates numerical predictions for *all* tokens, not just where
        label is a number token.

        Args:
            logits: 3D FloatTensor of shape BS x T x V.

        Returns:
            yhat: 2D FloatTensor BS x T containing numerical predictions.
            nt_mass: 2D FloatTensor BS x T with the cumulated mass assigned to all number tokens.
        """
        self._validate_inputs(logits, labels=None, loss_weights=None)

        # Calculate the token-level predictions
        yhat = self._get_dot_product(logits=logits)

        probs_all = F.softmax(logits, dim=-1)
        probs_nt = probs_all[:, :, self.is_number_token]
        nt_mass = probs_nt.sum(dim=-1)
        return yhat, cast(FloatTensor, nt_mass)

    def _get_dot_product(
        self, logits: FloatTensor, number_token_positions: Optional[BoolTensor] = None
    ) -> FloatTensor:
        """
        Applies dot product of number token values and their predicted probabilites.

        Args:
            logits: 3D FloatTensor of shape BS x T x V.
            number_token_positions: Optional 2D BoolTensor (BS x T) containing locations
                of number tokens.

        Returns:
            If `number_token_positions` is None, 2D FloatTensor of shape BS x T.
            Otherwise, 1D FloatTensor containing the predictions for the number tokens.
        """
        # apply softmax solely over the number token indices
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)
        values = self.number_values_dense.to(device=logits.device, dtype=logits.dtype)

        # compute the weighted average of number tokens
        if number_token_positions is None:
            # Calculate for all tokens
            yhat = torch.sum(softmax_probs * values, dim=-1)
        else:
            # Calculate selectively where labels are number tokens
            yhat = torch.sum(softmax_probs[number_token_positions] * values, dim=-1)
        return cast(FloatTensor, yhat)

    def forward(
        self,
        logits: FloatTensor,
        labels: LongTensor,
        loss_weights: Optional[Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> Tensor:
        """
        Computes the NTL based on the dot product between token values and their probs.

        Args:
            logits: 3D Tensor of shape BS x T x V.
            labels: 2D Tensor of shape BS x T.
            loss_weights: 2D Optional tensor of BS x T with token-wise loss weights.
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".
            ignore_index: The token ID to ignore in the labels. Defaults to -100.

        Returns:
            Loss tensor
                OD if reduction=="mean"|"sum"
                BS x T if reduction=="none"
        """
        self._validate_inputs(logits, labels, loss_weights)

        y, loss_weights = self._prepare_number_token_targets(
            labels, loss_weights, ignore_index
        )
        loss_weights = loss_weights.to(logits.dtype)
        number_token_positions = cast(BoolTensor, ~torch.isnan(y))

        # If no digit tokens in batch, or total of the relevant loss weights is zero, no need for upcoming calculations
        if not number_token_positions.any() or not loss_weights.any():
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=logits.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(
                    labels, dtype=logits.dtype, device=labels.device
                )
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        yhat = self._get_dot_product(
            logits=logits, number_token_positions=number_token_positions
        )

        # Apply specified loss function to y and yhat
        loss = self.loss_function(yhat, y[number_token_positions], reduction="none")

        # If reweigh: compute weights for NTL based on logits
        if self.reweigh:
            loss = self.reweigh_fn(
                logits=logits, loss=loss, number_token_positions=number_token_positions
            )

        loss = self._apply_reduction(
            loss=loss,
            reduction=reduction,
            loss_weights=loss_weights,
            number_token_positions=number_token_positions,
            logits=logits,
        )

        return loss


class NTLoss(AbstractNTLoss):
    """Class for Wasserstein-based NTLoss. This is the default in the ICML paper."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        vocab_size: Optional[int] = None,
        digit_level: bool = True,
        reweigh: bool = True,
        squash_factor: Optional[float] = None,
    ):
        """
        NTL constructor for the Wasserstein-based NTLoss.

        Args:
            tokenizer: Any HuggingFace tokenizer.
            vocab_size: Optional user-provided vocab size. If not provided, the
                tokenizer's vocab size is used.
            digit_level: Whether to ensure only digits are considered number tokens,
                stabilizing training with NTL. Defaults to True. Used for most
                experiments in the ICML paper.
            reweigh: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to True.
                NOTE: The ICML paper does *not* use this option which can lead to
                incorrect loss if most mass is placed outside of the number tokens.
            squash_factor: The optional squashing factor for the NTL. If provided,
                this number denotes the factor by which predicting the largest number
                token is worse than predicting the closest incorrect number token.
                E.g., with digit-level tokenization this factor is 9. Setting this
                to 1 will recover cross entropy. This argument is intended to handle
                irregular vocabs with large numerical token values.
        """
        super().__init__(
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            digit_level=digit_level,
            reweigh=reweigh,
        )

        self.squash_factor = squash_factor
        self.setup_distance_lookup(squash_factor)

    def setup_distance_lookup(
        self,
        squash_factor: Optional[float] = None,
    ) -> None:
        """
        Set up a lookup table for the distances between the number tokens.
        Use squash_factor to control by what factor the farthest number token is worse than the closest, incorrect number token.
        If not squash_factor is not set: with 10 number tokens (0-9), the squashing factor is 9.
        NOTE: With a squashing factor of 1, this basically collapses to cross entropy.

        Args:
            squash_factor: The optional squashing factor used.
        """

        # Get token ids for number tokens
        num_ids = torch.nonzero(self.is_number_token, as_tuple=True)[0]
        # Create mapping from number token ids to their index in order of appearance in vocab:
        # e.g. token "3" -> id 519 -> dist_idx 1, then abs dist to 3 for other NT values will be found in row/column 1
        vocab_to_dist_idx = torch.full((self.vocab_size,), -1, dtype=torch.long)
        # Use arange to ensure order of appearance
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

        else:
            lookup = diff

        self.vocab_to_dist_idx = vocab_to_dist_idx
        self.dist_lookup = lookup
        self.max_dist = lookup.max()

    def forward(
        self,
        logits: FloatTensor,
        labels: LongTensor,
        loss_weights: Optional[Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> Tensor:
        """
        Computes the NTL.

        Args:
            logits: 3D Tensor of shape BS x T x V.
            labels: 2D Tensor of shape BS x T.
            loss_weights: Optional 2D tensor of BS x T with token-specific weights.
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".
            ignore_index: The token ID to ignore in the labels. Defaults to -100.

        Returns:
            Loss tensor
                OD if reduction=="mean"|"sum"
                BS x T if reduction=="none"

        """
        self._validate_inputs(logits, labels, loss_weights)

        y, loss_weights = self._prepare_number_token_targets(
            labels, loss_weights, ignore_index
        )
        loss_weights = loss_weights.to(logits.dtype)
        number_token_positions = ~torch.isnan(y)

        # If no digit tokens in batch, or total of the relevant loss_weights is zero, no need for upcoming calculations
        if not number_token_positions.any() or not loss_weights.any():
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=logits.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(
                    labels, dtype=logits.dtype, device=labels.device
                )
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        # apply softmax and get number labels
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)

        # get distance between the true numbers and all possible number values from lookup table
        abs_diff = self.dist_lookup.to(dtype=logits.dtype, device=logits.device)[
            self.vocab_to_dist_idx.to(device=labels.device)[
                labels[number_token_positions]
            ]
        ]

        # loss is the absolute difference weighted by the softmax probs
        loss = (abs_diff * softmax_probs[number_token_positions]).sum(dim=-1)

        # If reweigh: compute weights for NTL based on logits
        if self.reweigh:
            loss = self.reweigh_fn(
                logits=logits, loss=loss, number_token_positions=number_token_positions
            )

        loss = self._apply_reduction(
            loss=loss,
            reduction=reduction,
            loss_weights=loss_weights,
            number_token_positions=number_token_positions,
            logits=logits,
        )

        return loss


class NumberLevelLoss(NTLossDotProduct):
    """Calculate NTL on a per-number (rather than per-token) basis."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        vocab_size: Optional[int] = None,
        float_level: bool = False,
        reweigh: bool = True,
        max_number_length: int = 20,
    ):
        """
        NTL constructor for the number-level NTLoss.

        Args:
            tokenizer: Any HuggingFace tokenizer.
            vocab_size: Optional user-provided vocab size. If not provided, the
                tokenizer's vocab size is used.
            float_level: Whether to calculate the loss for every float or every
                integer in the sequence. For `12.34`, if float_level=False, two
                loss terms will be calculated, respectively for `12` and `34`.
                If float_level=True, a single `.` does not break the contiguity
                of the identified number. Defaults to False.
            reweigh: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to True.
                NOTE: The ICML paper does *not* use this option which can lead to
                incorrect loss if most mass is placed outside of the number tokens.
                Using this will explode the NL-NTL in the current implementation,
                so reweighing for the NL-NTL needs to be refined.
            max_number_length: Maximum expected length of a number in tokens.
                Used for precomputing power masks. Defaults to 20.

        """
        # digit_level must be set to True.
        super().__init__(
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            digit_level=True,
            reweigh=reweigh,
            loss_function=F.l1_loss,  # unused
        )
        self.float_level = float_level
        dot_result = self.tokenizer.convert_tokens_to_ids(".")
        # Ensure we get an int, not a list
        self.dot: int = dot_result if isinstance(dot_result, int) else dot_result[0]

        # Precompute powers of 10 for efficiency
        self.max_number_length = max_number_length
        self.powers_of_10 = torch.pow(
            10.0, torch.arange(max_number_length, dtype=torch.float32)
        )

    def setup_max_dist(self):
        """
        Due to the MAPE loss calculation, the max dist is limited to 1.0
        """
        self.max_dist = torch.tensor(1.0)

    def convert_digits_to_numbers(
        self,
        y: FloatTensor,
        yhat: FloatTensor,
        number_token_positions: BoolTensor,
        labels: LongTensor,
    ):
        """
        Vectorized conversion of digit-level number tokens to number-level values.

        Output convention:
        - Only the *first digit* of each detected number span contains the full number.
        - All other digits (and in float_level=True also the dot token) inside the span
            are set to NaN and removed from number_token_positions.
        - float_level=False: '.' breaks number spans (12.34 -> "12" and "34")
        - float_level=True : a single '.' between digits is part of the span but contributes 0
                            (12.34 -> "1234" as integer-like concatenation)

        Args:
            y: (B, T) float, GT digit values at digit positions, NaN elsewhere
            yhat: (B, T) float, predicted digit values at all positions
            number_token_positions: (B, T) bool, True at digit positions
            labels: (B, T) long, token ids

        Returns:
            (y_new, yhat_new, number_token_positions_new) at number-level
        """
        B, T = y.shape
        device = y.device
        out_dtype = yhat.dtype

        is_digit = number_token_positions  # (B, T)
        if not is_digit.any():
            return y, yhat, number_token_positions

        # -------------------------------------------------------------------------
        # 1) Decide which tokens are considered "inside a number span"
        # -------------------------------------------------------------------------
        # Base: digits are always in spans
        in_number: BoolTensor = is_digit

        if self.float_level:
            is_dot = labels.eq(self.dot)  # (B, T)

            # dot is part of a number span only if it is *between* digits: d . d
            digit_prev = torch.zeros((B, T), dtype=torch.bool, device=device)
            digit_prev[:, 1:] = is_digit[:, :-1]

            digit_next = torch.zeros((B, T), dtype=torch.bool, device=device)
            digit_next[:, :-1] = is_digit[:, 1:]

            dot_between_digits = is_dot & digit_prev & digit_next

            # In float mode, those dots count as "in number" (but contribute 0 later)
            in_number = cast(BoolTensor, in_number | dot_between_digits)
        else:
            dot_between_digits = torch.zeros((B, T), dtype=torch.bool, device=device)

        # -------------------------------------------------------------------------
        # 2) Build a "continuation" mask: does position t continue a span from t-1?
        # -------------------------------------------------------------------------
        # If previous token is a digit => continuation for digits (and for dot-between-digits in float mode)
        digit_prev = torch.zeros((B, T), dtype=torch.bool, device=device)
        digit_prev[:, 1:] = is_digit[:, :-1]

        if self.float_level:
            is_dot = labels.eq(self.dot)

            dot_prev = torch.zeros((B, T), dtype=torch.bool, device=device)
            dot_prev[:, 1:] = is_dot[:, :-1]

            digit_prev2 = torch.zeros((B, T), dtype=torch.bool, device=device)
            if T > 2:
                digit_prev2[:, 2:] = is_digit[:, :-2]

            # Continue if:
            #  - previous is digit, OR
            #  - previous is dot and the token before that is digit (digit . digit)
            continues_span = digit_prev | (dot_prev & digit_prev2)
        else:
            continues_span = digit_prev

        # A span starts wherever we're "in_number" but not continuing a previous span
        span_start = in_number & ~continues_span

        # -------------------------------------------------------------------------
        # 3) Assign each in-number token a segment id (per batch element)
        # -------------------------------------------------------------------------
        # seg_id is 0 for non-number tokens, otherwise 1..K within each row
        seg_id = torch.cumsum(span_start.to(torch.int32), dim=1)
        seg_id = seg_id * in_number.to(torch.int32)  # zero out non-number tokens

        # How many segments max per row? Needed for a stable "global segment id"
        segs_per_row = seg_id.max(dim=1).values  # (B,)
        max_segs = int(segs_per_row.max().item())
        if max_segs == 0:
            return y, yhat, number_token_positions

        # Make segment ids unique across batch:
        # global_seg = b * (max_segs + 1) + seg_id
        stride = max_segs + 1
        batch_base = (torch.arange(B, device=device, dtype=torch.int64) * stride).view(
            B, 1
        )
        global_seg = batch_base + seg_id.to(torch.int64)  # (B, T), 0 means "no segment"

        # -------------------------------------------------------------------------
        # 4) Compute per-digit position-from-right within each segment
        # -------------------------------------------------------------------------
        # We only count digits (dots do not affect digit positions).
        # pos_from_right: rightmost digit has 1, next has 2, etc.
        digit_count_rev = torch.cumsum(
            torch.flip(is_digit.to(torch.int32), dims=[1]), dim=1
        )
        pos_from_right = torch.flip(
            digit_count_rev, dims=[1]
        )  # (B, T), 0 for non-digits

        # Exponent for base-10 weighting: rightmost digit exponent 0, next exponent 1, ...
        exponent = (pos_from_right - 1).clamp_min(0).to(torch.int64)

        # Keep exponents within our precomputed range (or assert if you prefer strict behavior)
        exponent = exponent.clamp_max(self.max_number_length - 1)

        pow10 = self.powers_of_10.to(device=device, dtype=out_dtype)  # (L,)
        scale = pow10[exponent]  # (B, T)

        # -------------------------------------------------------------------------
        # 5) Compute digit contributions and sum per segment via scatter_add
        # -------------------------------------------------------------------------
        # Only digits contribute; dots/non-number contribute 0.
        zeros = torch.zeros((), device=device, dtype=out_dtype)

        y_contrib = torch.where(is_digit, y.to(out_dtype) * scale, zeros)
        yhat_contrib = torch.where(is_digit, yhat * scale, zeros)

        total_segments = B * stride  # includes index 0 per row
        seg_sum_y = torch.zeros((total_segments,), device=device, dtype=out_dtype)
        seg_sum_yhat = torch.zeros((total_segments,), device=device, dtype=out_dtype)

        flat_seg = global_seg.view(-1)
        seg_sum_y.scatter_add_(0, flat_seg, y_contrib.view(-1))
        seg_sum_yhat.scatter_add_(0, flat_seg, yhat_contrib.view(-1))

        # -------------------------------------------------------------------------
        # 6) Write segment sums back only at the *first digit* position of each span
        # -------------------------------------------------------------------------
        # Important: if float_level=True, span_start could be a dot (but we want first *digit*).
        number_start = span_start & is_digit  # (B, T)

        y_new = y.clone()
        yhat_new = yhat.clone()

        # Everything inside a span but not the start digit becomes NaN (incl. dots, other digits)
        in_span_not_start = in_number & ~number_start
        y_new = y_new.masked_fill(in_span_not_start, float("nan"))
        yhat_new = yhat_new.masked_fill(in_span_not_start, float("nan"))

        # Fill starts with summed values
        start_seg = global_seg[number_start]  # (N,)
        y_new[number_start] = seg_sum_y[start_seg]
        yhat_new[number_start] = seg_sum_yhat[start_seg]

        # Mask now indicates number-level positions (one per number span)
        number_token_positions_new = number_start

        return y_new, yhat_new, number_token_positions_new

    def forward(
        self,
        logits: FloatTensor,
        labels: LongTensor,
        loss_weights: Optional[Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> Tensor:
        """
        Computes the NTL based on the dot product between token values and their probs.

        Args:
            logits: 3D Tensor of shape BS x T x V.
            labels: 2D Tensor of shape BS x T.
            loss_weights: 2D Optional tensor of BS x T with token-wise loss weights.
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".
            ignore_index: The token ID to ignore in the labels. Defaults to -100.

        Returns:
            Loss tensor
                0-D if reduction=="mean"|"sum"
                BS x T if reduction=="none"
        """
        self._validate_inputs(logits, labels, loss_weights)

        y, _ = self._prepare_number_token_targets(labels, loss_weights, ignore_index)
        number_token_positions = cast(BoolTensor, ~torch.isnan(y))

        # If no digit tokens in batch, or total of the relevant loss weights is zero, no need for upcoming calculations
        if not number_token_positions.any() or (
            loss_weights is not None and not loss_weights.any()
        ):
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=logits.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(labels, dtype=logits.dtype)
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        yhat = self._get_dot_product(logits=logits)

        y, yhat, number_token_positions = self.convert_digits_to_numbers(
            y, yhat, number_token_positions, labels
        )
        if loss_weights is None:
            loss_weights = torch.ones_like(labels, dtype=logits.dtype)
        loss_weights = loss_weights[number_token_positions]

        # NOTE: Alternative could be to apply specified loss function to normalized yhat
        # loss = self.loss_function(torch.div(
        #     yhat[number_token_positions],
        #     y[number_token_positions].clamp_min(torch.finfo(y.dtype).eps),
        # ), torch.ones_like(yhat), reduction="none")

        y_num = y[number_token_positions]
        yh_num = yhat[number_token_positions]
        # Calculate symmetric MAPE which is bounded in [0, 1]
        loss = (yh_num - y_num).abs() / (
            yh_num.abs() + y_num.abs() + torch.finfo(y.dtype).eps
        )

        # If reweigh: compute weights for NTL based on logits
        if self.reweigh:
            loss = self.reweigh_fn(
                logits=logits, loss=loss, number_token_positions=number_token_positions
            )

        loss = self._apply_reduction(
            loss=loss,
            reduction=reduction,
            loss_weights=loss_weights,
            number_token_positions=number_token_positions,
            logits=logits,
        )

        return loss

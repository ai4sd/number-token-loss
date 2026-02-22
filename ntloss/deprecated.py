from typing import Optional, cast

import torch
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from transformers import PreTrainedTokenizer

from .core import NTLossDotProduct


class NumberLevelLossLooped(NTLossDotProduct):
    """Class to calculate NTL on a per-number (rather than per-token) basis.

    This is the original implementation using Python loops.
    Kept for backward compatibility and testing.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        vocab_size: Optional[int] = None,
        float_level: bool = False,
        reweigh: bool = True,
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
        self.dot = self.tokenizer.convert_tokens_to_ids(".")

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
        Set up the order mask for the batch and convert digit-level number tokens to numerical values.

        Args:
            y: 2D FloatTensor of shape BS x T with target numerical values at digit-level (NaN for non-number tokens).
            yhat: 2D FloatTensor of shape BS x T containing the predictions for the number tokens at digit-level
                (includes predictions for non-number tokens).
            number_token_positions: 2D BoolTensor (BS x T) containing locations of number tokens at digit-level.
            labels: 2D LongTensor of shape BS x T with the target input IDs.

        Returns:
            y: 2D FloatTensor of shape BS x T with target numerical values at number-level (NaN for non-number tokens).
            yhat: 2D FloatTensor of shape BS x T containing the predictions for the number tokens at number-level
                (includes predictions for non-number tokens).
            number_token_positions: 2D BoolTensor (BS x T) containing locations of numerical values in y and yhat.
        """

        # Set up empty order_mask: will store power with which to scale digits
        order_mask = torch.zeros_like(y, dtype=yhat.dtype, device=y.device)

        # Extract numbers using number blocks
        for i in range(y.shape[0]):
            # For every item in batch: assume not starting with number block
            in_number_block = False
            end_digit = -1

            # Loop from end of sequence to beginning to extract numbers
            for j in range(y.shape[1] - 1, -1, -1):
                # Already in number block and a digit: increase order magnitude
                if in_number_block and number_token_positions[i, j]:
                    if not self.float_level or labels[i, j + 1] != self.dot:
                        previous_order_index = j + 1
                    else:
                        previous_order_index = j + 2
                    order_mask[i, j] = order_mask[i, previous_order_index] + 1

                # Not in number block: first instance of number = end digit
                elif number_token_positions[i, j]:
                    in_number_block = True
                    end_digit = j + 1

                # A dot can be considered part of a number if self.float_level
                elif (
                    in_number_block
                    and self.float_level
                    and labels[i, j] == self.dot
                    and labels[i, j + 1] != self.dot
                ):
                    # exp(-inf) = 0, thus, the dot does not contribute to the GT number calculation
                    order_mask[i, j] = -torch.inf
                    # Necessary to avoid having NaN when summing
                    y[i, j] = 0
                    yhat[i, j] = 0

                # In number block, but not a digit: end of number_block
                elif in_number_block:
                    in_number_block = False

                    # Reuse y and yhat tensors to store full numbers
                    y[i, j + 1] = torch.sum(
                        y[i, j + 1 : end_digit]
                        * torch.pow(10, order_mask[i, j + 1 : end_digit])
                    )
                    # Make sure non-relevant numerical values are turned into NaN
                    # This indicates non-number tokens
                    y[i, j + 2 : end_digit] = y[i, j]
                    yhat[i, j + 1] = torch.sum(
                        yhat[i, j + 1 : end_digit]
                        * torch.pow(10, order_mask[i, j + 1 : end_digit])
                    )

        # Update mask with locations of number tokens
        number_token_positions = cast(BoolTensor, ~torch.isnan(y))

        return y, yhat, number_token_positions

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

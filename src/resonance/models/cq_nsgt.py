import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolution block for encoding and decoding complex vector features
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, mask=None):
        # Clear padded values before and after every layer so they cannot leak back into the real part of a shorter CQ-NSGT block
        if mask is not None:
            x = x * mask

        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask

        return x


# Upsampling block for decoding and passing encoder skip connections
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(
        self,
        x,
        skip,
        mask=None,
        input_lengths=None,
        target_lengths=None,
        resize_plan=None,
    ):
        # A grouped batch can contain blocks with different real lengths.
        # Resize each one using its own lengths instead of the padded width.
        if resize_plan is not None:
            x = apply_time_resize(x, resize_plan)
        elif input_lengths is None or target_lengths is None:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        else:
            x = resize_time(
                x,
                input_lengths,
                target_lengths,
                skip.shape[-1],
            )

        # Join the decoder and encoder features before the convolution block
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, mask)


def create_time_mask(lengths, frequency_size, padded_length, dtype):
    positions = torch.arange(
        padded_length,
        device=lengths.device,
    )
    mask = positions[None, :] < lengths[:, None]
    mask = mask[:, None, None, :].expand(
        -1,
        1,
        frequency_size,
        -1,
    )
    return mask.to(dtype=dtype)


def resize_time(x, input_lengths, target_lengths, padded_target_length):
    plan = prepare_time_resize(
        input_lengths,
        target_lengths,
        padded_target_length,
        x.dtype,
    )
    return apply_time_resize(x, plan)


def prepare_time_resize(input_lengths, target_lengths, padded_target_length, dtype):
    # Recreate align_corners=False interpolation coordinates independently for every real block length inside the padded group
    target_positions = torch.arange(
        padded_target_length,
        device=input_lengths.device,
        dtype=dtype,
    )[None, :]
    input_lengths_float = input_lengths[:, None].to(dtype=dtype)
    target_lengths_float = target_lengths[:, None].to(dtype=dtype)

    source_positions = (
        (target_positions + 0.5)
        * input_lengths_float
        / target_lengths_float
        - 0.5
    )

    # PyTorch interpolation repeats the first and last valid values at its half-pixel boundary, clamp to each block rather than the padded tensor
    source_positions = torch.maximum(
        source_positions,
        torch.zeros_like(source_positions),
    )
    source_positions = torch.minimum(
        source_positions,
        input_lengths_float - 1,
    )

    # Gather the samples on either side of every interpolation position
    # This avoids grid_sample, whose backward operation is unavailable on MPS
    lower_positions = torch.floor(source_positions).to(dtype=torch.long)
    upper_positions = torch.ceil(source_positions).to(dtype=torch.long)
    interpolation_weight = (
        source_positions - lower_positions.to(dtype=dtype)
    )[:, None, None, :]
    target_mask = create_time_mask(
        target_lengths,
        1,
        padded_target_length,
        dtype,
    )
    return lower_positions, upper_positions, interpolation_weight, target_mask


def apply_time_resize(x, plan):
    # The gather coordinates depend only on the layout, never on model weights
    lower_positions, upper_positions, interpolation_weight, target_mask = plan
    gather_shape = (
        -1,
        x.shape[1],
        x.shape[2],
        -1,
    )
    lower_indices = lower_positions[:, None, None, :].expand(gather_shape)
    upper_indices = upper_positions[:, None, None, :].expand(gather_shape)
    lower_values = torch.gather(x, -1, lower_indices)
    upper_values = torch.gather(x, -1, upper_indices)

    resized = (
        lower_values * (1 - interpolation_weight)
        + upper_values * interpolation_weight
    )
    return resized * target_mask


class CQNSGTTransitionModel(nn.Module):
    def __init__(
        self,
        action_vector_size,
        action_embedding_size=32,
        *,
        cache_layouts=True,
    ):
        super().__init__()

        # Encoder block 1: two complex vector channels -> 32 feature channels
        self.e1 = ConvBlock(2, 32)
        self.pool1 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 2),
        )

        # Encoder block 2: 32 -> 64 feature channels
        self.e2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 2),
        )

        # Encode the requested audio action into a smaller feature vector
        self.action_encoder = nn.Sequential(
            nn.Linear(action_vector_size, 64),
            nn.GELU(),
            nn.Linear(64, action_embedding_size),
            nn.GELU(),
        )

        # Combine the encoded CQ-NSGT block with the action embedding
        self.bottleneck = ConvBlock(64 + action_embedding_size, 128)

        # Decoder block 1: 128 bottleneck channels -> 64 feature channels
        self.d2 = UpBlock(128, 64, 64)

        # Decoder block 2: 64 feature channels -> 32 feature channels
        self.d1 = UpBlock(64, 32, 32)

        # Predict changes to the real and imaginary vector channels
        self.output = nn.Conv2d(32, 2, kernel_size=1)

        # Similar time lengths share one padded model call. A ratio of 1.5
        # reduces 112 calls to about 23 without adding too much padding.
        self.block_bucket_ratio = 1.5

        # Reuse fixed masks and interpolation indices across training batches.
        # These tensors are derived data and must not enter model checkpoints.
        self._layout_cache = {}
        self.cache_layouts = cache_layouts

    def _apply(self, fn, recurse=True):
        # Moving or casting a model should release its old device-side layouts
        self._layout_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def _block_layout(self, lengths, frequency_size, padded_length, vectors):
        key = (
            tuple(lengths), frequency_size, padded_length,
            vectors.device, vectors.dtype,
        )
        if key not in self._layout_cache:
            # Bound memory when callers use many different clip/batch shapes
            if len(self._layout_cache) >= 64:
                self._layout_cache.clear()
            input_lengths = torch.tensor(
                lengths, device=vectors.device, dtype=torch.long,
            )
            pooled_lengths_1 = input_lengths // 2
            pooled_lengths_2 = pooled_lengths_1 // 2
            masks = tuple(
                create_time_mask(
                    current_lengths, frequency_size,
                    padded_length // divisor, vectors.dtype,
                )
                for current_lengths, divisor in (
                    (input_lengths, 1),
                    (pooled_lengths_1, 2),
                    (pooled_lengths_2, 4),
                )
            )
            self._layout_cache[key] = (
                masks,
                prepare_time_resize(
                    pooled_lengths_2, pooled_lengths_1,
                    padded_length // 2, vectors.dtype,
                ),
                prepare_time_resize(
                    pooled_lengths_1, input_lengths,
                    padded_length, vectors.dtype,
                ),
            )
        return self._layout_cache[key]

    def _forward_block(self, vectors, action_vector, lengths=None, layout=None):
        if vectors.ndim != 4 or vectors.shape[1] != 2:
            raise ValueError(
                'Expected vectors with shape [batch, 2, frequency, time].'
            )
        if vectors.shape[-1] < 4:
            raise ValueError(
                'CQ-NSGT vector blocks need at least 4 time positions.'
            )

        if lengths is None and layout is None:
            lengths = torch.full(
                (vectors.shape[0],),
                vectors.shape[-1],
                device=vectors.device,
                dtype=torch.long,
            )

        # Build masks for the original and both pooled time resolutions
        if layout is None:
            input_mask = create_time_mask(
                lengths, vectors.shape[-2], vectors.shape[-1], vectors.dtype,
            )
            pooled_lengths_1 = lengths // 2
            pooled_lengths_2 = pooled_lengths_1 // 2
            resize_plan_2 = resize_plan_1 = None
        else:
            masks, resize_plan_2, resize_plan_1 = layout
            input_mask, pooled_mask_1, pooled_mask_2 = masks
            pooled_lengths_1 = pooled_lengths_2 = None

        # Pass the real and imaginary vector channels through both encoders
        skip1 = self.e1(vectors, input_mask)
        h = self.pool1(skip1)
        if layout is None:
            pooled_mask_1 = create_time_mask(
                pooled_lengths_1, h.shape[-2], h.shape[-1], h.dtype,
            )
        h = h * pooled_mask_1

        skip2 = self.e2(h, pooled_mask_1)
        h = self.pool2(skip2)
        if layout is None:
            pooled_mask_2 = create_time_mask(
                pooled_lengths_2, h.shape[-2], h.shape[-1], h.dtype,
            )
        h = h * pooled_mask_2

        # Turn the action vector into an embedding at every encoded position
        action_embedding = self.action_encoder(action_vector)
        action_embedding = action_embedding[:, :, None, None]
        action_embedding = action_embedding.expand(
            -1,
            -1,
            h.shape[-2],
            h.shape[-1],
        )

        # Add the action information to the encoded CQ-NSGT features
        h = torch.cat([h, action_embedding], dim=1)
        h = self.bottleneck(h, pooled_mask_2)

        # Decode the features while restoring both encoder skip connections
        h = self.d2(
            h,
            skip2,
            mask=pooled_mask_1,
            input_lengths=pooled_lengths_2,
            target_lengths=pooled_lengths_1,
            resize_plan=resize_plan_2,
        )
        h = self.d1(
            h,
            skip1,
            mask=input_mask,
            input_lengths=pooled_lengths_1,
            target_lengths=lengths,
            resize_plan=resize_plan_1,
        )

        # Predict the change to each real and imaginary vector value
        predicted_delta = self.output(h) * input_mask

        # Rebuild the next audio state by adding the predicted vector change
        return (vectors + predicted_delta) * input_mask

    def _block_group_key(self, vectors):
        time_length = vectors.shape[-1]
        time_bucket = math.ceil(
            math.log(time_length, self.block_bucket_ratio)
        )
        return vectors.shape[-2], time_bucket

    def _group_blocks(self, vector_blocks):
        groups = {}

        # Blocks are grouped by frequency height and similar time length
        for block_index, vectors in enumerate(vector_blocks):
            key = self._block_group_key(vectors)
            groups.setdefault(key, []).append(block_index)

        return groups

    def _forward_group(self, vector_blocks, block_indices, action_vector):
        batch_size = action_vector.shape[0]
        padded_length = max(
            vector_blocks[index].shape[-1]
            for index in block_indices
        )
        padded_blocks = []
        grouped_lengths = []

        for block_index in block_indices:
            vectors = vector_blocks[block_index]
            time_length = vectors.shape[-1]

            # Right pad each block to the longest member of this group
            padded_blocks.append(
                F.pad(vectors, (0, padded_length - time_length))
            )
            grouped_lengths.extend([time_length] * batch_size)

        # Treat each grouped block as another independent batch entry
        grouped_vectors = torch.cat(padded_blocks, dim=0)
        grouped_actions = action_vector.repeat(len(block_indices), 1)
        if (
            not self.cache_layouts
            or torch.is_autocast_enabled(grouped_vectors.device.type)
        ):
            # Autocast may change activation dtypes within a block. Retain the
            # dynamic path so each mask/resize uses the actual activation dtype.
            layout = None
            lengths = torch.tensor(
                grouped_lengths, device=grouped_vectors.device,
                dtype=torch.long,
            )
        else:
            layout = self._block_layout(
                grouped_lengths, grouped_vectors.shape[-2],
                padded_length, grouped_vectors,
            )
            lengths = None
        grouped_predictions = self._forward_block(
            grouped_vectors,
            grouped_actions,
            lengths=lengths,
            layout=layout,
        )

        # Split and crop the results back into the original CQ-NSGT shapes
        predictions = []
        for group_index, block_index in enumerate(block_indices):
            start = group_index * batch_size
            end = start + batch_size
            time_length = vector_blocks[block_index].shape[-1]
            predictions.append(
                grouped_predictions[start:end, :, :, :time_length]
            )

        return predictions

    def _forward_blocks_separately(self, vector_blocks, action_vector):
        # Kept as a simple reference path for checking grouped predictions
        return [
            self._forward_block(vectors, action_vector)
            for vectors in vector_blocks
        ]

    def forward(self, vector_blocks, action_vector):
        # CQ-NSGT uses different time resolutions for different frequency bands.
        # The codec therefore returns a list of blocks instead of one spectrogram.
        if not isinstance(vector_blocks, list):
            raise ValueError('Expected a list of CQ-NSGT vector blocks.')

        predicted_blocks = [None] * len(vector_blocks)
        groups = self._group_blocks(vector_blocks)

        # Process every length bucket in one model call, then restore block order
        for block_indices in groups.values():
            group_predictions = self._forward_group(
                vector_blocks,
                block_indices,
                action_vector,
            )
            for block_index, prediction in zip(
                block_indices,
                group_predictions,
            ):
                predicted_blocks[block_index] = prediction

        # Keep the list structure so the codec can invert it back into audio
        return predicted_blocks

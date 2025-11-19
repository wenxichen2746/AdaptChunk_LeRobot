#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.types import RTCAttentionSchedule

from lerobot.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig, SmolVLA_CFG_Config
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self._last_action_chunk: Tensor | None = None
        self._last_model_action_chunk: Tensor | None = None
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._last_action_chunk = None
        self._last_model_action_chunk = None

    def init_rtc_processor(self):
        self.rtc_processor = None

        if self.config.rtc_config is not None and self.config.rtc_config.enabled:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            if self.model is not None:
                self.model.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    # def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
    #     # TODO: Check if this for loop is needed.
    #     # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
    #     # In the case of offline inference, we have the action in the batch
    #     # that why without the k != ACTION check, it will raise an error because we are trying to stack
    #     # on an empty container.
    #     for k in batch:
    #         if k in self._queues and k != ACTION:
    #             batch[k] = torch.stack(list(self._queues[k]), dim=1)

    #     images, img_masks = self.prepare_images(batch)
    #     state = self.prepare_state(batch)
    #     lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    #     lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

    #     actions = self.model.sample_actions(
    #         images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
    #     )

    #     # Unpad actions
    #     original_action_dim = self.config.action_feature.shape[0]
    #     actions = actions[:, :, :original_action_dim]

    #     if self.config.adapt_to_pi_aloha:
    #         actions = self._pi_aloha_encode_actions(actions)

    #     return actions
    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        decoding_strategy = getattr(self.config, "decoding_strategy", "naive")
        config_decoding_kwargs = dict(getattr(self.config, "decoding_kwargs", {}) or {})
        runtime_kwargs = dict(kwargs)
        prev_model_chunk = self._last_model_action_chunk

        if decoding_strategy == "bid":
            call_kwargs = {**config_decoding_kwargs, **runtime_kwargs}
            call_kwargs["prev_action_chunk"] = prev_model_chunk
            actions_full = self.model.sample_actions_bid(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise=noise,
                **call_kwargs,
            )
        elif decoding_strategy == "rtc":
            call_kwargs = {**config_decoding_kwargs, **runtime_kwargs}
            call_kwargs["prev_action_chunk"] = prev_model_chunk
            with torch.enable_grad():
                actions_full = self.model.sample_actions_rtc(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    noise=noise,
                    **call_kwargs,
                )
            actions_full = actions_full.detach()
        elif decoding_strategy in {"naive", "default"}:
            call_kwargs = {**config_decoding_kwargs, **runtime_kwargs}
            actions_full = self.model.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise=noise,
                **call_kwargs,
            )
        else:
            raise ValueError(
                f"Decoding strategy '{decoding_strategy}' is not supported by SmolVLAPolicy. "
                "Use SmolVLACFGPolicy for CFG or history-aware decoding."
            )

        self._last_model_action_chunk = actions_full.detach()

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions_full[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        self._last_action_chunk = actions
        return actions
    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    def _decoding_requires_grad(self) -> bool:
        strategy = getattr(self.config, "decoding_strategy", "naive")
        return strategy == "rtc"

    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        grad_ctx = torch.enable_grad() if self._decoding_requires_grad() else torch.no_grad()
        with grad_ctx:
            actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            grad_ctx = torch.enable_grad() if self._decoding_requires_grad() else torch.no_grad()
            with grad_ctx:
                actions = self._get_action_chunk(batch, noise, **kwargs)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            actions_is_pad = actions_is_pad.to(device=losses.device)
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


class SmolVLACFGPolicy(SmolVLAPolicy):
    """SmolVLA variant that conditions on expert action history."""

    config_class = SmolVLA_CFG_Config
    name = "smolvla_cfg"

    def __init__(self, config: SmolVLA_CFG_Config):
        # Ensure attributes exist before parent initialization hooks run.
        self._history_buffer = None
        self._last_action_chunk: Tensor | None = None
        self._last_model_action_chunk: Tensor | None = None
        self.model = None
        super().__init__(config)
        # Replace base model with CFG variant while reusing the RTC processor.
        self.model = VLAFlowMatching_CFG(config, rtc_processor=self.rtc_processor)
        # Reinitialize buffers so they align with the CFG model setup.
        self.reset()

    def reset(self):
        super().reset()
        self._init_history_buffer()
        self._last_action_chunk = None
        self._last_model_action_chunk = None

    def _init_history_buffer(self):
        history_steps = self.config.history_action_steps
        if history_steps > 0:
            action_dim = self.config.action_feature.shape[0]
            zero = torch.zeros(action_dim)
            self._history_buffer = deque(
                [(zero.clone(), False) for _ in range(history_steps)],
                maxlen=history_steps,
            )
        else:
            self._history_buffer = None

    def _split_action_history(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor | None]:
        history_steps = self.config.history_action_steps
        actions = batch[ACTION]
        if history_steps == 0:
            return None, None, self.prepare_action(batch), batch.get("action_is_pad")

        total_steps = actions.shape[1]
        if total_steps < history_steps + self.config.chunk_size:
            raise ValueError(
                f"Expected at least {history_steps + self.config.chunk_size} action steps, got {total_steps}."
            )

        history = actions[:, :history_steps, :]
        future = actions[:, history_steps : history_steps + self.config.chunk_size, :]

        history = pad_vector(history, self.config.max_action_dim)
        future = pad_vector(future, self.config.max_action_dim)

        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is None:
            action_is_pad = batch.get(f"{ACTION}_is_pad")
        if action_is_pad is None:
            action_is_pad = batch.get("actions_id_pad")

        history_mask = None
        future_is_pad = None
        if action_is_pad is not None:
            if action_is_pad.shape[1] < history_steps + self.config.chunk_size:
                raise ValueError("Padding mask length does not match action sequence length.")
            action_is_pad = action_is_pad.to(dtype=torch.bool)
            history_mask = ~action_is_pad[:, :history_steps]
            future_is_pad = action_is_pad[:, history_steps : history_steps + self.config.chunk_size]
        return history, history_mask, future, future_is_pad

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        action_history, action_history_mask, actions, actions_is_pad = self._split_action_history(batch)

        if action_history is not None:
            action_history = action_history.to(device=state.device, dtype=state.dtype)
            if action_history_mask is not None:
                action_history_mask = action_history_mask.to(device=state.device)
            if (
                self.training
                and getattr(self.config, "history_action_noise_std", 0.0) > 0.0
            ):
                history_noise = torch.randn_like(action_history) * self.config.history_action_noise_std
                if action_history_mask is not None:
                    valid_mask = action_history_mask.unsqueeze(-1).to(dtype=action_history.dtype)
                    history_noise = history_noise * valid_mask
                action_history = action_history + history_noise

        actions = actions.to(device=state.device, dtype=state.dtype)
        drop_action_history_mask = None
        drop_obs_mask = None
        if self.training:
            if (
                self.config.history_action_steps > 0
                and action_history is not None
                and self.config.drop_pastaction_prob > 0.0
            ):
                drop_action_history_mask = (
                    torch.rand(state.shape[0], device=state.device) < self.config.drop_pastaction_prob
                )
                if not drop_action_history_mask.any():
                    drop_action_history_mask = None
            if self.config.drop_obs_prob > 0.0:
                drop_obs_mask = torch.rand(state.shape[0], device=state.device) < self.config.drop_obs_prob
                if not drop_obs_mask.any():
                    drop_obs_mask = None
        loss_dict = {}
        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            action_history=action_history,
            action_history_mask=action_history_mask,
            noise=noise,
            time=time,
            drop_action_history_mask=drop_action_history_mask,
            drop_obs_mask=drop_obs_mask,
        )
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def _gather_inference_history(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor | None, Tensor | None]:
        if self._history_buffer is None or len(self._history_buffer) == 0:
            return None, None
        history_tensors: list[Tensor] = []
        for tensor, _ in self._history_buffer:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] != self.config.max_action_dim:
                tensor = pad_vector(tensor, self.config.max_action_dim)
            if tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.expand(batch_size, -1).contiguous()
            elif tensor.shape[0] != batch_size:
                raise ValueError(
                    f"History buffer batch size {tensor.shape[0]} does not match current batch size {batch_size}."
                )
            history_tensors.append(tensor.to(device=device, dtype=dtype))

        history = torch.stack(history_tensors, dim=1)  # (batch, hist, action_dim)

        history_valid = torch.tensor(
            [entry[1] for entry in self._history_buffer],
            dtype=torch.bool,
            device=device,
        )
        history_mask = history_valid.unsqueeze(0).expand(batch_size, -1)

        return history, history_mask

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        action_history, action_history_mask = self._gather_inference_history(
            state.shape[0], state.device, state.dtype
        )

        decoding_strategy = getattr(self.config, "decoding_strategy", "naive")
        config_decoding_kwargs = dict(getattr(self.config, "decoding_kwargs", {}) or {})
        runtime_kwargs = dict(kwargs)
        prev_model_chunk = self._last_model_action_chunk

        if decoding_strategy == "cfg":
            call_kwargs = {**config_decoding_kwargs, **runtime_kwargs}
            actions_full = self.model.sample_actions_cfg(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                action_history=action_history,
                action_history_mask=action_history_mask,
                noise=noise,
                **call_kwargs,
            )
        elif decoding_strategy == "bid":
            call_kwargs = {**config_decoding_kwargs, **runtime_kwargs}
            call_kwargs["prev_action_chunk"] = prev_model_chunk
            actions_full = self.model.sample_actions_bid(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                action_history=action_history,
                action_history_mask=action_history_mask,
                noise=noise,
                **call_kwargs,
            )
        elif decoding_strategy == "rtc":
            call_kwargs = {**config_decoding_kwargs, **runtime_kwargs}
            call_kwargs["prev_action_chunk"] = prev_model_chunk
            with torch.enable_grad():
                actions_full = self.model.sample_actions_rtc(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    action_history=action_history,
                    action_history_mask=action_history_mask,
                    noise=noise,
                    **call_kwargs,
                )
            actions_full = actions_full.detach()
        elif decoding_strategy == "naive_nulla":
            actions_full = self.model.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                action_history=action_history,
                action_history_mask=action_history_mask,
                null_action=True,
                noise=noise,
                **runtime_kwargs,
            )
        else:
            actions_full = self.model.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                action_history=action_history,
                action_history_mask=action_history_mask,
                noise=noise,
                **runtime_kwargs,
            )

        self._last_model_action_chunk = actions_full.detach()

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions_full[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        self._last_action_chunk = actions
        return actions

    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        action = super().select_action(batch, noise, **kwargs)
        self._update_history_buffer(action)
        return action

    def _update_history_buffer(self, action: Tensor) -> None:
        if self._history_buffer is None:
            return
        step_action = action.detach()

        if step_action.ndim == 1:
            step_action = step_action.unsqueeze(0)
        elif step_action.ndim != 2:
            raise ValueError(f"Unsupported action tensor shape {tuple(step_action.shape)}")

        if self.config.adapt_to_pi_aloha:
            step_action = self._pi_aloha_encode_actions_inv(step_action.unsqueeze(1))[:, 0, :]

        step_action = pad_vector(step_action, self.config.max_action_dim).cpu()
        self._history_buffer.append((step_action.clone(), True))

def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor



class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Partial call of the function, I used `partial` package at the beginning
            # but it was not working as expected, so I used a lambda function instead
            # I want to pass `x_t` as positionan argument to the partial call, because
            # it could have different naming in different models, and the rest of parameters
            # as named arguments
            denoise_step_partial_call = lambda input_x_t: self.denoise_step(  # noqa: E731
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=expanded_time,  # noqa: B023
            )

            if self.config.rtc_config is not None and self.config.rtc_config.enabled:
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon", self.config.rtc_config.execution_horizon)

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def sample_actions_bid(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        action_history=None,
        action_history_mask=None,
        *,
        prev_action_chunk: Tensor | None = None,
        inference_delay: int = 0,
        prefix_attention_horizon: int | None = None,
        n_samples: int = 4,
        prefix_attention_schedule: str | RTCAttentionSchedule = "exp",
        noise=None,
        **kwargs,
    ) -> Tensor:
        """Bidirectional Inference Decoding (BID) with overlap-aware backward loss."""
        del kwargs  # intentionally unused

        if n_samples <= 0:
            raise ValueError("`n_samples` must be a positive integer.")

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if prev_action_chunk is None or prev_action_chunk.shape[0] != bsize:
            return self.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                action_history=action_history,
                action_history_mask=action_history_mask,
                noise=noise,
            )

        prev_action_chunk = prev_action_chunk.to(device=device, dtype=dtype)

        overlap_start = min(self.config.n_action_steps, prev_action_chunk.shape[1])
        if overlap_start >= prev_action_chunk.shape[1]:
            return self.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                action_history=action_history,
                action_history_mask=action_history_mask,
                noise=noise,
            )

        prev_overlap_full = prev_action_chunk[:, overlap_start:, :]
        overlap_len = min(prev_overlap_full.shape[1], self.config.n_action_steps)
        if overlap_len <= 0:
            return self.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                action_history=action_history,
                action_history_mask=action_history_mask,
                noise=noise,
            )

        schedule_value = (
            prefix_attention_schedule.value
            if isinstance(prefix_attention_schedule, RTCAttentionSchedule)
            else str(prefix_attention_schedule)
        )
        prefix_attention_horizon = prefix_attention_horizon or overlap_len
        weights = self._compute_prefix_weights(
            inference_delay=inference_delay,
            horizon=prefix_attention_horizon,
            total=overlap_len,
            schedule=schedule_value,
            device=device,
            dtype=dtype,
        )

        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim),
                device,
            )

        if noise.shape[0] == bsize:
            noise_all = noise.repeat(n_samples, 1, 1)
        elif noise.shape[0] == n_samples * bsize:
            noise_all = noise
        else:
            raise ValueError(
                f"`noise` must have shape ({bsize}, C, A) or ({n_samples * bsize}, C, A); "
                f"got {tuple(noise.shape)}."
            )

        def _repeat_batch(tensor: Tensor | None) -> Tensor | None:
            if tensor is None:
                return None
            repeats = [n_samples] + [1] * (tensor.dim() - 1)
            return tensor.repeat(*repeats)

        images_rep = [_repeat_batch(img) for img in images]
        img_masks_rep = [_repeat_batch(mask) for mask in img_masks]
        state_rep = _repeat_batch(state)
        lang_tokens_rep = _repeat_batch(lang_tokens)
        lang_masks_rep = _repeat_batch(lang_masks)
        action_history_rep = _repeat_batch(action_history)
        action_history_mask_rep = _repeat_batch(action_history_mask)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images_rep,
            img_masks_rep,
            lang_tokens_rep,
            lang_masks_rep,
            state=state_rep,
            action_history=action_history_rep,
            action_history_mask=action_history_mask_rep,
            drop_obs_mask=None,
            drop_action_history_mask=None,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        time = torch.tensor(1.0, dtype=dtype, device=device)
        x_t = noise_all
        batch_ext = n_samples * bsize

        while time >= -dt / 2:
            expanded_time = time.expand(batch_ext)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=expanded_time,
            )
            x_t = x_t + dt * v_t
            time = time + dt

        strong_actions = x_t.view(
            n_samples, bsize, self.config.chunk_size, self.config.max_action_dim
        )
        prev_overlap = prev_overlap_full[:, :overlap_len, :]
        new_overlap = strong_actions[:, :, :overlap_len, :]

        diff = torch.linalg.norm(new_overlap - prev_overlap.unsqueeze(0), dim=-1)
        weighted_diff = diff * weights[None, None, :]
        loss = weighted_diff.sum(dim=-1)

        best_indices = torch.argmin(loss, dim=0)
        batch_indices = torch.arange(bsize, device=device)
        best_actions = strong_actions[best_indices, batch_indices]
        return best_actions

    def sample_actions_rtc(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        action_history=None,
        action_history_mask=None,
        *,
        prev_action_chunk: Tensor | None = None,
        inference_delay: int = 0,
        execution_horizon: int | None = None,
        prefix_attention_schedule: str | RTCAttentionSchedule | None = "exp",
        max_guidance_weight: float | None = None,
        noise=None,
        **kwargs,
    ) -> Tensor:
        """Real-Time Chunking decoding that mirrors the RTC processor behaviour."""
        del kwargs  # unused runtime kwargs reserved for future use

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim),
                device,
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            action_history=action_history,
            action_history_mask=action_history_mask,
            drop_obs_mask=None,
            drop_action_history_mask=None,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        rtc_cfg = self.config.rtc_config
        if rtc_cfg is None:
            rtc_cfg = RTCConfig()
            self.config.rtc_config = rtc_cfg

        if self.rtc_processor is None:
            self.rtc_processor = RTCProcessor(rtc_cfg)

        prev_chunk_left_over = None
        overlap_len = 0
        if prev_action_chunk is not None and prev_action_chunk.shape[0] == bsize:
            prev_action_chunk = prev_action_chunk.to(device=device, dtype=dtype)
            executed = min(self.config.n_action_steps, prev_action_chunk.shape[1])
            if executed < prev_action_chunk.shape[1]:
                leftover = prev_action_chunk[:, executed:, :]
                if leftover.shape[1] > 0:
                    overlap_len = min(leftover.shape[1], self.config.chunk_size)
                    prev_chunk_left_over = torch.zeros(
                        bsize,
                        self.config.chunk_size,
                        self.config.max_action_dim,
                        device=device,
                        dtype=dtype,
                    )
                    dim = min(leftover.shape[2], self.config.max_action_dim)
                    prev_chunk_left_over[:, :overlap_len, :dim] = leftover[:, :overlap_len, :dim]

        schedule_value = prefix_attention_schedule or rtc_cfg.prefix_attention_schedule
        schedule_enum = (
            schedule_value
            if isinstance(schedule_value, RTCAttentionSchedule)
            else RTCAttentionSchedule(schedule_value.upper())
        )

        execution_horizon_default = rtc_cfg.execution_horizon or self.config.n_action_steps
        execution_horizon_eff = execution_horizon or execution_horizon_default
        if overlap_len > 0:
            execution_horizon_eff = min(execution_horizon_eff, overlap_len)
        execution_horizon_eff = max(0, execution_horizon_eff)

        max_guidance_weight_eff = (
            max_guidance_weight if max_guidance_weight is not None else rtc_cfg.max_guidance_weight
        )

        # Temporarily override RTC config to reuse the standard sample_actions implementation.
        enabled_prev = rtc_cfg.enabled
        schedule_prev = rtc_cfg.prefix_attention_schedule
        execution_prev = rtc_cfg.execution_horizon
        weight_prev = rtc_cfg.max_guidance_weight

        rtc_cfg.enabled = True
        rtc_cfg.prefix_attention_schedule = schedule_enum
        rtc_cfg.execution_horizon = execution_horizon_eff
        rtc_cfg.max_guidance_weight = max_guidance_weight_eff

        try:
            actions = self.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                action_history=action_history,
                action_history_mask=action_history_mask,
                noise=noise,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=execution_horizon_eff,
            )
        finally:
            rtc_cfg.enabled = enabled_prev
            rtc_cfg.prefix_attention_schedule = schedule_prev
            rtc_cfg.execution_horizon = execution_prev
            rtc_cfg.max_guidance_weight = weight_prev

        return actions

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

class VLAFlowMatching_CFG(VLAFlowMatching):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__(config, rtc_processor=rtc_processor)
        null_token_dtype = torch.float32
        text_hidden = self.vlm_with_expert.config.text_config.hidden_size
        self.history_action_steps = getattr(self.config, "history_action_steps", 0)
        if self.history_action_steps > 0:
            self.history_action_proj = nn.Linear(self.config.max_action_dim, text_hidden)
            self.null_pastaction_tokens = nn.Parameter(
                torch.randn(
                    1,
                    self.history_action_steps,
                    text_hidden,
                    dtype=null_token_dtype,
                ) * 0.02
            )

        else:
            self.history_action_proj = None
            self.null_pastaction_tokens = None

        # Pre-create null observation tokens so they participate in the optimizer and have their gradients reset.
        state_hidden = self.state_proj.out_features
        self.null_observation_tokens = nn.ParameterDict({
            "vision": nn.Parameter(torch.randn(1, 1, text_hidden, dtype=null_token_dtype) * 0.02),
            "language": nn.Parameter(torch.randn(1, 1, text_hidden, dtype=null_token_dtype) * 0.02),
            "state": nn.Parameter(torch.randn(1, 1, state_hidden, dtype=null_token_dtype) * 0.02),
        })

    def _get_null_observation_token(self, key: str, hidden_size: int, dtype, device) -> torch.Tensor:
        if key not in self.null_observation_tokens:
            raise KeyError(f"Unknown null observation token key: {key}")
        param = self.null_observation_tokens[key]
        if param.shape[-1] != hidden_size:
            raise ValueError(f"Unexpected hidden size for null token '{key}': {param.shape[-1]} vs {hidden_size}")
        if param.dtype == dtype and param.device == device:
            return param
        return param.to(device=device, dtype=dtype)

    def set_requires_grad(self):
        super().set_requires_grad()

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        action_history: torch.Tensor | None = None,
        action_history_mask: torch.Tensor | None = None,
        drop_obs_mask: torch.BoolTensor | None = None,
        drop_action_history_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        drop_obs_mask_tensor = None
        if drop_obs_mask is not None:
            drop_obs_mask_tensor = drop_obs_mask.to(dtype=torch.bool, device=state.device if state is not None else None)
        drop_action_history_mask_tensor = None
        if drop_action_history_mask is not None:
            drop_action_history_mask_tensor = drop_action_history_mask.to(dtype=torch.bool, device=state.device if state is not None else None)
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = self.vlm_with_expert.embed_language_tokens(
                    self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                ).unsqueeze(0)
                if drop_obs_mask_tensor is not None:
                    null_tok = self._get_null_observation_token(
                        "language", image_start_token.shape[-1], image_start_token.dtype, image_start_token.device
                    )
                    image_start_token = torch.where(
                        drop_obs_mask_tensor[:, None, None],
                        null_tok.expand(img.shape[0], image_start_token.shape[1], -1),
                        image_start_token.expand(img.shape[0], -1, -1),
                    )
                else:
                    image_start_token = image_start_token.expand(img.shape[0], -1, -1)
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            if drop_obs_mask_tensor is not None:
                null_tok = self._get_null_observation_token("vision", img_emb.shape[-1], img_emb.dtype, img_emb.device)
                img_emb = torch.where(
                    drop_obs_mask_tensor[:, None, None],
                    null_tok.expand(bsize, num_img_embs, -1),
                    img_emb,
                )

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = self.vlm_with_expert.embed_language_tokens(
                    self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                ).unsqueeze(0)
                if drop_obs_mask_tensor is not None:
                    null_tok = self._get_null_observation_token(
                        "language", image_end_token.shape[-1], image_end_token.dtype, image_end_token.device
                    )
                    image_end_token = torch.where(
                        drop_obs_mask_tensor[:, None, None],
                        null_tok.expand(img.shape[0], image_end_token.shape[1], -1),
                        image_end_token.expand(img.shape[0], -1, -1),
                    )
                else:
                    image_end_token = image_end_token.expand(img.shape[0], -1, -1)
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        if drop_obs_mask_tensor is not None:
            null_tok = self._get_null_observation_token("language", lang_emb.shape[-1], lang_emb.dtype, lang_emb.device)
            lang_emb = torch.where(
                drop_obs_mask_tensor[:, None, None],
                null_tok.expand(lang_emb.shape[0], lang_emb.shape[1], -1),
                lang_emb,
            )

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        bsize = state_emb.shape[0]
        if drop_obs_mask_tensor is not None:
            null_tok = self._get_null_observation_token("state", state_emb.shape[-1], state_emb.dtype, state_emb.device)
            state_emb = torch.where(
                drop_obs_mask_tensor[:, None, None],
                null_tok.expand(bsize, state_emb.shape[1], -1),
                state_emb,
            )
        embs.append(state_emb)
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)

        if action_history is not None and self.history_action_proj is not None:
            history_emb = self.history_action_proj(action_history)
            # history_emb = history_emb * math.sqrt(history_emb.shape[-1])
            if drop_action_history_mask_tensor is not None and self.null_pastaction_tokens is not None:
                null_tok = self.null_pastaction_tokens.to(device=history_emb.device, dtype=history_emb.dtype)
                history_emb = torch.where(
                    drop_action_history_mask_tensor[:, None, None],
                    null_tok.expand(bsize, history_emb.shape[1], -1),
                    history_emb,
                )
            embs.append(history_emb)
            history_len = history_emb.shape[1]
            if action_history_mask is None:
                history_mask = torch.ones(bsize, history_len, dtype=torch.bool, device=history_emb.device)
            else:
                history_mask = action_history_mask.to(dtype=torch.bool, device=history_emb.device)
            pad_masks.append(history_mask)
            att_masks += [1] * history_len
            # att_masks += [0] * history_len

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        action_history=None,
        action_history_mask=None,
        noise=None,
        time=None,
        drop_action_history_mask: torch.BoolTensor | None = None,
        drop_obs_mask: torch.BoolTensor | None = None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            action_history=action_history,
            action_history_mask=action_history_mask,
            drop_obs_mask=drop_obs_mask,
            drop_action_history_mask=drop_action_history_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        action_history=None,
        action_history_mask=None,
        null_action: bool = False,
        null_obs: bool = False,
        noise=None,
        **kwargs,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        drop_obs_mask = None
        if null_obs:
            drop_obs_mask = torch.ones(bsize, dtype=torch.bool, device=device)
        drop_action_history_mask = None
        if action_history is not None and null_action:
            drop_action_history_mask = torch.ones(bsize, dtype=torch.bool, device=device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            action_history=action_history,
            action_history_mask=action_history_mask,
            drop_obs_mask=drop_obs_mask,
            drop_action_history_mask=drop_action_history_mask,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Partial call of the function, I used `partial` package at the beginning
            # but it was not working as expected, so I used a lambda function instead
            # I want to pass `x_t` as positionan argument to the partial call, because
            # it could have different naming in different models, and the rest of parameters
            # as named arguments
            denoise_step_partial_call = lambda input_x_t: self.denoise_step(  # noqa: E731
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=expanded_time,  # noqa: B023
            )

            # if self.config.rtc_config is not None and self.config.rtc_config.enabled:
            #     inference_delay = kwargs.get("inference_delay")
            #     prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
            #     execution_horizon = kwargs.get("execution_horizon", self.config.rtc_config.execution_horizon)

            #     v_t = self.rtc_processor.denoise_step(
            #         x_t=x_t,
            #         prev_chunk_left_over=prev_chunk_left_over,
            #         inference_delay=inference_delay,
            #         time=time,
            #         original_denoise_step_partial=denoise_step_partial_call,
            #         execution_horizon=execution_horizon,
            #     )
            # else:
            v_t = denoise_step_partial_call(x_t)

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def sample_actions_cfg(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        action_history=None,
        action_history_mask=None,
        noise=None,
        **kwargs,
    ) -> Tensor:
        """CFG-style decoding using independent nulls for observations and action history."""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Extract combination weights while leaving other kwargs untouched for forward compatibility.
        local_kwargs = dict(kwargs)
        w_ao = local_kwargs.pop("w_ao", 1.0)
        w_on = local_kwargs.pop("w_on", 0.0)
        w_na = local_kwargs.pop("w_na", 0.0)
        w_nn = local_kwargs.pop("w_nn", 0.0)

        weight_tensors = {
            "ao": self._normalize_weight_param(w_ao, bsize, device, dtype),
            "on": self._normalize_weight_param(w_on, bsize, device, dtype),
            "na": self._normalize_weight_param(w_na, bsize, device, dtype),
            "nn": self._normalize_weight_param(w_nn, bsize, device, dtype),
        }

        combo_specs = [
            ("ao", False, False),
            ("on", False, True),
            ("na", True, False),
            ("nn", True, True),
        ]
        num_variants = len(combo_specs)

        def _repeat_batch(tensor: Tensor | None) -> Tensor | None:
            if tensor is None:
                return None
            repeats = [num_variants] + [1] * (tensor.dim() - 1)
            return tensor.repeat(*repeats)

        # Repeat observation tensors for each guidance branch.
        images_rep = [_repeat_batch(img) for img in images]
        img_masks_rep = [_repeat_batch(mask) for mask in img_masks]
        state_rep = _repeat_batch(state)
        lang_tokens_rep = _repeat_batch(lang_tokens)
        lang_masks_rep = _repeat_batch(lang_masks)
        action_history_rep = _repeat_batch(action_history)
        action_history_mask_rep = _repeat_batch(action_history_mask)

        drop_obs_mask_all = torch.cat(
            [
                (torch.ones(bsize, dtype=torch.bool, device=device) if null_obs else torch.zeros(
                    bsize, dtype=torch.bool, device=device))
                for _, null_obs, _ in combo_specs
            ],
            dim=0,
        )

        if action_history is not None and self.history_action_proj is not None:
            drop_action_history_mask_all = torch.cat(
                [
                    (torch.ones(bsize, dtype=torch.bool, device=device) if null_action else torch.zeros(
                        bsize, dtype=torch.bool, device=device))
                    for _, _, null_action in combo_specs
                ],
                dim=0,
            )
        else:
            drop_action_history_mask_all = None

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images_rep,
            img_masks_rep,
            lang_tokens_rep,
            lang_masks_rep,
            state=state_rep,
            action_history=action_history_rep,
            action_history_mask=action_history_mask_rep,
            drop_obs_mask=drop_obs_mask_all,
            drop_action_history_mask=drop_action_history_mask_all,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        stacked_weights = torch.stack([weight_tensors[name] for name, _, _ in combo_specs], dim=0)

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        time = torch.tensor(1.0, dtype=dtype, device=device)
        x_t = noise

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            expanded_time_all = expanded_time.repeat(num_variants)
            x_t_all = x_t.repeat(num_variants, 1, 1)

            velocities_all = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t_all,
                timestep=expanded_time_all,
            )
            velocities_all = velocities_all.view(
                num_variants, bsize, self.config.chunk_size, self.config.max_action_dim
            )

            v_t = torch.sum(
                stacked_weights[:, :, None, None] * velocities_all,
                dim=0,
            )

            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    @staticmethod
    def _normalize_weight_param(param, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if isinstance(param, torch.Tensor):
            tensor = param.to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(param, dtype=dtype, device=device)

        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        elif tensor.ndim == 1:
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Weight tensor has leading dimension {tensor.shape[0]}, expected batch size {batch_size}."
                )
        else:
            raise ValueError(
                f"Weight parameter must be a scalar or 1D tensor, got shape {tuple(tensor.shape)}."
            )

        return tensor

    @staticmethod
    def _compute_prefix_weights(
        *,
        inference_delay: int,
        horizon: int,
        total: int,
        schedule: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if total <= 0:
            return torch.zeros(0, device=device, dtype=dtype)

        start = max(0, min(int(inference_delay), total))
        end = max(0, min(int(horizon), total))
        start = min(start, end)
        indices = torch.arange(total, device=device, dtype=dtype)

        schedule_key = schedule.lower()
        if schedule_key == "ones":
            weights = torch.ones(total, device=device, dtype=dtype)
        elif schedule_key == "zeros":
            weights = (indices < start).to(dtype)
        elif schedule_key in {"linear", "exp"}:
            denom = max(end - start + 1, 1)
            weights = (start - 1 - indices) / denom + 1
            weights = torch.clamp(weights, min=0.0, max=1.0)
            if schedule_key == "exp":
                weights = weights * torch.expm1(weights) / (math.e - 1)
        else:
            raise ValueError(f"Invalid prefix attention schedule: {schedule}")

        weights = torch.where(indices >= end, torch.zeros_like(weights), weights)
        return weights.to(dtype=dtype)

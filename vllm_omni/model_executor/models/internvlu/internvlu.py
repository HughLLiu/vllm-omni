# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM implementation of the InternVL-U autoregressive stage."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.models.internvl import InternVLChatModel
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.internvlu.processing import (
    InternVLUDummyInputsBuilder,
    InternVLUMultiModalProcessor,
    InternVLUProcessingInfo,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

# This order is part of the released checkpoint's parameter contract.
SPECIAL_TOKENS: tuple[str, ...] = (
    "</box>",
    "<box>",
    "<IMG_CONTEXT>",
    "</img>",
    "<img>",
    "</quad>",
    "<quad>",
    "</ref>",
    "<ref>",
    "<img_uncond>",
)


@MULTIMODAL_REGISTRY.register_processor(
    InternVLUMultiModalProcessor,
    info=InternVLUProcessingInfo,
    dummy_inputs=InternVLUDummyInputsBuilder,
)
class InternVLUChatModel(InternVLChatModel):
    """InternVision + Qwen3 with InternVL-U generation conditioning.

    The vision tower and language model are inherited from vLLM's native
    InternVL implementation.  This wrapper adds only the checkpoint-specific
    token embeddings, final-two-state capture, and image-request terminator.
    """

    have_multimodal_outputs = True
    omni_pooler_payload_include_hidden = False
    prefer_model_sampler = True
    skips_model_sampler_output_token_history = True
    supports_omni_query_start_loc = True
    supports_omni_decode_step_metadata = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config

        hidden_size = int(self.config.text_config.hidden_size)
        self.special_token_embedding = nn.Embedding(len(SPECIAL_TOKENS), hidden_size)

        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        vocab = tokenizer.get_vocab()
        missing = [token for token in SPECIAL_TOKENS if token not in vocab]
        if missing:
            raise ValueError("InternVL-U tokenizer is missing checkpoint special tokens: " + ", ".join(missing))

        special_token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in SPECIAL_TOKENS]
        self.register_buffer(
            "special_token_ids",
            torch.tensor(special_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.img_start_token_id = special_token_ids[SPECIAL_TOKENS.index("<img>")]

        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("InternVL-U requires an EOS token for internal image-request termination")
        self.eos_token_id = int(eos_token_id)

        # Deployment-wide text-then-image switch: the same engine-level
        # ``mm_processor_kwargs.think`` knob that makes the processor defer
        # the trailing <img> (see internvlu_chat_think.yaml).  Reading it here
        # keeps CoT controlled by a single configuration point.
        multimodal_config = getattr(vllm_config.model_config, "multimodal_config", None)
        mm_processor_kwargs = getattr(multimodal_config, "mm_processor_kwargs", None) or {}
        self.think = bool(mm_processor_kwargs.get("think"))

        self._sampler: Sampler | None = None
        self._last_step_input_ids: torch.Tensor | None = None
        self._last_step_query_start_loc: torch.Tensor | None = None

    def update_decode_step_metadata(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        omni_query_start_loc: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        """Keep sampler row metadata current even during CUDA graph replay."""
        if input_ids is not None:
            self._last_step_input_ids = input_ids
        self._last_step_query_start_loc = omni_query_start_loc

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply word, special-token, then vision embeddings in that order."""
        # Multimodal execution forwards prepared embeddings to the model and
        # therefore passes input_ids=None to forward(). Keep the authoritative
        # current-step IDs here for the model-local forced-EOS sampler.
        self._last_step_input_ids = input_ids
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
        )

        token_ids = self.special_token_ids.to(device=input_ids.device)
        matches = input_ids.unsqueeze(-1).eq(token_ids)
        has_special = matches.any(dim=-1)
        special_indices = matches.to(dtype=torch.int64).argmax(dim=-1)
        special_embeds = self.special_token_embedding(special_indices)
        inputs_embeds = torch.where(has_special.unsqueeze(-1), special_embeds, inputs_embeds)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required when merging InternVL-U image embeddings")
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    @staticmethod
    def _materialize_residual_stream(
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # vLLM keeps the residual stream as two tensors between decoder
        # blocks.  Their sum is the tensor that the Hugging Face model exposes
        # as hidden_states[-2] immediately before the final block.
        if residual is None:
            return hidden_states.clone()
        return hidden_states + residual

    def _forward_qwen3_with_conditioning(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        model = self.language_model.model

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = model.embed_input_ids(input_ids)
            residual = None
        else:
            if intermediate_tensors is None:
                raise ValueError("Pipeline-parallel InternVL-U rank requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        final_layer_index = int(model.config.num_hidden_layers) - 1
        pre_last_hidden: torch.Tensor | None = None
        for layer_index, layer in enumerate(
            islice(model.layers, model.start_layer, model.end_layer),
            start=model.start_layer,
        ):
            if layer_index == final_layer_index:
                pre_last_hidden = self._materialize_residual_stream(hidden_states, residual)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        if pre_last_hidden is None:
            raise RuntimeError(
                "InternVL-U conditioning capture did not execute on the pipeline rank containing the final Qwen3 block"
            )
        final_hidden, _ = model.norm(hidden_states, residual)
        return final_hidden, pre_last_hidden

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        omni_query_start_loc: torch.Tensor | None = None,
        **_: object,
    ) -> OmniOutput | IntermediateTensors:
        if input_ids is not None:
            self._last_step_input_ids = input_ids
        if omni_query_start_loc is not None:
            self._last_step_query_start_loc = omni_query_start_loc
        if intermediate_tensors is not None:
            inputs_embeds = None

        output = self._forward_qwen3_with_conditioning(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(output, IntermediateTensors):
            return output

        final_hidden, pre_last_hidden = output
        # Checkpoint contract: generation_decoder/config.json declares
        # vlm_select_layer=[-1, -2] with input_hidden_size = 2 * hidden_size,
        # i.e. final normalized state concatenated with the pre-final residual.
        conditioning = torch.cat((final_hidden, pre_last_hidden), dim=-1)
        return OmniOutput(
            text_hidden_states=final_hidden,
            multimodal_outputs={
                "hidden_states": {"output": conditioning},
            },
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        """Force EOS after a request whose current final input is ``<img>``."""
        if logits is None or logits.numel() == 0:
            return None
        if self._sampler is None:
            self._sampler = Sampler()

        force_eos = torch.zeros(
            logits.shape[0],
            dtype=torch.bool,
            device=logits.device,
        )

        input_ids = self._last_step_input_ids
        if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
            flat_ids = input_ids.reshape(-1).to(device=logits.device)
            query_start = self._last_step_query_start_loc
            if isinstance(query_start, torch.Tensor) and query_start.numel() == logits.shape[0] + 1:
                query_start = query_start.to(device=logits.device, dtype=torch.long)
                tail_indices = query_start[1:] - 1
                valid = query_start[1:] > query_start[:-1]
                tail_ids = flat_ids.index_select(0, tail_indices.clamp_min(0))
                force_eos = valid & tail_ids.eq(self.img_start_token_id)
            elif flat_ids.numel() == logits.shape[0]:
                force_eos = flat_ids.eq(self.img_start_token_id)

            min_score = torch.finfo(logits.dtype).min
            logits.masked_fill_(force_eos.unsqueeze(-1), min_score)
            eos_column = logits[:, self.eos_token_id]
            eos_column.masked_fill_(force_eos, 0.0)

        sampler_output = self._sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        if sampler_output is not None and force_eos.any():
            sampled_token_ids = sampler_output.sampled_token_ids
            sampled_token_ids[
                force_eos.to(device=sampled_token_ids.device),
                0,
            ] = self.eos_token_id

        # Think (text-then-image) deployments: the reference implementation
        # truncates the CoT at its terminating <|im_end|> and enters image
        # generation through an actual <img> token, so replace the sampled
        # EOS with <img>.  The forced-EOS rule above then terminates the
        # request one step later, after the <img> hidden state has been
        # produced.  ``self.think`` mirrors the same engine-level
        # mm_processor_kwargs knob the processor uses, so one deployment
        # setting drives both halves of the mechanism.
        #
        # Known one-token divergence from the reference: its slicing
        # (``eos_pos + input_ids_text.shape[1] - 1``) also drops the LAST CoT
        # token before <|im_end|>, so its <img> conditioning sees one CoT
        # token fewer.  That trailing ``- 1`` discards generated content with
        # no principled basis (it reads as an off-by-one), and a streaming
        # sampler cannot retract an already-processed token anyway, so we
        # deliberately keep the full CoT.  Direct-mode outputs are pinned by
        # golden-image e2e tests; think mode has a functionality-level e2e
        # only (no reference golden yet), so this divergence is documented
        # but not pixel-pinned.
        #
        # Known limitation vs the reference: if the CoT exhausts max_tokens
        # before sampling EOS, the reference truncates at the final position
        # and still appends <img>; producing that <img> hidden state here
        # would need one extra decode step after the length cap fires, which
        # a streaming sampler cannot request.  Such requests fail loudly in
        # the stage bridge with a message naming the max_tokens remedy
        # (internvlu_chat_think.yaml budgets 202 by default).
        if sampler_output is not None and self.think:
            sampled_token_ids = sampler_output.sampled_token_ids
            replace_rows = ~force_eos.to(device=sampled_token_ids.device) & (
                sampled_token_ids[:, 0] == self.eos_token_id
            )
            if replace_rows.any():
                sampled_token_ids[replace_rows, 0] = self.img_start_token_id
        return sampler_output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The inherited AutoWeightsLoader handles native InternVision/Qwen3
        # packing and also loads special_token_embedding without conversion.
        return super().load_weights(weights)

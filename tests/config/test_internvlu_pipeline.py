# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

import vllm_omni
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.stage_config import resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_model_index_uses_default_deploy(tmp_path: Path):
    """The root Diffusers index must resolve to a runnable sync pipeline."""
    (tmp_path / "model_index.json").write_text(
        '{"_class_name": "InternVLUPipeline"}',
        encoding="utf-8",
    )

    stages, _omni_lb_policy = StageConfigFactory.create_from_model(
        str(tmp_path),
        trust_remote_code=False,
    )

    assert stages is not None
    assert len(stages) == 2
    stage0, stage1 = stages
    assert stage0.yaml_engine_args["async_chunk"] is False
    assert stage1.yaml_engine_args["async_chunk"] is False
    assert stage0.yaml_engine_args["enable_prefix_caching"] is False
    assert stage0.yaml_engine_args["gpu_memory_utilization"] == 0.45
    assert stage0.yaml_engine_args["max_num_seqs"] == 3
    assert stage1.yaml_engine_args["diffusion_attention_backend"] == "TORCH_SDPA"
    assert stage0.yaml_extras["prompt_expand_func"].endswith(".expand_cfg_prompts")
    assert stage1.custom_process_input_func is not None
    assert stage1.custom_process_input_func.endswith(".ar2diffusion")


def test_think_deploy_statically_enables_text_then_image():
    """internvlu_chat_think.yaml layers the single think knob over the base deploy."""
    deploy_dir = Path(vllm_omni.__file__).parent / "deploy"
    merged = resolve_deploy_yaml(deploy_dir / "internvlu_chat_think.yaml")

    assert merged["pipeline"] == "internvlu_chat"
    stages = {stage["stage_id"]: stage for stage in merged["stages"]}

    stage0 = stages[0]
    # One knob drives both the processor formatting and the sampler.
    assert stage0["mm_processor_kwargs"] == {"think": True}
    assert stage0["default_sampling_params"] == {"max_tokens": 202}
    # Base stage-0 settings must survive the overlay.
    assert stage0["enforce_eager"] is True
    assert stage0["gpu_memory_utilization"] == 0.45
    assert stage0["max_num_seqs"] == 3

    assert stages[1]["diffusion_attention_backend"] == "TORCH_SDPA"

# Qwen3.5 Compatibility Plan

This document describes the repo changes needed to make VLM2Vec fully compatible with Qwen3.5 vision-language models, especially `Qwen/Qwen3.5-27B`.

Qwen3.5 should not be treated as only a renamed Qwen2.5-VL model. It keeps the same broad multimodal pattern, but the processor/model contract is stricter because the model builds 3D multimodal RoPE positions from processor-returned modality metadata.

## Qwen-VL Handling Today

The current Qwen2/Qwen2.5-VL path is centered in `src/model/processor.py`.

For model registration:

- `MODEL2BACKBONE` maps HF `config.model_type` values such as `qwen2_vl` and `qwen2_5_vl` to internal constants.
- `backbone2model` maps those internal constants to the vendored model classes.
- `VLM_IMAGE_TOKENS` and `VLM_VIDEO_TOKENS` provide placeholder tokens, currently `<|image_pad|>` and `<|video_pad|>` for Qwen backbones.

For processor loading:

- `load_processor()` has explicit branches for Qwen2-VL and Qwen2.5-VL.
- Those branches construct the local processor/image processor/tokenizer classes from `src/model/vlm_backbone/qwen2_vl` or `src/model/vlm_backbone/qwen2_5_vl`.
- Resize settings from `DataArguments` are threaded into the image processor.

For batch processing:

- `Qwen2_VL_process_fn()` manually iterates over each example because the repo supports mixed text-only, image, and video rows in the same batch.
- It calls the processor per row.
- It manually pads `input_ids`.
- It returns the fields needed by Qwen2/Qwen2.5-VL forward:
  - `input_ids`
  - `attention_mask`
  - `pixel_values`
  - `image_grid_thw`
  - `pixel_values_videos`
  - `video_grid_thw`

This is sufficient for Qwen2/Qwen2.5-VL because their model code can infer multimodal RoPE placement from placeholder token positions plus image/video grids.

## Why Qwen3.5 Needs Additional Handling

Qwen3.5 uses a 3D multimodal RoPE layout in the language model. Text spans receive 1D positions expanded across the three RoPE axes, while image/video spans receive true temporal-height-width coordinates.

The Qwen3.5 model forward accepts:

- `input_ids`
- `attention_mask`
- `pixel_values`
- `pixel_values_videos`
- `image_grid_thw`
- `video_grid_thw`
- `mm_token_type_ids`

The key difference is `mm_token_type_ids`.

`mm_token_type_ids` marks each sequence position as:

- `0`: text
- `1`: image
- `2`: video

Qwen3.5 uses this to split the sequence into text, image, and video spans before computing multimodal position IDs. If multimodal grids are present but `mm_token_type_ids` is missing, the model raises an error because it cannot compute correct M-RoPE positions.

This means the Qwen2-style process function is not enough if it reconstructs the output dictionary and drops processor-returned fields.

## Required Changes

### 1. Dependencies

Qwen3.5 requires a Transformers version that includes the Qwen3.5 model and processor implementation.

Update and verify:

- `pyproject.toml`
- `requirements.txt`
- `uv.lock`

The installed Transformers package must provide:

- `Qwen3_5ForConditionalGeneration`
- a Qwen3.5-compatible processor, either through `AutoProcessor` or a concrete class exposed by the installed version

Also consider optional performance dependencies:

- `flash-linear-attention`
- `causal-conv1d`

These are not required for correctness, but Qwen3.5's Gated DeltaNet layers use faster kernels when available.

### 2. Registry Cleanup

Update `src/model/processor.py`.

Current issues to fix:

- `QWEN2_VL_TOKENSELECTION` is assigned twice.
- `MODEL2BACKBONE` contains duplicate `qwen2_vl_tokenselection` entries.
- `backbone2model` contains duplicate `QWEN3_5` entries.

Expected state:

- `MODEL2BACKBONE["qwen3_5"] == QWEN3_5`
- `backbone2model[QWEN3_5] == Qwen3_5ForConditionalGeneration`
- only include `QWEN3_5_TOKENSELECTION` if a real Qwen3.5 token-selection implementation exists

Why:

- `get_backbone_name()` asserts that `hf_config.model_type` is in `SUPPORTED_MODELS`.
- Duplicated dictionary keys hide mistakes because Python silently keeps the last value.

### 3. Processor Loading

Add an explicit Qwen3.5 branch in `load_processor()`.

Current Qwen-VL handling:

- Qwen2/Qwen2.5 use vendored local processor classes.
- Processor resize settings are wired manually.

Needed for Qwen3.5:

- Use the official Qwen3.5-compatible processor from Transformers.
- Prefer `AutoProcessor.from_pretrained(...)` unless the installed Transformers version exposes a stable concrete Qwen3.5 processor class.
- Pass `trust_remote_code=True` if needed.
- Preserve resize behavior by forwarding image/video processor settings where supported.

Implementation shape:

```python
elif model_args.model_backbone == QWEN3_5:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_args.processor_name if model_args.processor_name else model_name_or_path,
        trust_remote_code=True,
    )
```

If resize controls are required, validate how the Qwen3.5 processor exposes image/video processors in the installed Transformers version. Do not assume the vendored Qwen2 image processor API is identical.

Why:

- Qwen3.5 model repos declare a Qwen3-compatible processor and processor config.
- The processor is responsible for creating not just visual tensors and grids, but also `mm_token_type_ids`.

### 4. Dedicated `Qwen3_5_process_fn`

Add and use a dedicated Qwen3.5 process function in `src/model/processor.py`.

There is already an initial `Qwen3_5_process_fn` in the working tree, but it still needs to:

- collect `mm_token_type_ids`
- pad `mm_token_type_ids` to match padded `input_ids`
- include `mm_token_type_ids` in the returned dictionary
- preserve any other processor-returned model inputs that Qwen3.5 needs
- be registered in `process_vlm_inputs_fns`

Register it:

```python
process_vlm_inputs_fns = {
    ...
    QWEN3_5: Qwen3_5_process_fn,
}
```

Why:

- Train and eval collators call `process_vlm_inputs_fns[model_backbone]`.
- Without the mapping, Qwen3.5 fails with a `KeyError`.
- Without `mm_token_type_ids`, Qwen3.5 fails or computes incorrect multimodal position IDs.

Recommended behavior:

- Iterate over `model_inputs["text"]` and `model_inputs["images"]` exactly like the Qwen2-VL process function.
- For text-only rows, call the processor without images/videos.
- For image rows, pass images.
- For video rows, pass videos.
- Collect per-row `input_ids` and `mm_token_type_ids`.
- Use `processor.tokenizer.pad()` for `input_ids`.
- Pad `mm_token_type_ids` with `0` to the same length as padded `input_ids`.

Padding `mm_token_type_ids` with `0` is appropriate because padding should not be treated as image/video. The `attention_mask` still masks out padded positions.

Implementation sketch:

```python
batch_encoding = processor.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
input_ids = batch_encoding["input_ids"].long()
attention_mask = batch_encoding["attention_mask"].long()

max_seq_len = input_ids.shape[1]
mm_token_type_ids = [
    torch.nn.functional.pad(
        torch.as_tensor(ids, dtype=torch.long),
        (0, max_seq_len - len(ids)),
        value=0,
    )
    for ids in per_example_mm_token_type_ids
]
mm_token_type_ids = torch.stack(mm_token_type_ids, dim=0)
```

Returned dictionary should include at least:

```python
{
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "mm_token_type_ids": mm_token_type_ids,
    "pixel_values": pixel_values,
    "image_grid_thw": image_grid_thw,
    "pixel_values_videos": pixel_values_videos,
    "video_grid_thw": video_grid_thw,
    "texts": texts,
    "images": visual_inputs,
}
```

### 5. Preserve Qwen3.5 Processor Output Carefully

The safest Qwen3.5 process function should avoid hardcoding only Qwen2-era keys.

Current Qwen2 behavior reconstructs a small dictionary and drops anything else the processor returned. That is acceptable for Qwen2/Qwen2.5, but risky for Qwen3.5.

For Qwen3.5:

- explicitly keep known keys required by model forward
- log or inspect unexpected processor keys during early testing
- avoid dropping keys such as `second_per_grid_ts`, `video_metadata`, or future processor metadata if the installed processor returns them and the model accepts them

Why:

- Qwen3.5 video handling differs from Qwen2/Qwen2.5. The model code notes that videos may be split by timestamps, so video grid handling is stricter.

### 6. Prompt Token Handling

Current repo prompt insertion uses:

- image: `<|image_pad|>`
- video: `<|video_pad|>`

Qwen3.5 tokenizer config uses the same image/video pad tokens, but its chat template wraps them as:

```text
<|vision_start|><|image_pad|><|vision_end|>
<|vision_start|><|video_pad|><|vision_end|>
```

Decision needed:

- If direct processor calls with raw `text=[...]` correctly expand/process bare `<|image_pad|>` and `<|video_pad|>`, keep current prompt generation.
- If the official Qwen3.5 processor expects the full chat-template wrapper, update Qwen3.5 prompt insertion to use the wrapped form.

Why:

- The number and location of placeholder tokens must match the number and location of visual features.
- Qwen3.5 replaces placeholder token embeddings with visual embeddings. A mismatch causes runtime errors.

Validation required:

- One text+image example.
- One text+video example.
- Check that `input_ids` contains the expected image/video token IDs.
- Check that visual feature count matches placeholder token count.

### 7. Model Build And Load Paths

Update `src/model/model.py`.

Build path:

- `MMEBModel.build()` already has a Qwen3.5 branch.
- Confirm that top-level and nested attention settings are correct for the installed Transformers version.

Load path:

- `MMEBModel.load()` currently has a VLM branch for Qwen2/Qwen2.5 but not Qwen3.5.
- Add `QWEN3_5` to the VLM load branch or add a dedicated Qwen3.5 branch.

Needed:

```python
if model_args.model_backbone in {
    LLAVA_NEXT,
    QWEN2_VL,
    QWEN2_5_VL,
    QWEN3_5,
    QWEN2_VL_TOKENSELECTION,
    QWEN2_5_VL_TOKENSELECTION,
    E5_V,
}:
    ...
```

For Qwen3.5, also set:

- `config.use_cache = False`
- `config.padding_side = "left"`
- `config._attn_implementation = "flash_attention_2"` if supported
- `config.vision_config._attn_implementation = "flash_attention_2"` if supported
- `config.text_config.use_cache = False` if the installed config stores cache under `text_config`

Why:

- Training uses `build()`, but eval/checkpoint resume may use `load()`.
- If `load()` falls through to `AutoModelForCausalLM`, the vision-language model can be loaded incorrectly.

### 8. Collator Allow Lists

Update:

- `src/data/collator/train_collator.py`
- `src/data/collator/eval_collator.py`

Current Qwen-specific resize decay assertions only allow Qwen2/Qwen2.5 constants. Add `QWEN3_5` if `image_decay_factor` should be supported with Qwen3.5.

Why:

- Otherwise Qwen3.5 will assert when using image decay settings even though it is a Qwen VLM.

### 9. LoRA Target Modules

Review `ModelArguments.lora_target_modules` in `src/arguments.py`.

Qwen3.5 has module names that differ from Qwen2/Qwen2.5, especially in Gated DeltaNet layers:

- `in_proj_qkv`
- `in_proj_z`
- `in_proj_b`
- `in_proj_a`
- `out_proj`
- `linear_fc1`
- `linear_fc2`

It also has full-attention layer modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

Why:

- Existing default LoRA target modules are Qwen2/Qwen2.5-oriented.
- If Qwen3.5-specific modules are not included, most linear-attention layers may receive no adapters.

Recommendation:

- Add a Qwen3.5-specific default or document a Qwen3.5 training config override.
- Avoid one global default if it leads to invalid module names for other backbones.

### 10. Token Selection Variant

Do not advertise full `QWEN3_5_TOKENSELECTION` support unless implemented.

Current state:

- `QWEN3_5_TOKENSELECTION` is declared.
- `backbone2model` points it to the normal `Qwen3_5ForConditionalGeneration`.
- There is no Qwen3.5 token-selection model implementation equivalent to the vendored Qwen2 token-selection code.

Needed:

- Either remove/disable `QWEN3_5_TOKENSELECTION` routing, or implement a real token-selection variant.

Why:

- Mapping a token-selection backbone name to the base model is misleading and may silently skip intended functionality.

## Testing Checklist

Run these in order.

### Processor smoke tests

For each of text-only, image, and video:

- call `load_processor()`
- call `Qwen3_5_process_fn()`
- print keys, shapes, and dtypes
- verify `mm_token_type_ids` exists
- verify `input_ids.shape == attention_mask.shape == mm_token_type_ids.shape`

Expected keys for image:

- `input_ids`
- `attention_mask`
- `mm_token_type_ids`
- `pixel_values`
- `image_grid_thw`

Expected keys for video:

- `input_ids`
- `attention_mask`
- `mm_token_type_ids`
- `pixel_values_videos`
- `video_grid_thw`

### Forward smoke tests

Run a no-grad forward pass with:

- text-only
- text+image
- text+video

Use:

```python
model(**inputs, return_dict=True, output_hidden_states=True)
```

Verify:

- no missing `mm_token_type_ids` error
- no image/video token count mismatch
- `hidden_states[-1]` exists
- pooled representation path in `MMEBModel.encode_input()` works

### Training smoke test

Run a tiny one-step training config:

- batch size 1 or 2
- one image dataset item
- one text-only item if available
- one video item if Qwen3.5 video support is required

Verify:

- collator selects `Qwen3_5_process_fn`
- tensors move to device correctly
- loss is finite

### Checkpoint/eval smoke test

Verify `MMEBModel.load()` with Qwen3.5:

- loads `Qwen3_5ForConditionalGeneration`
- does not fall back to `AutoModelForCausalLM`
- can encode one image/text example after loading

## Summary

The minimum compatibility work is:

1. Ensure dependencies expose Qwen3.5 model and processor support.
2. Clean Qwen3.5 registry entries in `src/model/processor.py`.
3. Add explicit Qwen3.5 processor loading.
4. Finish `Qwen3_5_process_fn` so it preserves and pads `mm_token_type_ids`.
5. Register `QWEN3_5` in `process_vlm_inputs_fns`.
6. Update `MMEBModel.load()` so Qwen3.5 uses the VLM load path.
7. Extend Qwen-specific collator allowlists where needed.
8. Review Qwen3.5 LoRA targets.
9. Validate text/image/video processor and forward smoke tests.

The core difference from Qwen2/Qwen2.5-VL is that Qwen3.5 needs processor-provided modality IDs to compute 3D multimodal RoPE correctly. The rest of the repo can mostly keep the existing Qwen-VL mixed-batch pattern once that metadata is preserved through the collator and into the model forward.

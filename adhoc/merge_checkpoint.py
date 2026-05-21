"""
Merge a checkpoint into a single directory.

Usage:

python adhoc/merge_checkpoint.py \
  --model_name Qwen/Qwen3.5-0.8b \
  --checkpoint_path /workspace/VLM2Vec/outputs/checkpoint-4750 \
  --save_dir /workspace/VLM2Vec/outputs/Qwen3.5-08b-merged-4750 \
  --lora \
  --pooling eos \
  --normalize True
"""


from src.arguments import ModelArguments
from transformers import HfArgumentParser, AutoProcessor

from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor



def main(save_dir: str, remaining_args: list):

    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses(args=remaining_args)
    model_args: ModelArguments

    model = MMEBModel.build(model_args)
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, "model_backbone", model_backbone)
    # processor.tokenizer.padding_side = "right"
    model = MMEBModel.load(model_args, is_trainable=False)
    model.config.save_pretrained(f'{save_dir}', safe_serialization=True)
    processor = load_processor(model_args)
    processor.save_pretrained(f'{save_dir}', safe_serialization=True)
    model.encoder._hf_peft_config_loaded = False
    model.encoder.save_pretrained(f'{save_dir}', safe_serialization=True)


if __name__ == "__main__":
    import argparse
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--save_dir", type=str, required=True)
    args, remaining = aparser.parse_known_args()
    save_dir = args.save_dir
    main(save_dir, remaining)

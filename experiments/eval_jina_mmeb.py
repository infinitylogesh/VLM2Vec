#!/usr/bin/env python3
"""
Standalone MMEB image evaluation for Jina v5-omni models.

Key insight for correct Jina encoding:
  - Jina's processor only embeds the image when '<image>' appears in the text.
    Without it, pixel_values are ignored and scores are random.
  - Queries with images: text must contain '<image>', use model.embed()
    with the retrieval adapter active.
  - Text-only candidates: use model.encode(texts, task='retrieval', prompt_name='document')
    which adds 'Document: ' prefix and applies the retrieval LoRA adapter.
  - Image candidates (i2i tasks): prepend '<image>' to cand text, use model.embed().

Usage:
    python experiments/eval_jina_mmeb.py \\
        --model_name jinaai/jina-embeddings-v5-omni-nano \\
        --output_dir outputs/evaluation/jina-nano \\
        --dataset_config experiments/public/eval/image.yaml \\
        --data_basedir data/vlm2vec_eval \\
        --batch_size 8
"""

import argparse
import io
import json
import os
import pickle
import sys
import yaml

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# ── project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Register all dataset parsers (image / video / visdoc)
import src.data.eval_dataset.image_cls_dataset           # noqa: F401
import src.data.eval_dataset.image_qa_dataset            # noqa: F401
import src.data.eval_dataset.image_i2t_eval              # noqa: F401
import src.data.eval_dataset.image_t2i_eval              # noqa: F401
import src.data.eval_dataset.image_i2i_vg_dataset        # noqa: F401
# Video parsers — Jina uses first frame of each clip as image proxy
import src.data.eval_dataset.msrvtt_dataset              # noqa: F401
import src.data.eval_dataset.msvd_dataset                # noqa: F401
import src.data.eval_dataset.didemo_dataset              # noqa: F401
import src.data.eval_dataset.vatex_dataset               # noqa: F401
import src.data.eval_dataset.youcook2_dataset            # noqa: F401
import src.data.eval_dataset.ssv2_dataset                # noqa: F401
import src.data.eval_dataset.video_classification_datasets  # noqa: F401
import src.data.eval_dataset.mvbench_dataset             # noqa: F401
import src.data.eval_dataset.nextqa_dataset              # noqa: F401
import src.data.eval_dataset.egoschema_dataset           # noqa: F401
import src.data.eval_dataset.activitynetqa_dataset       # noqa: F401
import src.data.eval_dataset.moment_retrieval_datasets   # noqa: F401
import src.data.eval_dataset.momentseeker_dataset        # noqa: F401
import src.data.eval_dataset.videomme_dataset            # noqa: F401
# Visdoc parsers
import src.data.eval_dataset.vidore_dataset              # noqa: F401
import src.data.eval_dataset.visrag_dataset              # noqa: F401

from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from src.utils.eval_utils.metrics import RankingMetrics


# ── minimal stand-in args ─────────────────────────────────────────────────────
class _Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_model_args():
    return _Namespace(
        model_backbone="gme",   # GME: no VLM image tokens added to text, clean instruction
        model_name="",
        model_type=None,
        checkpoint_path=None,
        lora=False,
    )


def _make_data_args(data_basedir=""):
    return _Namespace(
        image_resolution=None,
        data_basedir=data_basedir,
        max_len=512,
        resize_use_processor=True,
        image_decay_factor=None,
        resize_min_pixels=None,
        resize_max_pixels=None,
    )


# ── image loading ─────────────────────────────────────────────────────────────
def _load_pil(image_info) -> Image.Image | None:
    if image_info is None or not isinstance(image_info, dict):
        return None
    paths = image_info.get("paths") or []
    byts  = image_info.get("bytes") or []
    path      = paths[0] if paths else None
    byte_data = byts[0]  if byts  else None
    try:
        if byte_data is not None:
            return Image.open(io.BytesIO(byte_data)).convert("RGB")
        if path is not None:
            return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"  [warn] image load failed ({path}): {e}")
    return None


# ── encoding ──────────────────────────────────────────────────────────────────
# Default; overridden at runtime from processor.image_token
IMAGE_TOKEN = "<image>"


def _get_image_token(processor) -> str:
    """Return the correct image placeholder for this processor variant.
    Nano (LlavaEuroBert) uses '<image>'; Small (Qwen2-VL) uses '<|image_pad|>'.
    """
    return getattr(processor, "image_token", None) or "<image>"


@torch.no_grad()
def _encode_image_group(model, processor, texts, images, device):
    """Encode a batch where every item has an image.
    Prepends the model-specific image token to text so the processor inserts
    visual tokens.  Uses model.embed() with the currently active adapter.
    """
    img_tok = _get_image_token(processor)
    # Prepend image token so the processor correctly maps pixel_values
    prompted_texts = [f"{img_tok} {t}" if t.strip() else img_tok for t in texts]
    try:
        inputs = processor(
            text=prompted_texts, images=images,
            return_tensors="pt", padding=True, truncation=False,
        )
    except Exception as e:
        print(f"  [warn] image batch processor failed, item-by-item: {e}")
        parts = []
        for t, img in zip(prompted_texts, images):
            try:
                inp = processor(text=[t], images=[img], return_tensors="pt", truncation=False)
                inp = {k: v.to(device) for k, v in inp.items() if isinstance(v, torch.Tensor)}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb = model.embed(**inp)
                parts.append(emb.cpu().float())
            except Exception as e2:
                print(f"    [warn] item failed: {e2}; using zero vector")
                hidden = getattr(model.config, "hidden_size", 1024)
                parts.append(torch.zeros(1, hidden))
        return torch.cat(parts, dim=0)

    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embs = model.embed(**inputs)
    return embs.cpu().float()


@torch.no_grad()
def _encode_text_group(model, texts, device, batch_size, prompt_name="document"):
    """Encode text-only items using model.encode() with retrieval adapter.
    prompt_name: 'query' for search queries, 'document' for candidates/documents.
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, task="retrieval", prompt_name=prompt_name)
        all_embs.append(embs.cpu().float())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_dataset(model, processor, dataset, text_key, image_key, encode_side, device, batch_size, desc=""):
    """Encode all items in a HF dataset."""
    all_embs  = []
    all_infos = []

    for start in tqdm(range(0, len(dataset), batch_size), desc=desc):
        rows = dataset[start : start + batch_size]
        n = len(rows[text_key])

        texts, images = [], []
        for i in range(n):
            ex_texts  = rows[text_key][i]
            ex_images = rows[image_key][i]
            for text, img_info in zip(ex_texts, ex_images):
                texts.append(text.strip() if text else "")
                images.append(_load_pil(img_info))

        # Partition by image presence
        img_indices  = [i for i, img in enumerate(images) if img is not None]
        text_indices = [i for i, img in enumerate(images) if img is None]

        result = [None] * len(texts)

        if img_indices:
            g_texts  = [texts[i]  for i in img_indices]
            g_images = [images[i] for i in img_indices]
            g_embs = _encode_image_group(model, processor, g_texts, g_images, device)
            for out_i, emb in zip(img_indices, g_embs):
                result[out_i] = emb

        if text_indices:
            g_texts = [texts[i] for i in text_indices]
            pname = "query" if encode_side == "qry" else "document"
            g_embs = _encode_text_group(model, g_texts, device, batch_size, prompt_name=pname)
            for out_i, emb in zip(text_indices, g_embs):
                result[out_i] = emb

        all_embs.append(torch.stack(result).numpy())

        for i in range(n):
            info = rows["dataset_infos"][i]
            if encode_side == "qry":
                all_infos.append(info)
            else:
                all_infos.append(info.get("cand_name", f"unk_{start+i}"))

    return np.vstack(all_embs), all_infos


# ── scoring ───────────────────────────────────────────────────────────────────
def score_dataset(qry_embs, qry_infos, cand_embed_dict, task_config, dataset_name, output_path):
    eval_type = task_config.get("eval_type", "global")
    metrics_list = task_config.get("metrics") or ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
    pred_dicts = []

    if eval_type == "global":
        cand_keys  = list(cand_embed_dict.keys())
        cand_embs  = np.stack([cand_embed_dict[k] for k in cand_keys])
        cos_scores = np.dot(qry_embs, cand_embs.T)
        ranked     = np.argsort(-cos_scores, axis=1)
        for ranked_row, gt_info in zip(ranked, qry_infos):
            rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
            rel_scores = gt_info.get("rel_scores")
            pred_dicts.append({
                "prediction": [cand_keys[i] for i in ranked_row],
                "label":      rel_docids,
                "rel_scores": rel_scores,
            })
    else:  # local
        for qry_emb, gt_info in zip(qry_embs, qry_infos):
            cand_names = gt_info["cand_names"]
            cand_embs  = np.stack([cand_embed_dict[k] for k in cand_names])
            cos_score  = np.dot(qry_emb, cand_embs.T)
            ranked     = np.argsort(-cos_score)
            rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
            rel_scores = gt_info.get("rel_scores")
            pred_dicts.append({
                "prediction": [cand_names[i] for i in ranked],
                "label":      rel_docids,
                "rel_scores": rel_scores,
            })

    metrics    = RankingMetrics(metrics_list)
    score_dict = metrics.evaluate(pred_dicts)
    score_dict["num_pred"] = len(pred_dicts)
    score_dict["num_data"] = len(qry_infos)

    formatted = {k: f"{v:.4f}" for k, v in score_dict.items() if isinstance(v, float)}
    print(f"  Score {dataset_name}: {formatted}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(score_dict, f, indent=4)
    print(f"  Saved → {output_path}")
    return score_dict


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",     required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--dataset_config", default="experiments/public/eval/image.yaml")
    parser.add_argument("--data_basedir",   default="data/vlm2vec_eval")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--device",         default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"\nLoading model: {args.model_name}")
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    # Set retrieval adapter as default for all image queries
    model.set_adapter(["retrieval"])
    print(f"Model loaded. Active task: retrieval")

    with open(args.dataset_config) as f:
        dataset_configs = yaml.safe_load(f)

    model_args = _make_model_args()
    data_args  = _make_data_args(data_basedir=args.data_basedir)

    all_scores = {}
    for dataset_name, task_config in dataset_configs.items():
        score_path = os.path.join(args.output_dir, f"{dataset_name}_score.json")
        if os.path.exists(score_path):
            with open(score_path) as f:
                sd = json.load(f)
            print(f"[skip] {dataset_name}")
            all_scores[dataset_name] = sd
            continue

        print(f"\n{'='*60}\n{dataset_name}\n{'='*60}")

        tc = dict(task_config)
        for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
            if args.data_basedir and tc.get(key):
                tc[key] = os.path.join(args.data_basedir, tc[key])

        try:
            qry_dataset, corpus = AutoEvalPairDataset.instantiate(
                model_args=model_args,
                data_args=data_args,
                **tc,
            )
            cand_dataset = generate_cand_dataset(qry_dataset, corpus)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [error] {dataset_name}: {e} — skipping")
            continue

        print(f"  #queries={len(qry_dataset)}  #cands={len(cand_dataset)}")

        qry_embed_path  = os.path.join(args.output_dir, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(args.output_dir, f"{dataset_name}_tgt")
        info_path       = os.path.join(args.output_dir, f"{dataset_name}_info.jsonl")

        # Query embeddings
        if os.path.exists(qry_embed_path) and os.path.exists(info_path):
            with open(qry_embed_path, "rb") as f: qry_embs = pickle.load(f)
            with open(info_path) as f: qry_infos = [json.loads(l) for l in f]
            print(f"  [cache] queries")
        else:
            qry_embs, qry_infos = encode_dataset(
                model, processor, qry_dataset,
                text_key="query_text", image_key="query_image",
                encode_side="qry", device=device,
                batch_size=args.batch_size,
                desc=f"{dataset_name} queries",
            )
            with open(qry_embed_path, "wb") as f: pickle.dump(qry_embs, f)
            with open(info_path, "w") as f:
                for info in qry_infos:
                    f.write(json.dumps(info) + "\n")

        # Candidate embeddings
        if os.path.exists(cand_embed_path):
            with open(cand_embed_path, "rb") as f: cand_embed_dict = pickle.load(f)
            print(f"  [cache] candidates")
        else:
            cand_embs, cand_ids = encode_dataset(
                model, processor, cand_dataset,
                text_key="cand_text", image_key="cand_image",
                encode_side="cand", device=device,
                batch_size=args.batch_size,
                desc=f"{dataset_name} candidates",
            )
            cand_embed_dict = {cid: emb for cid, emb in zip(cand_ids, cand_embs)}
            with open(cand_embed_path, "wb") as f: pickle.dump(cand_embed_dict, f)

        sd = score_dataset(qry_embs, qry_infos, cand_embed_dict, task_config, dataset_name, score_path)
        all_scores[dataset_name] = sd

    # Summary
    primary = {}
    for ds, sd in all_scores.items():
        for k, v in sd.items():
            if isinstance(v, float):
                primary[ds] = v
                break
    if primary:
        avg = np.mean(list(primary.values()))
        print(f"\n{'='*60}")
        print(f"Overall avg ({len(primary)} datasets): {avg:.4f}")
        for ds, v in primary.items():
            print(f"  {ds}: {v:.4f}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "per_dataset": primary,
            "avg": float(np.mean(list(primary.values()))) if primary else 0.0,
        }, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()

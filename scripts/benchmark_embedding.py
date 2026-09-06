"""嵌入编码加速对比：sentence-transformers(PyTorch) vs ONNX Runtime fp32 / int8 量化（CPU）。

不依赖 optimum（其 2.x 与 sentence-transformers 6 的 ONNX 后端不兼容），
直接用 torch.onnx.export 导出「Transformer + 均值池化 + L2 归一化」计算图，
再分别以 fp32 与 int8（quantize_dynamic）推理测速。

用法：python scripts/benchmark_embedding.py
结果写入 eval/results/benchmark_embedding.json
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  ***REMOVED*** noqa: E402

CACHE = ROOT / "data" / ".cache"
N_TEXTS = 256
REPEAT = 3


def make_texts() -> list[str]:
    return [f"这是第{i}条测试句子，用于嵌入编码基准测试，内容涉及设备{i}的维护要求与费用标准{i}。" for i in range(N_TEXTS)]


def main():
    import torch
    import torch.nn as nn
    import torch.onnx

    from rag.embeddings import get_model

    texts = make_texts()
    st_model = get_model()

    def bench_st() -> float:
        st_model.encode(texts[:16], show_progress_bar=False)  ***REMOVED*** 预热
        best = float("inf")
        for _ in range(REPEAT):
            t0 = time.time()
            st_model.encode(texts, batch_size=64, show_progress_bar=False)
            best = min(best, time.time() - t0)
        return best

    t_torch = bench_st()
    print(f"PyTorch(CPU):   {t_torch:.2f}s / {N_TEXTS} 条 → {N_TEXTS / t_torch:.0f} 条/s")

    ***REMOVED*** ---------- 导出 ONNX：Transformer + 均值池化 + L2 归一化 ----------
    class MeanPoolModel(nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            hidden = self.transformer(**kwargs)[0]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    core = MeanPoolModel(st_model._first_module().auto_model).eval()
    onnx_fp32 = CACHE / "text2vec-fp32.onnx"
    onnx_int8 = CACHE / "text2vec-int8.onnx"
    CACHE.mkdir(parents=True, exist_ok=True)

    result = {"model": "shibing624/text2vec-base-chinese", "n_texts": N_TEXTS, "repeat": REPEAT}
    try:
        if not onnx_fp32.exists():
            enc = st_model.tokenizer(["测试输入"], return_tensors="pt")
            sample = (enc["input_ids"], enc["attention_mask"],
                      enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
            ***REMOVED*** 始终传入并命名 3 个输入（BERT 需要 token_type_ids）
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            torch.onnx.export(
                core, sample, str(onnx_fp32),
                input_names=input_names, output_names=["embedding"],
                dynamic_axes={n: {0: "batch", 1: "seq"} for n in input_names} | {"embedding": {0: "batch"}},
                opset_version=17, do_constant_folding=True,
                dynamo=False)  ***REMOVED*** torch 2.14 的 dynamo 导出器与 onnxruntime 不兼容，用传统导出
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic

        if not onnx_int8.exists():
            quantize_dynamic(str(onnx_fp32), str(onnx_int8), weight_type=QuantType.QInt8)

        sess_fp32 = ort.InferenceSession(str(onnx_fp32), providers=["CPUExecutionProvider"])
        sess_int8 = ort.InferenceSession(str(onnx_int8), providers=["CPUExecutionProvider"])
        tok = st_model.tokenizer

        def bench_onnx(sess, tag) -> float:
            enc = tok(texts[:16], return_tensors="np", padding=True, truncation=True, max_length=512)
            sess.run(None, {k: v.astype(np.int64) for k, v in enc.items()})  ***REMOVED*** 预热
            best = float("inf")
            for _ in range(REPEAT):
                enc = tok(texts, return_tensors="np", padding=True, truncation=True, max_length=512)
                feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in
                         [i.name for i in sess.get_inputs()]}
                t0 = time.time()
                sess.run(None, feeds)
                best = min(best, time.time() - t0)
            print(f"{tag}: {best:.2f}s / {N_TEXTS} 条 → {N_TEXTS / best:.0f} 条/s")
            return best

        t_fp32 = bench_onnx(sess_fp32, "ONNX fp32(CPU):")
        t_int8 = bench_onnx(sess_int8, "ONNX int8(CPU):")
        result.update({
            "pytorch_seconds": round(t_torch, 3),
            "onnx_fp32_seconds": round(t_fp32, 3),
            "onnx_int8_seconds": round(t_int8, 3),
            "pytorch_texts_per_sec": round(N_TEXTS / t_torch, 1),
            "onnx_fp32_texts_per_sec": round(N_TEXTS / t_fp32, 1),
            "onnx_int8_texts_per_sec": round(N_TEXTS / t_int8, 1),
            "speedup_fp32": round(t_torch / t_fp32, 2),
            "speedup_int8": round(t_torch / t_int8, 2),
            "note": "同机同批大小对比；PyTorch 编码含 sentence-transformers 归一化流水线",
        })
    except Exception as e:  ***REMOVED*** noqa: BLE001
        print(f"ONNX 导出/推理失败（{type(e).__name__}: {str(e)[:150]}）")
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    result["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = ROOT / "eval" / "results" / "benchmark_embedding.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 结果已写入 {out}")


if __name__ == "__main__":
    main()

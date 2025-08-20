import argparse
import dataclasses
import os
import time
from typing import List

import numpy as np
import torch_xla.debug.profiler as xp
from tqdm import tqdm
from typing import List, Tuple, Dict
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser
import contextlib
DURATION_MS = int(os.getenv("VLLM_TPU_PROFILE_DURATION_MS", 3000))
DELAY_MS = int(os.getenv("VLLM_TPU_PROFILE_DELAY_MS", 0))

TTFT_METRIC_NAME = "vllm:time_to_first_token_seconds"
TPOT_METRIC_NAME = "vllm:time_per_output_token_seconds"
try:
    from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector  # type: ignore
except Exception:
    Counter = Gauge = Histogram = Vector = type("X", (), {})  # dummy

# ========= max_len 计算 (代码不变) =========
def _required_len_for(prompt_tokens: int, max_new_tokens: int) -> int:
    need = prompt_tokens + max_new_tokens
    return min(int(need * 1.05) + 256, 200_000)

def _iter_children_of_vector(vec_obj):
    for attr in ("children", "metrics", "series", "values", "samples", "items"):
        if hasattr(vec_obj, attr):
            val = getattr(vec_obj, attr)
            if isinstance(val, dict):
                for v in val.values():
                    yield v
            else:
                with contextlib.suppress(TypeError):
                    for v in val:
                        yield v
def _collect_hist_sum_count(metrics, metric_name: str):
    total_sum = 0.0
    total_count = 0.0
    for m in metrics:
        mname = getattr(m, "name", None)
        if mname != metric_name:
            continue
        if isinstance(m, Histogram) or m.__class__.__name__ == "Histogram":
            total_sum += float(getattr(m, "sum", 0.0))
            total_count += float(getattr(m, "count", 0.0))
            continue
        if isinstance(m, Vector) or m.__class__.__name__ == "Vector":
            for child in _iter_children_of_vector(m):
                if isinstance(child, Histogram) or child.__class__.__name__ == "Histogram":
                    total_sum += float(getattr(child, "sum", 0.0))
                    total_count += float(getattr(child, "count", 0.0))
    return total_sum, total_count

def _metrics_snapshot(llm) -> Dict[str, float]:
    try:
        mets = llm.get_metrics()
    except Exception:
        return {"ttft_sum": 0.0, "ttft_cnt": 0.0, "tpot_sum": 0.0, "tpot_cnt": 0.0}
    ttft_sum, ttft_cnt = _collect_hist_sum_count(mets, TTFT_METRIC_NAME)
    tpot_sum, tpot_cnt = _collect_hist_sum_count(mets, TPOT_METRIC_NAME)
    return {"ttft_sum": ttft_sum, "ttft_cnt": ttft_cnt, "tpot_sum": tpot_sum, "tpot_cnt": tpot_cnt}
def _metrics_delta(before: dict, after: dict):
    return {
        "ttft_sum": after["ttft_sum"] - before["ttft_sum"],
        "ttft_cnt": after["ttft_cnt"] - before["ttft_cnt"],
        "tpot_sum": after["tpot_sum"] - before["tpot_sum"],
        "tpot_cnt": after["tpot_cnt"] - before["tpot_cnt"],
    }


def main(args: argparse.Namespace):
    print(args)
    req_max_len = _required_len_for(args.input_len, args.output_len)
    args.max_num_batched_tokens = max(req_max_len, args.batch_size, 16384)
    args.max_model_len = req_max_len
    print(f"max_num_batched_tokens: {args.max_num_batched_tokens}")


    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    # _ = xp.start_server(9012)

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion():
        start_time = time.perf_counter()
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    # Warmup
    print("Warming up...")
    snap_before = _metrics_snapshot(llm)
    warmup_latencies = []
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        warmup_latencies.append(run_to_completion())
    print(f"Average warmup latency: {np.mean(warmup_latencies):.4f}s")
    print(f"Tokens per second: {args.batch_size * args.output_len / np.mean(warmup_latencies):.2f}")
    snap_after = _metrics_snapshot(llm)
    delta = _metrics_delta(snap_before, snap_after)
    ttft = (delta["ttft_sum"] / delta["ttft_cnt"]) if delta["ttft_cnt"] > 0 else float("nan")
    avg_tpot = (delta["tpot_sum"] / delta["tpot_cnt"]) if delta["tpot_cnt"] > 0 else float("nan")
    decode_tps = (1.0 / avg_tpot) if avg_tpot > 0 else float("nan")
    print(f"TTFT (V1 metrics): {ttft:.4f} s")
    print(f"解码吞吐 (V1 metrics): {decode_tps:.2f} tok/s")
    # Profile
    profile_dir = args.profile_result_dir
    print(f"Profiling (results will be saved to '{profile_dir}')...")
    # Enable tracing on server
    # xp.trace_detached("localhost:9012",
    #                   profile_dir,
    #                   delay_ms=DELAY_MS,
    #                   duration_ms=DURATION_MS)
    if DELAY_MS == 0:
        time.sleep(1.0)
    profile_latencies = []
    snap_before = _metrics_snapshot(llm)
    for _ in tqdm(range(args.num_iters), desc="Profile iterations"):
        profile_latencies.append(run_to_completion())
    print(f"Average profile latency: {np.mean(profile_latencies):.4f}s")
    print(f"Tokens per second: {args.batch_size * args.output_len / np.mean(profile_latencies):.2f}")
    snap_after = _metrics_snapshot(llm)
    delta = _metrics_delta(snap_before, snap_after)
    ttft = (delta["ttft_sum"] / delta["ttft_cnt"]) if delta["ttft_cnt"] > 0 else float("nan")
    avg_tpot = (delta["tpot_sum"] / delta["tpot_cnt"]) if delta["tpot_cnt"] > 0 else float("nan")
    decode_tps = (1.0 / avg_tpot) if avg_tpot > 0 else float("nan")
    print(f"TTFT (V1 metrics): {ttft:.4f} s")
    print(f"解码吞吐 (V1 metrics): {decode_tps:.2f} tok/s")

    # save to json
    metrics = {}
    metrics["TTFT (V1)"] = ttft
    metrics["Decode TPS (V1)"] = decode_tps
    metrics["Tokens per second"] = args.batch_size * args.output_len / np.mean(profile_latencies)
    import json
    with open(os.path.join(profile_dir, f"metrics_{args.model}_{args.input_len}_{args.output_len}_{args.batch_size}.json"), "w") as f:
        json.dump(metrics, f)

    return


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=5,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=1,
                        help='Number of iterations to run for profiling.')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default="profiles",
        help=
        ('path to save the pytorch profiler output. Can be visualized '
         'with ui.perfetto.dev or Tensorboard '
         '(https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm).'
         ))

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)

# export XLA_HLO_DEBUG=1
# export MODEL=unsloth/Meta-Llama-3.1-70B-Instruct
# export VLLM_TPU_PROFILE_DURATION_MS=2000
# export VLLM_TPU_PROFILE_DELAY_MS=1000

# rm -rf ~/.cache/vllm/xla_cache
# # python3 tpu_profiling.py \
# #     --model $MODEL \
# #     --input-len 4096 \
# #     --output-len 128 \
# #     --batch-size 512 \
# #     --enforce-eager \
# #     --profile-result-dir profiles \
# #     --max-model-len 4352 --tensor-parallel-size 8

# # table 1:16384 tokens per batch
# python3 tpu_profiling.py \
#     --model $MODEL \
#     --input-len 4096 \
#     --output-len 128 \
#     --batch-size 512 \
#     --enforce-eager \
#     --max-num-batched-tokens 16384 \
#     --profile-result-dir profiles \
#     --max-model-len 4352 --tensor-parallel-size 8

# # table 2: 5600 tokens per batch
# python3 tpu_profiling.py \
#     --model $MODEL \
#     --input-len 5600 \
#     --output-len 83000 \
#     --batch-size 4 \
#     --enforce-eager \
#     --profile-result-dir profiles \
#     --max-model-len 4352 --tensor-parallel-size 8

# export XLA_HLO_DEBUG=1
# export MODEL=Qwen/Qwen2.5-7B-Instruct-1M
# export VLLM_TPU_PROFILE_DURATION_MS=3000
# export VLLM_TPU_PROFILE_DELAY_MS=0

# python3 tpu_profiling.py \
#     --model $MODEL \
#     --input-len 4096 --output-len 128 \
#     --batch-size 512 --enforce-eager \
#     --max-model-len 4352 \
#     --max-num-batched-tokens 16384 \
#     --tensor-parallel-size 1 \
#     --profile-result-dir profiles


# python3 tpu_profiling.py \
#     --model $MODEL \
#     --input-len 5600 --output-len 83000 \
#     --batch-size 4 --enforce-eager \
#     --tensor-parallel-size 4 \
#     --profile-result-dir profiles
import argparse
import dataclasses
import os
import time
from typing import List

import numpy as np
import torch_xla.debug.profiler as xp
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser

DURATION_MS = int(os.getenv("VLLM_TPU_PROFILE_DURATION_MS", 3000))
DELAY_MS = int(os.getenv("VLLM_TPU_PROFILE_DELAY_MS", 0))


def main(args: argparse.Namespace):
    print(args)

    # import torch_xla.core.xla_model as xm
    # device = xm.xla_device()
    # _ = xp.start_server(9012)
    # print all visible devices
    # print(f"Visible devices: {xm.get_visible_devices()}")
    # print(f"Using device: {device}")
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
        first_token_time = None
        
        # Use streaming to capture first token timing
        outputs = llm.generate(dummy_prompts,
                              sampling_params=sampling_params,
                              use_tqdm=False,
                              stream=True)
        
        # Process streaming outputs to find first token
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                first_output = output.outputs[0]
                if hasattr(first_output, 'token_ids') and first_output.token_ids:
                    # This is the first token - record the time
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    # Continue processing to completion
                    continue
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        
        return latency, ttft

    # Warmup
    print("Warming up...")
    warmup_latencies = []
    warmup_ttfts = []
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        latency, ttft = run_to_completion()
        warmup_latencies.append(latency)
        if ttft is not None:
            warmup_ttfts.append(ttft)
    print(f"Average warmup latency: {np.mean(warmup_latencies):.4f}s")
    if warmup_ttfts:
        print(f"Average warmup TTFT: {np.mean(warmup_ttfts):.4f}s")

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
    profile_ttfts = []
    for _ in tqdm(range(args.num_iters), desc="Profile iterations"):
        latency, ttft = run_to_completion()
        profile_latencies.append(latency)
        if ttft is not None:
            profile_ttfts.append(ttft)
    print(f"Average profile latency: {np.mean(profile_latencies):.4f}s")
    if profile_ttfts:
        print(f"Average profile TTFT: {np.mean(profile_ttfts):.4f}s")
        print(f"TTFT statistics:")
        print(f"  Min TTFT: {np.min(profile_ttfts):.4f}s")
        print(f"  Max TTFT: {np.max(profile_ttfts):.4f}s")
        print(f"  Std TTFT: {np.std(profile_ttfts):.4f}s")

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
# export MODEL=meta-llama/Llama-3.1-70B-Instruct
# export VLLM_TPU_PROFILE_DURATION_MS=2000
# export VLLM_TPU_PROFILE_DELAY_MS=1000

# rm -rf ~/.cache/vllm/xla_cache
# python3 profiling.py \
#     --model $MODEL \
#     --input-len 1 \
#     --output-len 128 \
#     --batch-size 32 \
#     --enforce-eager \
#     --profile-result-dir profiles \
#     --max-model-len 2048 --tensor-parallel-size 4




# export XLA_HLO_DEBUG=1
# export MODEL=Qwen/Qwen2.5-7B-Instruct
# export VLLM_TPU_PROFILE_DURATION_MS=3000
# export VLLM_TPU_PROFILE_DELAY_MS=0

# python3 profiling.py \
#     --model $MODEL \
#     --input-len 1024 --output-len 1 \
#     --batch-size 1 --enforce-eager \
#     --max-model-len 2048 \
#     --tensor-parallel-size 1 \
#     --profile-result-dir profiles
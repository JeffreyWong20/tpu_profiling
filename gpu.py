# 文件名: benchmark_single.py
import os
import gc
import time
import statistics
import argparse
from typing import List, Tuple, Dict
import contextlib
import torch
import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
# ========= 强制使用 vLLM V1 =========
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
# ========= 试图导入 V1 metrics 类型（兼容不同版本）=========
try:
    from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector  # type: ignore
except Exception:
    Counter = Gauge = Histogram = Vector = type("X", (), {})  # dummy
# ========= 配置（这些现在作为默认值，可以被 Shell 脚本覆盖）=========
DTYPE = "auto"
TP = 4
GPU_MEM_UTIL = 0.90
TRUST_REMOTE_CODE = True
SEED = 1234
TEMPERATURE = 0.0
TOP_P = 1.0
# ========= 场景定义 =========
# 注意：我们将场景名称简化为 'long' 和 'short'，以便于从命令行调用
SCENARIOS = {
    # Llama-3.3-70B
    "meta-llama/Llama-3.3-70B-Instruct_long": {
        "prompt_tokens": 5600, "max_new_tokens": 85300,
    },
    "meta-llama/Llama-3.3-70B-Instruct_short": {
        "prompt_tokens": 1024, "max_new_tokens": 128,
    },
    # GPT-OSS-20B
    "openai/gpt-oss-20b_long": {
        "prompt_tokens": 5600, "max_new_tokens": 85300,
    },
    "openai/gpt-oss-20b_short": {
        "prompt_tokens": 1024, "max_new_tokens": 128,
    },
    # RedHat-8B-Quantized
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8_long": {
        "prompt_tokens": 5600, "max_new_tokens": 85300,
    },
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8_short": {
        "prompt_tokens": 1024, "max_new_tokens": 128,
    },
}
# ========= 精确 token prompt 构造 (代码不变) =========
def build_exact_token_prompt(tokenizer, target_len: int) -> str:
    # ... (这部分代码与你原来的一样，这里省略以保持简洁)
    if target_len <= 1:
        ids = tokenizer("A", add_special_tokens=False)["input_ids"]
        if len(ids) >= 1:
            return tokenizer.decode(ids[:1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    base_text = ("You are a helpful assistant. Please analyze the following input and respond succinctly. ")
    chunk = " ".join(["data"] * 100) + ". "
    text = base_text + chunk * 200
    lo, hi = 0, len(text)
    target_ids = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ids = tokenizer(text[:mid], add_special_tokens=False)["input_ids"]
        if len(ids) == target_len:
            target_ids = ids
            break
        if len(ids) < target_len:
            lo = mid + 1
        else:
            hi = mid - 1
    if target_ids is None:
        ids = tokenizer(text[:lo], add_special_tokens=False)["input_ids"]
        if len(ids) > target_len:
            target_ids = ids[:target_len]
        else:
            filler = " data"
            while len(ids) < target_len:
                ids = tokenizer(tokenizer.decode(ids) + filler, add_special_tokens=False)["input_ids"]
            target_ids = ids[:target_len]
    prompt = tokenizer.decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    final_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(final_ids) != target_len:
        print(f"Warning: Tokenizer inconsistency. Target: {target_len}, Got: {len(final_ids)}. Truncating/padding.")
        if len(final_ids) > target_len:
            final_ids = final_ids[:target_len]
        else:
            final_ids.extend(final_ids[:target_len-len(final_ids)])
        prompt = tokenizer.decode(final_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) == target_len, "Prompt construction failed after retry."
    return prompt
# ========= V1 metrics 抽取工具 (代码不变) =========
# ... (所有 _iter_children_of_vector, _collect_hist_sum_count, _metrics_snapshot, _metrics_delta 函数保持不变)
TTFT_METRIC_NAME = "vllm:time_to_first_token_seconds"
TPOT_METRIC_NAME = "vllm:time_per_output_token_seconds"
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
# ========= 生成包装 (代码不变) =========
def decorated_generate(llm: LLM, prompts: List[str], params: SamplingParams):
    return llm.generate(prompts, params)
# ========= max_len 计算 (代码不变) =========
def _required_len_for(prompt_tokens: int, max_new_tokens: int) -> int:
    need = prompt_tokens + max_new_tokens
    return min(int(need * 1.05) + 256, 200_000)
# ========= LLM init / destroy (代码不变, 但现在更可靠) =========
def _init_llm(model_name: str, req_max_len: int, max_num_seqs: int) -> LLM:
    nvtx.range_push(f"LLM_init [{model_name}]")
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=TP,
            dtype=DTYPE,
            trust_remote_code=TRUST_REMOTE_CODE,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_num_seqs=max_num_seqs,
            max_model_len=req_max_len,
            max_num_batched_tokens=max(req_max_len, max_num_seqs, 16384),
            disable_log_stats=False,
        )
    finally:
        nvtx.range_pop()
    return llm
def _destroy_llm(llm):
    # 虽然脚本退出时OS会清理，但在脚本内部保持良好实践仍然是好的
    if llm is None:
        return
    with contextlib.suppress(Exception):
        if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "_shutdown_workers"):
            llm.llm_engine._shutdown_workers()
    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.2)
# ========= 主要执行逻辑 =========
def run_benchmark(model_name: str, scenario_name: str, batch_size: int):
    """
    执行单次基准测试
    """
    scenario_key = f"{model_name}_{scenario_name}"
    sc = SCENARIOS.get(scenario_key)
    if not sc:
        print(f"错误: 未找到模型 '{model_name}' 和场景 '{scenario_name}' 的组合。")
        return
    prompt_tokens = sc["prompt_tokens"]
    max_new_tokens = sc["max_new_tokens"]
    req_max_len = _required_len_for(prompt_tokens, max_new_tokens)
    print(f"\n===== 测试开始: Model={model_name}, Scenario={scenario_name}, BS={batch_size} =====")
    print(f"配置: prefill={prompt_tokens}, decode={max_new_tokens}, max_model_len={req_max_len}")
    llm = None
    try:
        # 1. 加载 Tokenizer 并构建 Prompt
        print("加载分词器中...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=TRUST_REMOTE_CODE)
        prompt_text = build_exact_token_prompt(tokenizer, prompt_tokens)
        print("分词器和 Prompt 准备就绪。")
        # 2. 初始化 LLM
        print(f"--- 初始化 LLM (bs={batch_size}) ---")
        llm = _init_llm(model_name, req_max_len, batch_size)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            ignore_eos=True,
            stop=None,
            stop_token_ids=[],
            min_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
            n=1,
        )
        # 3. 执行生成
        print(f"开始生成 (bs={batch_size})...")
        prompts = [prompt_text] * batch_size
        torch.cuda.synchronize()
        snap_before = _metrics_snapshot(llm)
        t0 = time.perf_counter()
        nvtx.range_push(f"generate [{model_name}]-[{scenario_name}] bs={batch_size}")
        outputs = decorated_generate(llm, prompts, sampling_params)
        nvtx.range_pop()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        snap_after = _metrics_snapshot(llm)
        duration = t1 - t0
        # 4. 计算并打印结果
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        avg_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs) / batch_size
        throughput = total_output_tokens / duration if duration > 0 else float("inf")
        delta = _metrics_delta(snap_before, snap_after)
        ttft = (delta["ttft_sum"] / delta["ttft_cnt"]) if delta["ttft_cnt"] > 0 else float("nan")
        avg_tpot = (delta["tpot_sum"] / delta["tpot_cnt"]) if delta["tpot_cnt"] > 0 else float("nan")
        decode_tps = (1.0 / avg_tpot) if avg_tpot > 0 else float("nan")
        print("\n--- 结果 ---")
        print(f"执行时间: {duration:.4f} s")
        print(f"实际平均输入 tokens: {avg_prompt_tokens:.2f} (目标 {prompt_tokens})")
        print(f"生成总 tokens: {total_output_tokens}")
        print(f"吞吐(生成tokens/秒): {throughput:.2f}")
        print(f"TTFT (V1 metrics): {ttft:.4f} s")
        print(f"解码吞吐 (V1 metrics): {decode_tps:.2f} tok/s")
        print("----------\n")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            print(f"\n[OOM] 在 bs={batch_size} 时发生 OOM。测试失败。")
        else:
            print(f"\n[异常] 在 bs={batch_size} 时发生运行时错误: {e}。测试失败。")
    except Exception as e:
        print(f"\n[未知错误] 在 bs={batch_size} 时发生未知错误: {e}。测试失败。")
    finally:
        # 5. 清理资源
        print("释放该实例的 LLM 与显存...")
        _destroy_llm(llm)
        print("清理完成。")
def main():
    parser = argparse.ArgumentParser(description="vLLM V1 单次基准测试脚本")
    parser.add_argument("--model", type=str, required=True, help="要测试的模型名称")
    parser.add_argument("--scenario_name", type=str, required=True, choices=["long", "short"], help="场景名称 ('long' or 'short')")
    parser.add_argument("--batch_size", type=int, required=True, help="要测试的批量大小")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("错误：需要 CUDA GPU。")
        return
    run_benchmark(args.model, args.scenario_name, args.batch_size)
if __name__ == "__main__":
    main()
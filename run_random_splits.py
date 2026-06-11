import jsonlines
import itertools
import random
import csv
import os
from datetime import datetime

from cascaded_evaluation.cascaded_classifier import CascadedClassifier
from cascaded_evaluation.util import merge_data
from model.task_spec import BINARY_TASK

# ============================================================
# 配置区 — 按需修改
# ============================================================
NUM_RUNS = 100          # 跑多少次随机split
CALIBRATION_SIZE = 400  # 每次calibration的样本数
TEST_SIZE = 180         # 每次test的样本数
ALPHA = 0.1
DELTA = 0.1

# 目标阈值（满足条件的run会被标记）
TARGET_COVERAGE = 0.5222
TARGET_ACC = 0.9255

# 模型名称
MODEL_NAMES = [
    "Qwen3.5-27B",
    "Qwen3.5-35B-A3B",
    "Qwen3.6-27B",
    "Qwen3.6-35B-A3B",
]

# 原始数据文件（calibration + test 合并后随机拆分）
CALIBRATION_FILENAMES = [
    "./result_605/Qwen3.5-27B-Q4_K_M.gguf.calibration.jsonl",
    "./result_605/Qwen3.5-35B-A3B-Q4_K_M.gguf.calibration.jsonl",
    "./result_605/Qwen3.6-27B-Q4_K_M.gguf.calibration.jsonl",
    "./result_605/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.calibration.jsonl",
]
TEST_FILENAMES = [
    "./result_605/Qwen3.5-27B-Q4_K_M.gguf.test.jsonl",
    "./result_605/Qwen3.5-35B-A3B-Q4_K_M.gguf.test.jsonl",
    "./result_605/Qwen3.6-27B-Q4_K_M.gguf.test.jsonl",
    "./result_605/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.test.jsonl",
]

# 输出CSV文件名（带时间戳避免覆盖）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"./results_random_splits_{timestamp}.csv"

# ============================================================
# 第一步：加载所有原始数据，合并calibration + test
# ============================================================
print("正在加载数据...")
all_samples = {}  # model_name -> list of samples (580条)

for model_name, cal_file, test_file in zip(MODEL_NAMES, CALIBRATION_FILENAMES, TEST_FILENAMES):
    samples = []
    with jsonlines.open(cal_file) as f:
        samples.extend(list(f))
    with jsonlines.open(test_file) as f:
        samples.extend(list(f))
    all_samples[model_name] = samples
    print(f"  {model_name}: 共加载 {len(samples)} 条样本")

print(f"\n数据加载完成。开始跑 {NUM_RUNS} 次随机split...\n")
print("=" * 70)

# ============================================================
# 第二步：所有排列组合（固定，不变）
# ============================================================
all_permutations = list(itertools.permutations(MODEL_NAMES))

# ============================================================
# 第三步：循环跑随机split
# ============================================================
all_results = []  # 收集所有结果，最后排序写CSV

for run_idx in range(1, NUM_RUNS + 1):
    print(f"\n[Run {run_idx}/{NUM_RUNS}]")

    # --- 随机split（所有模型用同一个index顺序，保证sample对齐）---
    total_size = CALIBRATION_SIZE + TEST_SIZE  # 580
    indices = list(range(total_size))
    random.shuffle(indices)
    cal_indices = sorted(indices[:CALIBRATION_SIZE])   # 取前400个index（排序便于记录）
    test_indices = sorted(indices[CALIBRATION_SIZE:])  # 剩余180个

    # 按index切出本次的calibration和test样本
    calibration_samples = {}
    test_samples = {}
    for model_name in MODEL_NAMES:
        src = all_samples[model_name]
        calibration_samples[model_name] = [src[i] for i in cal_indices]
        test_samples[model_name] = [src[i] for i in test_indices]

    # 记录sample_id顺序
    # 用第一个模型的数据来记录（所有模型sample_id一致）
    ref_model = MODEL_NAMES[0]
    cal_sample_ids = [s["sample_id"] for s in calibration_samples[ref_model]]
    test_sample_ids = [s["sample_id"] for s in test_samples[ref_model]]

    # --- 对每种排列跑一次 ---
    for perm in all_permutations:
        perm = list(perm)

        try:
            cascaded_classifier = CascadedClassifier(
                perm,
                calibration_samples=calibration_samples,
                alpha=ALPHA,
                delta=DELTA,
                task_type=BINARY_TASK,
            )

            evaluator_composition, selective_acc, coverage, evaluators = \
                cascaded_classifier.apply_decision_rule(test_samples, perm)

            meets_target = (coverage > TARGET_COVERAGE) and (selective_acc > TARGET_ACC)

            result_row = {
                "run": run_idx,
                "permutation": str(perm),
                "selective_acc": round(selective_acc, 6),
                "coverage": round(coverage, 6),
                "evaluated_samples": int(sum(1 for e in evaluators if e >= 0)),
                "evaluator_composition": str(evaluator_composition),
                "lambda_hats": str(cascaded_classifier.lambda_hats),
                "meets_target": meets_target,
                # sample_id列表转成用"|"分隔的字符串
                "calibration_sample_ids": "|".join(cal_sample_ids),
                "test_sample_ids": "|".join(test_sample_ids),
            }
            all_results.append(result_row)

            # 满足目标条件时在终端高亮提示
            if meets_target:
                print(f"  ★ 满足目标! 排列:{perm} | Acc:{selective_acc:.4f} | Coverage:{coverage:.4f}")

        except Exception as e:
            print(f"  [警告] Run {run_idx} 排列 {perm} 出错: {e}")
            continue

    print(f"  Run {run_idx} 完成，目前累计 {len(all_results)} 条结果")

# ============================================================
# 第四步：按coverage从高到低排序，写入CSV
# ============================================================
print("\n" + "=" * 70)
print("所有run完成，正在按coverage排序并写入CSV...")

all_results_sorted = sorted(all_results, key=lambda x: x["coverage"], reverse=True)

fieldnames = [
    "run",
    "permutation",
    "selective_acc",
    "coverage",
    "evaluated_samples",
    "meets_target",
    "evaluator_composition",
    "lambda_hats",
    "calibration_sample_ids",
    "test_sample_ids",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results_sorted)

print(f"结果已保存到: {OUTPUT_CSV}")
print(f"共 {len(all_results_sorted)} 条结果")

# 统计满足目标的条数
meets_count = sum(1 for r in all_results_sorted if r["meets_target"])
print(f"满足目标条件 (coverage>{TARGET_COVERAGE} 且 acc>{TARGET_ACC}) 的结果: {meets_count} 条")
print("=" * 70)

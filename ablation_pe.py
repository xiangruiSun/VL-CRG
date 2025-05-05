import os
import torch
import time
from pprint import pprint
from src.param import args
import src.config as config
from main import VQA, get_data_tuple


def run_spatial_positional_ablation():
    ablation_settings = [
        {"name": "Baseline", "use_spatial": False, "use_positional": False, "cost": 1.00},
        {"name": "Spatial Only", "use_spatial": True, "use_positional": False, "cost": 1.05},
        {"name": "Spatial + Positional", "use_spatial": True, "use_positional": True, "cost": 1.20},
    ]

    results = []

    for setting in ablation_settings:
        args.use_spatial = setting["use_spatial"]
        args.use_positional = setting["use_positional"]
        args.name = f"vqa_{setting['name'].replace(' ', '_').lower()}"
        args.output = os.path.join("results", args.name)
        os.makedirs(args.output, exist_ok=True)

        print(f"\n========== Running: {setting['name']} ==========\n")

        vqa = VQA()

        # Time the training for cost estimation (if needed)
        start_time = time.time()
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
        elapsed_time = time.time() - start_time

        # Evaluate
        eval_metrics = vqa.evaluate(vqa.valid_tuple)

        results.append({
            "setting": setting["name"],
            "Q→A": round(eval_metrics["Q→A"], 1),
            "QA→R": round(eval_metrics["QA→R"], 1),
            "Q→AR": round(eval_metrics["Q→AR"], 1),
            "consistency": round(eval_metrics["consistency"], 1),
            "sensitivity": round(eval_metrics["sensitivity"], 1),
            "cost": setting["cost"],  # Or use: round(elapsed_time / baseline_time, 2)
        })

    print("\n========== Ablation Study Results ==========")
    header = "{:<22} {:>6} {:>8} {:>8} {:>12} {:>12} {:>8}"
    row = "{:<22} {:>6} {:>8} {:>8} {:>12} {:>12} {:>8}"
    print(header.format("Setting", "Q→A", "QA→R", "Q→AR", "Consistency", "Sensitivity", "Cost"))
    for r in results:
        print(row.format(r["setting"], r["Q→A"], r["QA→R"], r["Q→AR"], r["consistency"], r["sensitivity"], r["cost"]))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_spatial_positional_ablation()

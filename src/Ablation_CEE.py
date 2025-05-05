import os
import torch
from pprint import pprint
from src.param import args
import src.config as config
from main import VQA, get_data_tuple


def run_ablation_study():
    results = {}

    for use_crg in [False, True]:
        # Set experiment name and CRG flag
        args.use_crg = use_crg
        args.name = f"vqa_crg_{'on' if use_crg else 'off'}"
        args.output = os.path.join("results", args.name)
        os.makedirs(args.output, exist_ok=True)

        print(f"\n========== Running {'with' if use_crg else 'without'} CRG ==========\n")

        # Initialize model and load pretrained if necessary
        vqa = VQA()

        # Train and evaluate
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
        acc, consistency, sensitivity = vqa.evaluate(vqa.valid_tuple)

        # Store results
        results[args.name] = {
            "accuracy": round(acc * 100, 2),
            "consistency": round(consistency * 100, 2),
            "sensitivity": round(sensitivity * 100, 2),
        }

    # Print results table
    print("\n========== Ablation Study Results ==========")
    print("{:<15} {:>10} {:>15} {:>15}".format("Setting", "Accuracy", "Consistency", "Sensitivity"))
    for name, metrics in results.items():
        print("{:<15} {:>10} {:>15} {:>15}".format(
            name, metrics["accuracy"], metrics["consistency"], metrics["sensitivity"]
        ))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_ablation_study()


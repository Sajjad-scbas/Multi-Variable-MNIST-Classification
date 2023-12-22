#!/usr/bin/env python3

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gt_path",
        type=str,
        help="The path of a ground truth annotation file",
    )

    parser.add_argument(
        "pred_path",
        type=str,
        help="The path of a file with predictions",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.gt_path) as f:
        gt = json.load(f)

    with open(args.pred_path) as f:
        predictions = json.load(f)

    assert set(gt.keys()) == set(predictions.keys()), \
        "The predicted samples do not match the ground truth samples"

    num_matches = 0
    for sample_name, sample_label in gt.items():
        if tuple(sample_label) == tuple(predictions[sample_name]):
            num_matches += 1

    print(f"Accuracy: {num_matches / len(gt):.2%}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import json
import argparse


def yml_to_json(input_path: str, output_path: str):
    # 1. 读取 YAML
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 基本 sanity check
    if not isinstance(data, list):
        raise ValueError("Top-level YAML structure must be a list of scenarios.")

    # 2. 写出 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,  # 保留中文
            indent=2             # 可读格式
        )

    print(f"Converted {len(data)} scenarios:")
    print(f"  YAML  -> {input_path}")
    print(f"  JSON  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Tidybot-style YAML dataset to JSON.")
    parser.add_argument("--input", required=True, help="Input YAML file (e.g., scenarios.yml)")
    parser.add_argument("--output", required=True, help="Output JSON file (e.g., scenarios.json)")
    args = parser.parse_args()

    yml_to_json(args.input, args.output)


if __name__ == "__main__":
    main()

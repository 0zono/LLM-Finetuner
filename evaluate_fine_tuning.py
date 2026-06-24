from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia modelo base ou adaptador LoRA")
    parser.add_argument("--config", default="config.yaml", help="Configuração do domínio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--test", default="data/output/latest/test.jsonl")
    parser.add_argument("--output", default="reports/model_evaluation.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Instale as dependências opcionais: pip install -e \".[training]\""
        ) from error

    from fine_tune import normalize_tool_calls
    from src.core.config import PipelineConfig
    from src.core.tool_registry import ToolRegistry

    pipeline_config = PipelineConfig.from_yaml(args.config)
    registry = ToolRegistry.from_file(pipeline_config.tools_file) if pipeline_config.tools_file else None
    tools = registry.as_openai_tools() if registry else []

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [
        json.loads(line)
        for line in Path(args.test).read_text(encoding="utf-8").splitlines()
        if line
    ]
    results = []
    for row in rows:
        messages = normalize_tool_calls(row["messages"])
        expected = extract_expected(messages[-1])
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predicted, json_valid = parse_tool_call(text)
        results.append(score_example(expected, predicted, json_valid, text))

    report = aggregate(results)
    report.update({"model": args.model, "adapter": args.adapter, "examples": results})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))
    return 0


def extract_expected(message: dict[str, Any]) -> dict[str, Any] | None:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    function = call.get("function", call)
    return {"name": function.get("name"), "arguments": function.get("arguments", {})}


def parse_tool_call(text: str) -> tuple[dict[str, Any] | None, bool]:
    candidate = text.strip().strip("`")
    if candidate.startswith("json"):
        candidate = candidate[4:].lstrip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        return None, False
    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None, False
    if "tool_calls" in data:
        data = data["tool_calls"][0]
    function = data.get("function", data)
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None, False
    return {"name": function.get("name"), "arguments": arguments}, True


def score_example(expected, predicted, json_valid: bool, raw: str) -> dict[str, Any]:
    expected_args = (expected or {}).get("arguments", {})
    predicted_args = (predicted or {}).get("arguments", {})
    expected_pairs = set(expected_args.items())
    predicted_pairs = set(predicted_args.items())
    matched = len(expected_pairs & predicted_pairs)
    return {
        "json_valid": json_valid,
        "json_required": expected is not None,
        "tool_correct": (expected or {}).get("name") == (predicted or {}).get("name"),
        "argument_tp": matched,
        "argument_fp": len(predicted_pairs) - matched,
        "argument_fn": len(expected_pairs) - matched,
        "exact_match": expected == predicted,
        "expected": expected,
        "predicted": predicted,
        "raw": raw,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    tp = sum(item["argument_tp"] for item in results)
    fp = sum(item["argument_fp"] for item in results)
    fn = sum(item["argument_fn"] for item in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    json_results = [item for item in results if item["json_required"]]
    return {
        "count": count,
        "json_valid_rate": (
            sum(item["json_valid"] for item in json_results) / len(json_results)
            if json_results
            else 0
        ),
        "tool_accuracy": sum(item["tool_correct"] for item in results) / count if count else 0,
        "argument_precision": precision,
        "argument_recall": recall,
        "argument_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "exact_match_rate": sum(item["exact_match"] for item in results) / count if count else 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())

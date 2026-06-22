from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA/QLoRA da prova de utilidade")
    parser.add_argument("--model", required=True, help="Modelo local ou identificador Hugging Face")
    parser.add_argument("--train", default="data/output/latest/train.jsonl")
    parser.add_argument("--validation", default="data/output/latest/validation.jsonl")
    parser.add_argument("--output", default="models/tcc-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit(
            "Instale as dependências opcionais: pip install -e \".[training]\""
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = None
    if args.qlora:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
    )
    train_dataset = load_dataset(Path(args.train), tokenizer)
    validation_dataset = load_dataset(Path(args.validation), tokenizer)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        logging_steps=5,
        eval_strategy="epoch" if len(validation_dataset) else "no",
        save_strategy="epoch",
        report_to="none",
        seed=42,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset if len(validation_dataset) else None,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    return 0


def load_dataset(path: Path, tokenizer):
    from datasets import Dataset
    from src.core.tool_registry import tool_schemas

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rendered = []
    for row in rows:
        if "messages" in row:
            messages = normalize_tool_calls(row["messages"])
            text = tokenizer.apply_chat_template(
                messages,
                tools=tool_schemas(),
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = (
                f"### Instrução\n{row['instruction']}\n"
                f"### Entrada\n{row.get('input', '')}\n"
                f"### Resposta\n{row['output']}"
            )
        rendered.append({"text": text})
    return Dataset.from_list(rendered)


def normalize_tool_calls(messages):
    normalized = []
    for message in messages:
        item = dict(message)
        if item.get("tool_calls"):
            item["tool_calls"] = [
                call
                if "function" in call
                else {
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in item["tool_calls"]
            ]
        normalized.append(item)
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())

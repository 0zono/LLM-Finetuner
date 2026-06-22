# LLM Fine-Tuning Dataset Pipeline

Pipeline modular para preparar, gerar, curar, validar e exportar datasets de
fine-tuning. Suporta `tool_calling`, `instruction_following` e `chat`; o estudo
experimental principal usa chamadas de ferramentas de suporte técnico.

## Instalação

```powershell
python -m venv venv
.\venv\Scripts\python -m pip install -e ".[dev]"
```

## Execução

```powershell
.\venv\Scripts\python main.py --config config.yaml
```

Os arquivos `train.jsonl`, `validation.jsonl`, `test.jsonl` e `invalid.jsonl`
são gravados no diretório configurado. O manifesto e a avaliação do validador
ficam no diretório de relatórios.

## LM local

A integração usa `POST /v1/chat/completions`, compatível com LM Studio, vLLM,
llama.cpp server e outros servidores OpenAI-compatible. Ajuste no YAML:

```yaml
enable_generation: true
curation_mode: llm
llm:
  enabled: true
  base_url: http://localhost:1234/v1
  model: nome-do-modelo-carregado
```

As respostas são armazenadas em cache por hash. O nome da variável da chave de
API é configurável; servidores locais sem autenticação usam o valor padrão.

## Tarefas

- `tool_calling`: sementes com `text`, `tool` e `arguments`.
- `instruction_following`: sementes com `instruction`, `input` opcional e `output`.
- `chat`: sementes com `messages` ou com `text` e `response`.

## Experimentos

```powershell
.\venv\Scripts\python experiment.py --config config.yaml
.\venv\Scripts\python -m pytest
```

O executor de ablação compara baseline, geração, curadoria heurística e
LLM-juiz. Para usar as configurações que chamam o modelo, mantenha o servidor
local ativo.

## Prova de utilidade com LoRA

Após gerar os splits, instale o grupo opcional e execute:

```powershell
.\venv\Scripts\python -m pip install -e ".[training]"
.\venv\Scripts\python fine_tune.py --model caminho-ou-modelo --qlora
.\venv\Scripts\python evaluate_fine_tuning.py --model caminho-ou-modelo --output reports/base.json
.\venv\Scripts\python evaluate_fine_tuning.py --model caminho-ou-modelo --adapter models/tcc-lora --output reports/ajustado.json
```

O treinamento fica deliberadamente separado da contribuição principal: ele
serve como prova de que o dataset exportado é consumível por um treinamento real.

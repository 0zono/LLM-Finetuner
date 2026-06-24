# LLM Fine-Tuning Dataset Pipeline

Pipeline modular para preparar, gerar, curar, validar e exportar datasets de
fine-tuning. Suporta `tool_calling`, `instruction_following` e `chat`. Domínios,
ferramentas e contratos são carregados por configuração, sem regras de negócio
no núcleo da aplicação.

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
llama.cpp server e outros servidores OpenAI-compatible:

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

## Domínios configuráveis

Um caso de `tool_calling` é definido por três artefatos:

```yaml
domain: suporte_tecnico
tools_file: domains/suporte_tecnico/tools.json
seed_path: data/seeds/suporte_tecnico.json
```

O arquivo de ferramentas contém nomes, descrições e parâmetros em JSON Schema.
Adicionar um domínio não exige alterar `src/`: basta criar o registro, as
sementes e um YAML.

O split é atribuído às sementes antes do aumento. Somente o conjunto de treino
recebe variações da LLM; validação e teste permanecem compostos por exemplos
humanos. Variações duplicadas, iguais à semente ou sem identificadores
obrigatórios são rejeitadas e registradas em `invalid.jsonl`.

Dois exemplos acompanham o projeto:

```powershell
# Exemplo 1: suporte técnico e chamados
.\venv\Scripts\python main.py --config config.yaml

# Exemplo 2: biblioteca
.\venv\Scripts\python main.py --config config.biblioteca.yaml
```

## Experimentos

```powershell
.\venv\Scripts\python experiment.py --config config.yaml --output reports/ablation/suporte.csv
.\venv\Scripts\python experiment.py --config config.biblioteca.yaml --output reports/ablation/biblioteca.csv
.\venv\Scripts\python -m pytest
```

O executor de ablação compara baseline, geração, curadoria heurística e
LLM-juiz. Para as configurações que chamam o modelo, mantenha o servidor local
ativo.

## Prova de utilidade com LoRA

Após gerar os splits, instale o grupo opcional e execute:

```powershell
.\venv\Scripts\python -m pip install -e ".[training]"
.\venv\Scripts\python fine_tune.py --config config.yaml --model caminho-ou-modelo --qlora
.\venv\Scripts\python evaluate_fine_tuning.py --config config.yaml --model caminho-ou-modelo --output reports/base.json
.\venv\Scripts\python evaluate_fine_tuning.py --config config.yaml --model caminho-ou-modelo --adapter models/tcc-lora --output reports/ajustado.json
```

O treinamento é separado da contribuição principal e serve como prova de que o
dataset exportado pode ser consumido por um treinamento real.

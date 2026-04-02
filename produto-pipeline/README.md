# Produto Pipeline v3

## 1. O que é este sistema

Pipeline de produto automatizado composto por **5 agentes CrewAI** que trabalham em sequência para transformar requisitos de produto em código pronto para revisão.

O pipeline cobre todo o ciclo: desde a análise de necessidades do utilizador até à geração de código e revisão de qualidade, integrando com Linear (gestão de projeto), Notion (documentação), GitHub (repositório) e modelos Claude/OpenAI.

```
Requisito → ProductThinker → SprintPlanner → PromptEngineer → VibeCoder → QAAgent → PR
```

## 2. Estrutura de pastas

```
produto-pipeline/
├── agents/                     # Definição dos 5 agentes CrewAI
│   ├── product_thinker.py      # Analisa requisitos, produz product brief
│   ├── sprint_planner.py       # Decompõe em tarefas, cria issues no Linear
│   ├── prompt_engineer.py      # Gera prompts otimizados para o VibeCoder
│   ├── vibe_coder.py           # Gera código e abre pull requests no GitHub
│   └── qa_agent.py             # Revê qualidade, valida critérios, aprova PR
├── context/                    # Memória e indexação de conhecimento
│   ├── memory.py               # Estado partilhado entre agentes na sessão
│   └── rag_index.py            # Índice RAG com LlamaIndex (GitHub + Notion)
├── crews/
│   └── produto_crew.py         # Orquestra os agentes numa Crew CrewAI v3
├── tasks/
│   └── produto_tasks.py        # Define as Task de cada etapa do pipeline
├── tests/                      # Testes unitários e de integração (pytest)
├── .env.example                # Template das variáveis de ambiente necessárias
├── requirements.txt            # Dependências Python do projeto
├── main.py                     # Ponto de entrada — valida env e inicia a Crew
└── README.md                   # Este ficheiro
```

### Responsabilidade de cada ficheiro

| Ficheiro | Responsabilidade |
|---|---|
| `agents/product_thinker.py` | Consome contexto de Linear/Notion e produz um brief estruturado |
| `agents/sprint_planner.py` | Estima esforço, define prioridades e cria issues no Linear |
| `agents/prompt_engineer.py` | Transforma specs técnicas em prompts otimizados para geração de código |
| `agents/vibe_coder.py` | Gera implementações completas com testes e abre PRs no GitHub |
| `agents/qa_agent.py` | Revisão estática, cobertura de testes e feedback estruturado |
| `context/memory.py` | Persiste estado da sessão e partilha contexto entre agentes |
| `context/rag_index.py` | Indexa docs e código existente para enriquecer o contexto dos agentes |
| `crews/produto_crew.py` | Monta a Crew, define sequência e estratégia de processo |
| `tasks/produto_tasks.py` | Declara cada Task com descrição, agente e critérios de aceitação |
| `main.py` | Valida configuração, imprime estado e invoca `ProdutoCrew.kickoff()` |

## 3. Como correr

### Pré-requisitos

- Python 3.11+
- Acesso à API da Anthropic e/ou OpenAI

### Instalação

```bash
cd produto-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuração

```bash
cp .env.example .env
# Edita .env e preenche as variáveis obrigatórias:
#   ANTHROPIC_API_KEY e GITHUB_TOKEN são obrigatórias
#   As restantes são opcionais consoante as integrações usadas
```

### Execução

```bash
python main.py
```

O pipeline valida as variáveis de ambiente, lista os agentes disponíveis e inicia `ProdutoCrew.kickoff()`.

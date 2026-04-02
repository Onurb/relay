import os
import sys

from dotenv import load_dotenv

REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "GITHUB_TOKEN"]

AGENTS = [
    "ProductThinkerAgent  — análise de requisitos e product brief",
    "SprintPlannerAgent   — decomposição em tarefas e issues no Linear",
    "PromptEngineerAgent  — engenharia de prompts para geração de código",
    "VibeCoderAgent       — geração de código e pull requests",
    "QAAgent              — revisão de qualidade e aprovação de PRs",
]


def validate_env() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        print(f"[ERRO] Variáveis de ambiente obrigatórias em falta: {', '.join(missing)}")
        print("Copia .env.example para .env e preenche os valores.")
        sys.exit(1)


def main() -> None:
    load_dotenv()
    validate_env()

    print("Pipeline v3 iniciado")
    print()
    print("Agentes que vão ser carregados:")
    for agent in AGENTS:
        print(f"  - {agent}")
    print()

    # TODO: substituir pelo kickoff real quando ProdutoCrew estiver implementada
    # from crews.produto_crew import ProdutoCrew
    # crew = ProdutoCrew()
    # resultado = crew.kickoff()
    # print(resultado)

    print("[AVISO] ProdutoCrew ainda não implementada — a executar em modo placeholder.")


if __name__ == "__main__":
    main()

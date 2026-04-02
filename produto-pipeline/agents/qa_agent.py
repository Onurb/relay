# TODO: implementar na semana 3 — QA Agent
from crewai import Agent


class QAAgent:
    """Agente de quality assurance. Analisa o código gerado pelo VibeCoder, executa
    revisão estática, verifica cobertura de testes, valida contra os critérios de
    aceitação definidos e aprova ou rejeita o pull request com feedback estruturado.
    """

    def build(self) -> Agent:
        """Constrói e devolve a instância CrewAI do agente."""
        pass

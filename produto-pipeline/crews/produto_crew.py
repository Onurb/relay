# TODO: implementar na semana 1 — ProdutoCrew
from crewai import Crew


class ProdutoCrew:
    """Orquestra os 5 agentes do pipeline de produto numa Crew CrewAI v3.
    Define a sequência de execução, partilha de contexto entre agentes e
    a estratégia de processo (sequencial ou hierárquica conforme a tarefa).
    """

    def build(self) -> Crew:
        """Monta e devolve a Crew com todos os agentes e tarefas configurados."""
        pass

    def kickoff(self, inputs: dict | None = None) -> str:
        """Inicia a execução do pipeline completo e devolve o resultado final."""
        pass

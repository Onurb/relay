# TODO: implementar na semana 2 — PipelineMemory
from typing import Any


class PipelineMemory:
    """Gestão de memória partilhada entre agentes ao longo do pipeline.
    Mantém o estado da sessão atual (brief, tarefas, artefactos gerados)
    e persiste contexto relevante para execuções futuras via LlamaIndex.
    """

    def store(self, key: str, value: Any) -> None:
        """Guarda um valor na memória do pipeline."""
        pass

    def retrieve(self, key: str) -> Any:
        """Recupera um valor previamente guardado."""
        pass

    def clear(self) -> None:
        """Limpa a memória da sessão atual."""
        pass

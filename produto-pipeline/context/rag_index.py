# TODO: implementar na semana 2 — RAGIndex
from typing import List


class RAGIndex:
    """Índice de Retrieval-Augmented Generation construído com LlamaIndex.
    Indexa documentação do produto, histórico de decisões arquiteturais e
    código existente para enriquecer o contexto fornecido aos agentes.
    """

    def build_from_github(self, repo: str) -> None:
        """Constrói o índice a partir do conteúdo de um repositório GitHub."""
        pass

    def build_from_notion(self, database_id: str) -> None:
        """Constrói o índice a partir de uma base de dados Notion."""
        pass

    def query(self, prompt: str, top_k: int = 5) -> List[str]:
        """Consulta o índice e devolve os k excertos mais relevantes."""
        pass

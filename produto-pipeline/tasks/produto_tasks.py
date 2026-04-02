# TODO: implementar na semana 1 — ProdutoTasks
from crewai import Task


class ProdutoTasks:
    """Define todas as tarefas do pipeline de produto para a Crew CrewAI v3.
    Cada método devolve uma Task configurada com descrição, agente responsável,
    critérios de aceitação e output esperado.
    """

    def product_analysis_task(self) -> Task:
        """Tarefa de análise de requisitos atribuída ao ProductThinkerAgent."""
        pass

    def sprint_planning_task(self) -> Task:
        """Tarefa de planeamento de sprint atribuída ao SprintPlannerAgent."""
        pass

    def prompt_engineering_task(self) -> Task:
        """Tarefa de engenharia de prompts atribuída ao PromptEngineerAgent."""
        pass

    def code_generation_task(self) -> Task:
        """Tarefa de geração de código atribuída ao VibeCoderAgent."""
        pass

    def qa_review_task(self) -> Task:
        """Tarefa de revisão de qualidade atribuída ao QAAgent."""
        pass

from langchain_ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI

class LLM:

    def __init__(self, provider: str = "ollama", model: str = "llama3:8b", temperature: float = 0):
        """Initializes the default model when an instance of LLM is created."""
        self.llm = self.init_llm(provider, model, temperature)

    def init_llm(self, provider: str, model: str, temperature: float = 0):
        """Initializes the LLM model with the specified provider."""
        if provider == "google":
            self.llm = ChatVertexAI(
                model=model,
                temperature=temperature,
            )
        elif provider == "ollama":
            self.llm = ChatOllama(
                model=model,
                temperature=temperature,
            )
        else:
            raise ValueError("Invalid provider. Must be 'google' or 'ollama'.")

        return self.llm

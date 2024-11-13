from agent.agent_state import AgentState
from llm.llm import LLM
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

class Task:

    TEMPLATE = """Eres un forofo del {team} y vas a tener conversación con un Fan de un equipo contrario, sé prudente.
    Responde con una frase o dos como mucho.
    Si ves que estáis en un punto cordial porque tiene razón, despídete con 'adiós'.
    Esto es lo que te está diciendo tu oponente:
    """

    def __init__(self):
        self.llm_ollama=LLM().llm
        self.llm_gemini=LLM(provider="google", model="gemini-pro").llm


    def call_BCN_llm(self, state: AgentState):
        prompt_template = PromptTemplate(
            template=self.TEMPLATE,
        ).format(team="Barcelona")
        messages = [SystemMessage(content=prompt_template), HumanMessage(content=state['messages'][-1])]
        message = self.llm_ollama.invoke(messages)
        print(f"Response from BCN LLM: {message.content}")
        print(f"--------------------")
        return {'messages': [message.content]}
    
    def call_RM_llm(self, state: AgentState):
        prompt_template = PromptTemplate(
            template=self.TEMPLATE,
        ).format(team="Real Madrid")
        messages = [SystemMessage(content=prompt_template), HumanMessage(content=state['messages'][-1])]
        message = self.llm_gemini.invoke(messages)
        print(f"Response from RM: {message.content}")
        print(f"--------------------")
        return {'messages': [message.content]}
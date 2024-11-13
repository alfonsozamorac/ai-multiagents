from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from agent.agent_state import AgentState
from tasks.tasks import Task


class GraphAgents:

    def __init__(self):
        memory = MemorySaver()
        builder=self._init_builder()
        self.graph = builder.compile(checkpointer=memory)

    def _init_builder(self) -> StateGraph:
        builder = StateGraph(AgentState)
        tasks = Task()
        builder.add_node('call_RM_llm', tasks.call_RM_llm)
        builder.add_node('call_BCN_llm', tasks.call_BCN_llm)
        builder.add_edge('call_RM_llm', 'call_BCN_llm')
        builder.add_conditional_edges('call_BCN_llm', 
                                      GraphAgents._stop_condition, {'end_conver': END, 'call_RM_llm': 'call_RM_llm'})
        builder.set_entry_point('call_RM_llm')
        return builder

    def _stop_condition(state: AgentState) -> str:
        result = state['messages'][-1]
        if "adios" in result.lower() or "adiós" in result.lower():
            return 'end_conver'
        return 'call_RM_llm'


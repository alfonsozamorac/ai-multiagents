from agent.graph_agents import GraphAgents
import uuid

def main():
    agent = GraphAgents()
    thread_id = str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    initial_message = "El Real Madrid no juega limpio."
    print("Initial Message:", initial_message)
    print(f"--------------------")
    agent.graph.invoke({'messages': [initial_message]}, config=config)

if __name__ == "__main__":
    main()
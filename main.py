from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

def main():
    model = ChatGroq(
        model="llama-3.1-8b-instant",  # fast and free
        temperature=0
    )

    tools = []

    agent_executor = create_agent(model=model, tools=tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        print("Assistant: ", end="")

        response = agent_executor.invoke(
            {"messages": [HumanMessage(content=user_input)]}
        )

        print(response["messages"][-1].content)
        print()

if __name__ == "__main__":
    main()
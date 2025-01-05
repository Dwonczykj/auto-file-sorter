
from openai import OpenAI
from mem0 import Memory


class CustomerSupportAIAgent:
    def __init__(self):
        """
        Initialize CustomerSupportAIAgent, configure memory and OpenAI client.
        """
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                    "max_tokens": 1500,
                }
            },
            "history_db_path": "history.db"
        }
        self.memory = Memory.from_config(config)
        self.client = OpenAI()
        self.app_id = "customer-support"

    def handle_query(self, query, user_id=None):
        """
        Handle customer query and store related information in memory.

        :param query: The customer query to process.
        :param user_id: Optional user ID to associate with memory.
        """
        # Send a streaming chat completion request to AI
        stream = self.client.chat.completions.create(
            model="gpt-4",
            stream=True,
            messages=[
                {"role": "system", "content": "You are a customer support AI agent."},
                {"role": "user", "content": query}
            ]
        )
        # Store the query in memory
        self.memory.add(query, user_id=user_id, metadata={"app_id":
                                                          self.app_id})

        # Print AI's response in real-time
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

    def get_memories(self, user_id=None):
        """
        Retrieve all memories associated with a given customer ID.

        :param user_id: Optional user ID to filter memories.
        :return: List of memories.
        """
        return self.memory.get_all(user_id=user_id)

    def get_memory(self, memory_id):
        """
        Retrieve a specific memory by its ID.
        :param memory_id: The ID of the memory to retrieve.
        :return: The memory object.
        """
        return self.memory.get(memory_id)


def example():
    # Instantiate CustomerSupportAIAgent
    support_agent = CustomerSupportAIAgent()

    # Define a customer ID
    customer_id = "learning-AI-from-scratch"

    # Handle customer query
    support_agent.handle_query(
        "I need help with my recent order. It hasn't arrived yet.", user_id=customer_id)

    memories = support_agent.get_memories(user_id=customer_id)
    for m in memories:
        print(m['text'])


if __name__ == "__main__":
    example()

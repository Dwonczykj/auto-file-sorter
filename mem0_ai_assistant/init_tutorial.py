import os
from mem0 import Memory

assert os.environ["OPENAI_API_KEY"], f"OPENAI_API_KEY is not set"
os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
#  https://levelup.gitconnected.com/how-i-build-an-agent-with-long-term-personalized-memory-54b7f4272d5f

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

m = Memory.from_config(config)

result = m.add("I am working on improving my tennis skills. 
Suggest some online courses.", user_id="alice", 
metadata={"category": "hobbies"})

all_memories = m.get_all()

# print(all_memories)

memory_id = all_memories[0]["id"]

specific_memory = m.get(memory_id)


related_memories = m.search(query="What are Alice's hobbies?", user_id="alice")



result = m.update(memory_id=memory_id, data="Likes to play tennis on weekends")

history = m.history(memory_id=memory_id)


m.delete(memory_id=memory_id)  
m.delete_all(user_id="alice")  
m.reset()

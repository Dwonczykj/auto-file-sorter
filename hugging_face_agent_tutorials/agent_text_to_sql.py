# # Agent for text-to-SQL with automatic error correction
# _Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_
#
# In this tutorial, we'll see how to implement an agent that leverages SQL using `transformers.agents`.
#
# What's the advantage over a standard text-to-SQL pipeline?
#
# A standard text-to-sql pipeline is brittle, since the generated SQL query can be incorrect. Even worse, the query could be incorrect, but not raise an error, instead giving some incorrect/useless outputs without raising an alarm.
#
# 👉 Instead, **an agent system is able to critically inspect outputs and decide if the query needs to be changed or not**, thus giving it a huge performance boost.
#
# Let's build this agent! 💪
# ## Setup SQL tables
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    insert,
    inspect,
    text,
)

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# create city SQL table
table_name = "receipts"
receipts = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("customer_name", String(16), primary_key=True),
    Column("price", Float),
    Column("tip", Float),
)
metadata_obj.create_all(engine)
rows = [
    {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
    {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
    {"receipt_id": 3, "customer_name": "Woodrow Wilson", "price": 53.43, "tip": 5.43},
    {"receipt_id": 4, "customer_name": "Margaret James", "price": 21.11, "tip": 1.00},
]
for row in rows:
    stmt = insert(receipts).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)
# Let's check that our system works with a basic query:
with engine.connect() as con:
    rows = con.execute(text("""SELECT * from receipts"""))
    for row in rows:
        print(row)
# ## Build our agent
#
# Now let's make our SQL table retrievable by a tool.
#
# The tool's `description` attribute will be embedded in the LLM's prompt by the agent system: it gives the LLM information about how to use the tool. So that is where we want to describe the SQL table.
inspector = inspect(engine)
columns_info = [(col["name"], col["type"]) for col in inspector.get_columns("receipts")]

table_description = "Columns:\n" + "\n".join(
    [f"  - {name}: {col_type}" for name, col_type in columns_info]
)
print(table_description)
# Now let's build our tool. It needs the following: (read [the documentation](https://huggingface.co/docs/transformers/en/agents#create-a-new-tool) for more detail)
# - A docstring with an `Args:` part
# - Type hints
from transformers.agents import tool


@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the table. Returns a string representation of the result.
    The table is named 'receipts'. Its description is as follows:
        Columns:
        - receipt_id: INTEGER
        - customer_name: VARCHAR(16)
        - price: FLOAT
        - tip: FLOAT

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    with engine.connect() as con:
        rows = con.execute(text(query))
        for row in rows:
            output += "\n" + str(row)
    return output
# Now let us create an agent that leverages this tool.
#
# We use the `ReactCodeAgent`, which is `transformers.agents`' main agent class: an agent that writes actions in code and can iterate on previous output according to the ReAct framework.
#
# The `llm_engine` is the LLM that powers the agent system. `HfEngine` allows you to call LLMs using HF's Inference API, either via Serverless or Dedicated endpoint, but you could also use any proprietary API: check out [this other cookbook](agent_change_llm) to learn how to adapt it.
from transformers.agents import ReactCodeAgent, HfApiEngine

agent = ReactCodeAgent(
    tools=[sql_engine],
    llm_engine=HfApiEngine("meta-llama/Meta-Llama-3-8B-Instruct"),
)
agent.run("Can you give me the name of the client who got the most expensive receipt?")
# ## Increasing difficulty: Table joins
#
# Now let's make it more challenging! We want our agent to handle joins across multiple tables.
#
# So let's make a second table recording the names of waiters for each `receipt_id`!
table_name = "waiters"
receipts = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("waiter_name", String(16), primary_key=True),
)
metadata_obj.create_all(engine)

rows = [
    {"receipt_id": 1, "waiter_name": "Corey Johnson"},
    {"receipt_id": 2, "waiter_name": "Michael Watts"},
    {"receipt_id": 3, "waiter_name": "Michael Watts"},
    {"receipt_id": 4, "waiter_name": "Margaret James"},
]
for row in rows:
    stmt = insert(receipts).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)
# We need to update the `SQLExecutorTool` with this table's description to let the LLM properly leverage information from this table.
updated_description = """Allows you to perform SQL queries on the table. Beware that this tool's output is a string representation of the execution output.
It can use the following tables:"""

inspector = inspect(engine)
for table in ["receipts", "waiters"]:
    columns_info = [(col["name"], col["type"]) for col in inspector.get_columns(table)]

    table_description = f"Table '{table}':\n"

    table_description += "Columns:\n" + "\n".join(
        [f"  - {name}: {col_type}" for name, col_type in columns_info]
    )
    updated_description += "\n\n" + table_description

print(updated_description)
# Since this request is a bit harder than the previous one, we'll switch the llm engine to use the more powerful [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)!
sql_engine.description = updated_description

agent = ReactCodeAgent(
    tools=[sql_engine],
    llm_engine=HfApiEngine("Qwen/Qwen2.5-72B-Instruct"),
)

agent.run("Which waiter got more total money from tips?")
# It directly works! The setup was surprisingly simple, wasn't it?
#
# ✅ Now you can go build this text-to-SQL system you've always dreamt of! ✨

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "ollama"

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="llama3.2-vision",base_url="http://127.0.0.1:11434/v1")

from langchain_core.messages import HumanMessage, SystemMessage

# messages = [
#     SystemMessage("Translate the following from English into chinese"),
#     HumanMessage("hi!,i'm a developer starter,what you can do for me?"),
# ]

# # print(model.invoke(messages))

# # print(model.invoke("Hello"))

# # print(model.invoke([{"role": "user", "content": "Hello"}]))

# # print(model.invoke([HumanMessage("Hello")]))

# for token in model.stream(messages):
#     print(token.content, end="|")

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "chinese", "text": "hi!"})

response = model.invoke(prompt)
print(response.content)
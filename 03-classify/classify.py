import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "ollama"

from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="llama3.2-vision",base_url="http://192.168.99.41:11434")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# tagging_prompt = ChatPromptTemplate.from_template(
#     """
# Extract the desired information from the following passage.

# Only extract the properties mentioned in the 'Classification' function.

# Passage:
# {input}
# """
# )


# class Classification(BaseModel):
#     sentiment: str = Field(description="The sentiment of the text")
#     aggressiveness: int = Field(
#         description="How aggressive the text is on a scale from 1 to 10"
#     )
#     language: str = Field(description="The language the text is written in")


# # LLM
# llm = ChatOpenAI(temperature=0, model="llama3-groq-tool-use",base_url="http://192.168.99.41:11434/v1").with_structured_output(
#     Classification
# )


# inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = llm.invoke(prompt)

# print(response)

# inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = llm.invoke(prompt)

# print(response.dict())

class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enucm=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )
    
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = ChatOpenAI(temperature=0, model="llama3-groq-tool-use",base_url="http://192.168.99.41:11434/v1").with_structured_output(
    Classification
)

# inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = llm.invoke(prompt)
# print(response)

inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)
print(response)

# inp = "Weather is ok here, I can go outside without much more than a coat"
# prompt = tagging_prompt.invoke({"input": inp})
# response = llm.invoke(prompt)
# print(response)

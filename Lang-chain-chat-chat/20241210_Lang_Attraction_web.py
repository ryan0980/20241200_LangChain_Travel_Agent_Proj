#!/usr/bin/env python
# coding: utf-8

# ## Step 1: 环境准备与配置

import os
from dotenv import load_dotenv
load_dotenv()  # 自动加载 .env 文件中配置的环境变量

# 验证环境变量是否正确加载（可选）
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LANGCHAIN_ENDPOINT:", os.getenv("LANGCHAIN_ENDPOINT"))
print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))

from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
import bs4
from langchain_community.document_loaders import WebBaseLoader
import getpass

# ## Step 2: 初始化LLM、嵌入及向量数据库

# 初始化LLM
llm = ChatOpenAI(temperature=0)

# 初始化embeddings与向量存储
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Or another model if preferred

vector_store = InMemoryVectorStore(embeddings)

# 定义意图分类函数
def intent_classification(question: str) -> bool:
    """
    使用LLM判断用户的问题是否与寻找酒店相关。
    简单示例：通过prompt让LLM回答“是”或“否”。
    实际中需更丰富的Prompt设计与测试。
    """
    prompt_text = f"用户的问题：{question}\n这个问题是否与寻找酒店相关？回答是或否。"
    result = llm.invoke(prompt_text)
    answer = result.content.strip()
    return "是" in answer  # 简单判断包含"是"即判断为酒店相关

# Load and chunk contents of the new "Things To Do" page
loader = WebBaseLoader(
    web_paths=("https://travel.usnews.com/Washington_DC/Things_To_Do/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("activity-description", "activity-title", "activity-header")  # Update based on actual classes
        )
    ),
)
docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")
if docs:
    print("Sample document content:")
    print(docs[0].page_content[:500])  # Print the first 500 characters

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into chunks
all_splits = text_splitter.split_documents(docs)

print(f"Number of chunks created: {len(all_splits)}")

# Index the document chunks
vector_store.add_documents(documents=all_splits)

print(f"Number of chunks indexed: {len(all_splits)}")

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Example query
response = graph.invoke({"question": "What are the top attractions in Washington DC?"})
print(response["answer"])

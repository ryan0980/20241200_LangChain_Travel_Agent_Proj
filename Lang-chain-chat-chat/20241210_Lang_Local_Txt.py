#!/usr/bin/env python
# coding: utf-8

# ## 导入必要的库

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage

# ## 步骤 1: 加载环境变量

load_dotenv()  # 自动加载 .env 文件中的环境变量

# 获取 OpenAI API 密钥
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

# ## 步骤 2: 初始化模型和嵌入

# 初始化 ChatOpenAI 模型
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# 初始化 OpenAI 嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

# ## 步骤 3: 加载并拆分本地文本文件

# 使用 TextLoader 加载本地文本文件
loader = TextLoader(file_path=r"C:\Users\ryan0\OneDrive\Obsidian_Lib\11_GWU\13_24FA\CSCI_6365_A_ML\20241113_AML_Final_Proj\data\attraction_resource\DC_attraction.txt", encoding="utf8")

try:
    docs = loader.load()
    print(f"加载的文档数量: {len(docs)}")
    if docs:
        print("示例文档内容:")
        print(docs[0].page_content[:500])  # 显示前500个字符
except Exception as e:
    print("加载文档时发生错误:", e)
    docs = []

# 仅在成功加载文档后继续
if docs:
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 将文档拆分为更小的块
    all_splits = text_splitter.split_documents(docs)
    print(f"创建的文本块数量: {len(all_splits)}")

    # 使用 FAISS 创建向量存储并索引文本块
    vector_store = FAISS.from_documents(all_splits, embeddings)
    print(f"已索引的文本块数量: {len(all_splits)}")

    # ## 步骤 4: 定义查询函数

    def answer_question(query):
        # 检索与查询最相关的文档块
        retrieved_docs = vector_store.similarity_search(query)

        # 将检索到的内容合并为一个字符串
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 构建与模型的对话消息
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Question: {query}\nContext: {context}\nAnswer:")
        ]

        # 获取模型的回答
        response = llm(messages)
        answer = response.content.strip()
        return answer

    # ## 步骤 5: 测试查询

    # 示例查询
    user_query = "where is Vietnam Veterans and Korean War Veterans Memorials？"
    answer = answer_question(user_query)
    print("\n用户查询:", user_query)
    print("模型回答:", answer)
else:
    print("没有文档可供处理。程序结束。")

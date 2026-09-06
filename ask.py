import os
from dotenv import load_dotenv

load_dotenv()  ***REMOVED*** 从项目根目录 .env 读取密钥，避免硬编码

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

***REMOVED*** ========== 需要修改的地方 ==========
API_KEY = os.getenv("API_KEY", "")  ***REMOVED*** 密钥统一放在 .env，不要写进代码

DB_PATH = "./chroma_db"
MODEL_NAME = "qwen3.5-flash"  ***REMOVED*** 通义千问 Turbo 版，便宜且快
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

print("正在加载向量数据库和模型，请稍候...")

***REMOVED*** 1. 加载入库时用的同一个嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})  ***REMOVED*** 每次检索最相关的 3 个片段

***REMOVED*** 2. 初始化大模型
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1  ***REMOVED*** 温度调低，让回答更严谨、少胡扯
)

***REMOVED*** 3. 定义 Prompt（提示词）模板
template = """请严格根据以下参考资料回答问题。如果资料中没有相关信息，请直接回答“抱歉，知识库中没有找到相关内容”，不要自己编造。

参考资料：
{context}

问题：{question}
"""
prompt = PromptTemplate.from_template(template)

***REMOVED*** 4. 开始对话
print("\n========== 知识库问答已启动 ==========")
print("输入 'quit' 退出\n")

while True:
    query = input("你: ")
    if query.lower() == 'quit':
        break
    
    ***REMOVED*** 从数据库检索相关片段
    docs = retriever.invoke(query)
    ***REMOVED*** 把检索到的内容拼成一段文本
    context = "\n\n".join([doc.page_content for doc in docs])
    
    ***REMOVED*** 组装提示词，发送给大模型
    final_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(final_prompt)
    
    print(f"\nAI: {response.content}\n")
import os, glob, fitz
from langchain_core.documents import Document                    
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PDF_DIR = r"C:\Users\86166\Desktop\Foundations-of-LLMs-main\《大模型基础》教材\《大模型基础》分章节内容"
MD_DIR  = r"C:\Users\86166\Desktop\llm-universe-main\markdown"
DB_PATH = "./chroma_db"

def load_pdfs(folder):
    docs = []
    for f in glob.glob(os.path.join(folder, "*.pdf")):
        pdf = fitz.open(f)
        for i, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": os.path.basename(f), "page": i+1}))
    return docs

def load_md(folder):
    docs = []
    for f in glob.glob(os.path.join(folder, "**/*.md"), recursive=True):
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read()
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(f)}))
    return docs

print("📖 加载PDF...")
pdf_docs = load_pdfs(PDF_DIR)
print(f"   共 {len(pdf_docs)} 个PDF页面")

print("📖 加载Markdown...")
md_docs = load_md(MD_DIR)
print(f"   共 {len(md_docs)} 个Markdown文件")

all_docs = pdf_docs + md_docs
print(f"✅ 总计 {len(all_docs)} 份文档")

print("✂️ 切片中...")
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = splitter.split_documents(all_docs)
print(f"✅ 共 {len(chunks)} 个文本块")

print("🧠 加载嵌入模型（首次会下载，耐心等待）...")
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

print("💾 写入向量数据库...")
db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
db.persist()
print(f"🎉 完成！数据库已保存到 {DB_PATH}")
# from langchain_community.document_loaders import FireCrawlLoader
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from loader import CustomMarkdownLoader

# loader = FireCrawlLoader(
#     url="https://docs.interop.io/desktop",
#     api_url="http://localhost:3002/",
#     mode="crawl",
# )

loader = DirectoryLoader(
    path="c:/work/git/docs-desktop/docs",
    glob="**/*.md",
    loader_cls=CustomMarkdownLoader,
)

docs = []
docs_lazy = loader.lazy_load()

# async variant:
# docs_lazy = await loader.alazy_load()

for doc in tqdm(docs_lazy, desc="Loading documents"):
    docs.append(doc)

print(docs[0].page_content[:100])
print(docs[0].metadata)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    collection_name="docs",
    persist_directory="./.chroma",
)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-4o-mini")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How do I create a feature metric in JS?")
print(answer)

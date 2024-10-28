# from langchain_community.document_loaders import FireCrawlLoader
import chromadb.errors
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_settings import SettingsConfigDict
from tqdm import tqdm
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore

from rag_demo.loader import CustomMarkdownLoader
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from chromadb import chromadb

# loader = FireCrawlLoader(
#     url="https://docs.interop.io/desktop",
#     api_url="http://localhost:3002/",
#     mode="crawl",
# )


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    openai_api_key: SecretStr
    chroma_persist_directory: str = "./.chroma"
    chroma_collection: str = "docs"
    docs_path: str


config = Config()


def setup_vector_store(config: Config) -> VectorStore:
    chroma_client = chromadb.PersistentClient(path=config.chroma_persist_directory)
    try:
        if chroma_client.get_collection(config.chroma_collection):
            return Chroma(
                embedding_function=OpenAIEmbeddings(),
                collection_name=config.chroma_collection,
                client=chroma_client,
            )
    except chromadb.errors.InvalidCollectionException:
        print(f"No document collection found. Populating from {config.docs_path}")
        pass

    loader = DirectoryLoader(
        path=config.docs_path,
        glob="**/*.md",
        loader_cls=CustomMarkdownLoader,
    )

    docs = []
    docs_lazy = loader.lazy_load()

    for doc in tqdm(docs_lazy, desc="Loading documents"):
        docs.append(doc)

    print(docs[0].page_content[:100])
    print(docs[0].metadata)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("Adding documents to vector db")

    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=OpenAIEmbeddings(),
        collection_name=config.chroma_collection,
    )
    vectorstore.add_documents(documents=splits)

    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


vectorstore = setup_vector_store(config)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How do I create a feature metric in JS?")
print(answer)

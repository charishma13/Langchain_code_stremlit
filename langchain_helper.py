import config

# Load docs
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import PromptTemplate

vectordb_file_path = "faiss_index"

def create_vector_db():

    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column = "prompt",csv_args={
            "delimiter": ",", "fieldnames": ["prompt", "response"],})

    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    vectordb = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    vectordb.save_local(vectordb_file_path)

def get_qa_chain():

    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # LLM
    llm = ChatOpenAI(temperature = 0.7)
    
    # Store splits

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

    PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    qa_chain = (
        {
            "context": vectordb.as_retriever(score_threshold=0.7),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain.invoke("Do you have javascript course ?"))

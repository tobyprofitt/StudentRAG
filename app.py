import streamlit as st
import os
import dotenv
dotenv.load_dotenv()

### LANGCHAIN IMPORTS ###
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

def load_model():
    with open('textbook_as_text.txt', 'r', encoding= 'utf-8') as f:
        text = f.read()

    loader = TextLoader(
        'textbook_as_text.txt',
        encoding='utf-8'
    )

    # Create vector embeddings for the textbook
    # docs = loader.load()
    # chunk_size, chunk_overlap = 1000, 200
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splits = text_splitter.split_documents(docs)

    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain



model = load_model()

def generate_response(query):
    return model.invoke(query)

st.title('RAG LLM Textbook Assistant')

user_query = st.text_input("Enter your query:")

if st.button('Generate Response'):
    if user_query:
        response = generate_response(user_query)
        st.write(response)
    else:
        st.write("Please enter a query to generate a response.")

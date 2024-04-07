import time
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os

# openai.api_key=os.getenv("OPENAI_API_KEY")
load_dotenv()
key = os.getenv("OPEN_AI_KEY")


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
@st.cache_resource(show_spinner=False)
def load_data(data_path: str):
    
    loader = DirectoryLoader(data_path, glob="**/*.docx")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=key), persist_directory="data", collection_name="chroma_data")

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "maximal_marginal_relevance": True})
    return retriever, len(docs), len(splits), vectorstore



retriever, len_docs, len_splits, vectorstore = load_data("docs")
st.success(f"Loaded {len_docs} docs and {len_splits} splits")

def get_chain():
    
    system_template = """Sadece aşağıdaki içeriğe göre cevap ver:
        {context}

        soru: {question}
        """
    
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
        
    prompt = ChatPromptTemplate.from_messages(messages)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=key)
    
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )   

    
    return retrieval_chain

chain = get_chain()

# Streamed response emulator
def response_generator(message):
    
    response = chain.invoke(message)
    
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Sorunu sor..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
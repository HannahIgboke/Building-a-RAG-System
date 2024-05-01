#importing all modules required
import streamlit as st
import nltk
nltk.download('punkt')
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

f = open("genai_apps/keys/gemini_api_key.txt")
key = f.read()

#setting up the headers
st.title('❓Query me about the "Leave No Context Behind paper by Google."')

#taking user input
user_prompt = st.text_area("What's your question?")

#if the button is clicked
if st.button("Query") == True:
  
    #loading the document
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader('Leave No Context Behind.pdf')
    pages = loader.load_and_split()

    #splitting the document into chunks
    from langchain_text_splitters import NLTKTextSplitter
    text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    #loading the API key and defining the embedding model
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, model = 'models/embedding-001')

    #storing the chunks in the chromadb vector store
    from langchain_community.vectorstores import Chroma

    #embedding each chunk and loading it into the vector store
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
    db.persist()

    #setting a connection with the ChromaDB
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

    #converting chroma db_connection to retriever object
    retriever = db_connection.as_retriever(search_kwargs={'k':5})

    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content = """You are a helpful AI bot.
        You take the context and question from the user.
        Your answer should be based on the specific context.
        """),
        HumanMessagePromptTemplate.from_template("""
        Answer the question based on the given context.
        Context: 
        {context}
        
        Question:
        {question}

        Answer:
        """)                                              
    ])

    #defining the chat_model of choice
    chat_model = ChatGoogleGenerativeAI(google_api_key=key, 
                                    model="gemini-1.5-pro-latest")

    #cereating output parser
    output_parser = StrOutputParser()

    #creating the lag chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {'context':retriever | format_docs, 'question': RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )


    #if the prompt is provided
    if user_prompt:
        response = rag_chain.invoke(user_prompt)
        
        #printing the response on the webpage
        st.write(response)
import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import S3FileLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.memory import PostgresChatMessageHistory
import boto3
import tempfile
import time
import hashlib
import secrets
import os
from dotenv import load_dotenv
import logging
logging.getLogger('botocore').setLevel(logging.ERROR)


def styled_header(text):
    header_html = f"""
    <div style="background-color:#4CAF50;text-align:center;padding:10px">
    <h1 style="color:white;text-align:center;">{text}</h1>
    </div>
    """
    return header_html


def styled_subheader(text, font_size="24px", color="#8A2BE2", background_color="#f0e5ff"):
    subheader_html = f"""
    <div style="box-shadow:0 2px 10px #ddd; padding: 5px; background-color: {background_color}; border-radius: 5px; margin: 10px 0; text-align: center;">
        <h3 style="color: {color}; font-size: {font_size}; margin: 0;">{text}</h3>
    </div>
    """
    return subheader_html


def generate_session_id():
    t = int(time.time() * 1000)
    r = secrets.randbelow(1000000)
    return hashlib.md5(bytes(str(t) + str(r), 'utf-8'), usedforsecurity=False).hexdigest()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=512,
        chunk_overlap=103,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", region_name="us-east-1")
    try:
        if text_chunks is None:
            return PGVector(
                connection_string=CONNECTION_STRING,
                embedding_function=embeddings,
            )
        return PGVector.from_texts(texts=text_chunks, embedding=embeddings, connection_string=CONNECTION_STRING)
    except Exception as e:
        raise e  # Raise the exception to see the actual error


def get_bedrock_llm(selected_llm):
    print(f"[INFO] Selected LLM is : {selected_llm}")
    if selected_llm in ['amazon.titan-text-express-v1']:
        llm = Bedrock(model_id=selected_llm, model_kwargs={
                      'max_tokens_to_sample': 4096})

    elif selected_llm in ['amazon.titan-tg1-large', 'amazon.titan-text-express-v1']:
        llm = Bedrock(
            model_id=selected_llm,
            model_kwargs={
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1,
            }
        )
    else:
        raise ValueError(f"Unsupported LLM: {selected_llm}")

    return llm


def get_conversation_chain(vectorstore, selected_llm):
    # llm = Bedrock(model_id="anthropic.claude-instant-v1",region_name="us-east-1")
    llm = get_bedrock_llm(selected_llm)
    _connection_string = CONNECTION_STRING.replace(
        '+psycopg2', '').replace(':5432', '')
    message_history = PostgresChatMessageHistory(
        connection_string=_connection_string,
        session_id=generate_session_id()
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=message_history, return_source_documents=True, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def color_text(text, color="black"):
    return f'<span style="color:{color}">{text}</span>'

bot_template = "🤖 BOT : {0}"
user_template = "👤 USER : {0}"


def handle_userinput(user_question):
    bot_template = "🤖 BOT : {0}"
    user_template = "👤 USER : {0}"
    try:
        response = st.session_state.conversation({'question': user_question})
        st.markdown(color_text(user_template.format(
            response['question']), color="blue"), unsafe_allow_html=True)
        st.markdown(color_text(bot_template.format(
            response['answer']), color="green"), unsafe_allow_html=True)
        print("Response", response)
    except ValueError as e:
        st.write(e)
        st.write("😞 Sorry, please ask again in a different way.")
        return
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(color_text(user_template.format(
                message.content), color="blue"), unsafe_allow_html=True)

        else:
            st.markdown(color_text(bot_template.format(
                message.content), color="green"), unsafe_allow_html=True)


def main():
    # Updated header styling
    st.markdown(styled_header(
        "Prueba de concepto Amazon Bedrock 📚🦜"), unsafe_allow_html=True)

    options = ["📄 PDFs",  "📑 CSV"]
    # Using radio button instead of select box
    st.markdown(styled_subheader("📌 Fuente de información 📌"),
                unsafe_allow_html=True)
    selected_source = st.radio("", options)

    # Add LLM selection UI
    st.markdown(styled_subheader("🤖 Selecciona un modelo LLM 🤖"), unsafe_allow_html=True)
    llm_options = [
        'amazon.titan-tg1-large',
        'amazon.titan-text-express-v1'

    ]

    selected_llm = st.radio("Choose an LLM", options=llm_options)

    # PDF Section
    if selected_source == "📄 PDFs":
        pdf_docs = st.file_uploader(
            "📥 Upload your PDFs here:", type="pdf", accept_multiple_files=True)
        if st.button("🔄 Process"):
            with st.spinner("🔧 Entrenando mi base de datos de Vectores"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore is None:
                    st.write("Failed to initialize vector store.")
                    return
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, selected_llm)

    elif selected_source == "📑 CSV":
        csv_docs = st.file_uploader(
            "Upload your CSV here and click on 'Process'", type="csv", accept_multiple_files=False)
        if st.button("Process"):
            with st.spinner("Entrenando mi base de datos de Vectores"):
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(csv_docs.getvalue())
                    tmp_file_path = tmp_file.name
                loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                    'delimiter': ','})
                docs = loader.load()
                for i in docs:
                    text_chunks = get_text_chunks(i.page_content)
                    vectorstore = get_vectorstore(text_chunks)
                    if vectorstore is None:
                        st.write("Failed to initialize vector store.")
                        return
                print("información cargada correctamente")
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, selected_llm)

    st.sidebar.header("🗣️ Chat with Bot")
    user_question = st.sidebar.text_input("💬 Ask a question about your data:")
    if user_question:
        handle_userinput(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(
            get_vectorstore(None), selected_llm)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


# enter the appropriate DB name
if __name__ == '__main__':
    load_dotenv()

    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER"),
        user=os.environ.get("PGVECTOR_USER"),
        password=os.environ.get("PGVECTOR_PASSWORD"),
        host=os.environ.get("PGVECTOR_HOST"),
        port=os.environ.get("PGVECTOR_PORT"),
        database=os.environ.get("PGVECTOR_DATABASE")
    )

    main()

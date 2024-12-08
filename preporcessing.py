import os
import streamlit as st
import fitz
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import KNNRetriever

load_dotenv()

def load_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [Document(page_content=pdf_document.load_page(i).get_text(), metadata={"page": i}) for i in range(pdf_document.page_count)]

def embed_documents(documents):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(documents, embedding=embeddings)

def retrieve_answer(question, vector_store):
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    return context_text

def main():
    st.set_page_config(page_title="RAG Document Chat", page_icon=":books:")
    st.header("Document Embedding and Retrieval")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process PDFs") and pdf_files:
            documents = [doc for pdf in pdf_files for doc in load_pdf(pdf)]
            st.session_state.vector_store = embed_documents(documents)
            st.success("PDFs processed successfully!")

    user_question = st.text_input("Ask a question about the document:")
    if user_question and st.session_state.vector_store:
        answer = retrieve_answer(user_question, st.session_state.vector_store)
        st.write("Answer:", answer)
    elif user_question:
        st.warning("Please upload and process a PDF document first.")

if __name__ == "__main__":
    main()
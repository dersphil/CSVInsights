import streamlit as st
import fitz  
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from groq import Groq  
from langchain.llms import OpenAI
import pandas as pd
from io import StringIO

load_dotenv()

def query_llm_api(user_input):
    client = Groq()
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": f"""Based on csv data given, analyze and answer questions according to data. Question from user is: {user_input}
                               According to data given, answer the above questions"""
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False  
    )
    
    response_content = completion.choices[0].message.content
    return response_content

def custom_qa_system(question, vector_store, language):
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)

    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    combined_input = f"Context:\n{context_text}\n\nQuestion:\n{question}"

    response = query_llm_api(combined_input, language)
    return response


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def process_file(pdf_files):
    documents = []
    for pdf in pdf_files:
        pdf_document = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            documents.append(Document(page_content=page_text, metadata={"page": page_num}))

    # Specify persist directory
    vector_store = Chroma.from_documents(documents, embedding=embeddings, persist_directory="chroma_db")
    vector_store.persist()  # Ensures database tables are created
    return vector_store

# Streamlit App
def process_csv_and_create_agent(csv_file):
    df = pd.read_csv(csv_file)
    csv_agent = create_csv_agent(
        OpenAI(temperature=0), 
        csv_file, 
        verbose=True
    )
    return csv_agent

def main():
    st.set_page_config(page_title="Chat with CSV", page_icon=":books:")
    st.header("ConvoTutor - CSV Agent")

    if "csv_agent" not in st.session_state:
        st.session_state.csv_agent = None

    with st.sidebar:
        st.subheader("Upload CSV File")
        csv_file = st.file_uploader("Upload CSV file", type="csv")
        
        if csv_file:
            with st.spinner("Processing CSV file..."):
                st.session_state.csv_agent = process_csv_and_create_agent(csv_file)
            st.success("CSV file processed and agent created!")

    user_question = st.text_input("Ask a question about the CSV data:")

    if user_question and st.session_state.csv_agent:
        with st.spinner("Querying the CSV data..."):
            answer_csv = st.session_state.csv_agent.run(user_question)
        st.write("Answer from CSV data:", answer_csv)

    elif user_question:
        st.warning("Please upload and process a CSV file first.")

if __name__ == "__main__":
    main()
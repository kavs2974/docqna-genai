import streamlit as st
from utils import load_document, create_chunks, embed_and_store
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

st.set_page_config(page_title="📄 GenAI Doc QnA", layout="centered")
st.title("📄 Document-based Question Answering System (GenAI)")
st.markdown("Upload a PDF or Word document and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'docx'])

if uploaded_file:
    with st.spinner("Reading document..."):
        raw_text = load_document(uploaded_file)

    with st.spinner("Chunking and embedding..."):
        chunks = create_chunks(raw_text)
        vectorstore = embed_and_store(chunks)

    st.success("Document processed. You can now ask questions!")

    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Generating answer..."):
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )

            result = qa_chain({"query": query})
            st.markdown("### ✅ Answer")
            st.write(result["result"])
try:
    st.set_page_config(page_title="📄 Doc QnA")
    st.title("📄 Document-based QnA System")
    # ... your full Streamlit app code here ...

except Exception as e:
    st.error("❌ Something went wrong")
    st.code(traceback.format_exc())
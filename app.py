import streamlit as st
from utils import load_document, create_chunks, embed_and_store
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import traceback

load_dotenv()

st.set_page_config(page_title="📄 GenAI Doc QnA", layout="centered")
st.title("📄 Document-based Question Answering System (GenAI)")
st.markdown("Upload a PDF or Word document and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx"])

if uploaded_file:
    try:
        with st.spinner("Reading document..."):
            raw_text = load_document(uploaded_file)

        with st.spinner("Chunking and embedding..."):
            chunks = create_chunks(raw_text)
            vectorstore = embed_and_store(chunks)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        st.success("✅ Document processed! Ask your questions below:")

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain.run(query)
                st.markdown(f"**Answer:** {result}")

    except Exception as e:
        st.error("An error occurred while processing the document.")
        st.exception(traceback.format_exc())

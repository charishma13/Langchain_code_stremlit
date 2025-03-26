import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.title("Codebasics QA")
bnb = st.button("Create Knowledgebase")
if bnb:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke(question)

    st.header("Answer")
    st.write(response)
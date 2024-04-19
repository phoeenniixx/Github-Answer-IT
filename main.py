import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from help import *
local = False

if local:
    from dotenv import load_dotenv
    load_dotenv()
if local:
    user_key = os.environ.get('OPENAI_API_KEY')
else:
    user_key = st.text_input("Enter your OpenAI Key", "")
if user_key:
    os.environ['OPENAI_API_KEY'] = user_key
    st.title("Answer IT")
    st.sidebar.title("Text Section")

    git_link = st.sidebar.text_area("Enter the GitHub Repo:","https://github.com/phoeenniixx/Image-captioning")
    helper = Helper(git_link)
    st.write("Downloading Models")
    clone_path = helper.clone()
    placeholder = st.empty()

    query = st.text_input("Question: ")
    process_text = st.button("Ask Question")
    if git_link:

        st.write("Downloading Models")

        helper.extract_all_files()
        helper.chunk_files()
        store_name = helper.last_name
        chunks = helper.texts_combined
        st.write(len(chunks))

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)

        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

    if process_text:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)

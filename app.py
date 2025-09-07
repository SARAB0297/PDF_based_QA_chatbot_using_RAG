
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##title setup
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDF's and ask query from its content")

#import GROQ API Key
api_key=st.text_input("Enter your Groq API Key:",type="password")

#check GROQ API Key
if api_key:
    llm=ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    #chat interface
    session_id=st.text_input("Session_ID",value="default_session")

    #manage chat history with states

    if 'store' not in st.session_state:
        st.session_state.store= {}

    uploaded_files=st.file_uploader("Choose one or multiple PDF files to upload", type="pdf",accept_multiple_files=True)

    #process upload PDFs
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb")as file:
                file.write( uploaded_file.getvalue())
                file_name=uploaded_file.name
            
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
#splitting and creating embeddings
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        TOP_K = 5  # small, interview-friendly number
        show_diag = st.checkbox("Show retrieval quality", value=False)
        conf= None
        pairs=[]
        def retrieve_with_scores(query: str, k: int = TOP_K):   
    # Chroma returns (Document, relevance_score) with score in [0,1], higher = better
            return vectorstore.similarity_search_with_relevance_scores(query, k=k)

        def retrieval_confidence(pairs, faiss_mode=False):
    # simple, explainable metric = best relevance score
            if not pairs:
                return 0.0
            scores = [s for _, s in pairs]
            if faiss_mode:
        # convert distance → similarity in [0,1]
                sims = [1 - s for s in scores]
                return max(sims)
            else:
                return max(scores)



        contextualize_q_system_prompt=(
               "Given a chat history and the latest user question"
            "which might reference context in chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. DO NOT answer the question,"
            "just reformulate it if needed and else return as it is."
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        #Answer Question
        system_prompt=(
                "You are an assistant for question answering tasks."
                "Use the following pieces of retrieved context to answer the question."
                " If you don't know the answer, say that you don't know."
                "Use three sentences maximum and keep the answers concise."
                "\n\n"
                "{context}"
            )
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
    
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"

        )
        user_input=st.text_input("Your questions:")
        if user_input:
    # 1) Diagnose retrieval quality for THIS query (scores come from Chroma)
            pairs = retrieve_with_scores(user_input, k=TOP_K)
            conf = retrieval_confidence(pairs)

    # 2) Run your existing conversational RAG chain (no change)
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.write("Assistant:", response["answer"])

    # 3) (Optional) Show top-k with scores on demand
        if show_diag:
            st.caption(f"Retrieval confidence: {conf:.2f} (max relevance score in top-{TOP_K})")
            rows = []
            for rank, (doc, score) in enumerate(pairs, 1):
                rows.append(f"{rank}. score={score:.2f} · source={doc.metadata.get('source','')} · page={doc.metadata.get('page','')}")
            st.text("\n".join(rows))

           

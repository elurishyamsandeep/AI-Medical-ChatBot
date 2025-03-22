import os
import streamlit as st

# Use the new import location for HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Path to your local FAISS vectorstore
DB_FAISS = r"D:\gopractice\New folder\chatmodel\vectorstore\db_fiass"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def main():
    st.title("Ask Chatbot!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    # Display past chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    # Chat input
    user_prompt = st.chat_input("Pass your prompt here")
    
    if user_prompt:
        st.chat_message('user').markdown(user_prompt)
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})
        
        # Define your custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
                
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            # Invoke the QA chain and display only the result text
            response = qa_chain.invoke({'query': user_prompt})
            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

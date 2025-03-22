from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
HF_TOKEN= os.environ.get("HF_TOKEN")
huggingface_repoid="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repoid):
    llm = HuggingFaceEndpoint(
    repo_id=huggingface_repoid,
    temperature=0.3,
    model_kwargs={"token": HF_TOKEN, "max_length": "512"}
)

    return llm

DB_FAISS = r"D:\gopractice\New folder\chatmodel\vectorstore\db_fiass"


custom_prompt_template="""
use the pieces of information provided in the context to answer user's question 
if you don't know the answer ,just say that you don't know ,don't try to make up an answer
.Don't provide anything out of the given context.

Context:{context}
Question:{question}

Strat the answer directly .No small talk please.

"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

custom_prompt = set_custom_prompt(custom_prompt_template)

embeeding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


db=FAISS.load_local(DB_FAISS,embeeding_model,allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repoid),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': custom_prompt}
)




#invove with single query
user_query = input("Write query here: ")
response = qa_chain.invoke({'query': user_query})

# Print only the result text
print("Response:", response.get("result", "No result returned"))

# Optionally, print a short summary for each source document
if "source_documents" in response:
    print("\nSource Documents (first 100 characters):")
    for i, doc in enumerate(response["source_documents"], start=1):
        summary = doc.page_content.strip().replace("\n", " ")[:100]
        print(f"Doc {i}: {summary}...")


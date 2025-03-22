
```markdown
# RAG-based AI Medical Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) AI medical chatbot designed to convert medical textbooks into a searchable vector database. The chatbot leverages cutting-edge NLP techniques and integrates the Mistral-7B-Instruct language model from HuggingFace to generate contextually accurate responses, while also providing source references from the original book content.

## Features

- **Knowledge Extraction & Indexing:**  
  Converts medical textbook content into vector representations using HuggingFace embeddings and FAISS, enabling rapid retrieval of relevant information.

- **Mistral LLM Integration:**  
  Utilizes the `mistralai/Mistral-7B-Instruct-v0.3` language model to generate accurate, context-aware answers based on the indexed data.

- **Source Attribution:**  
  Every answer is accompanied by references to the source information in the original book, ensuring transparency and traceability.

- **Interactive Web Interface:**  
  Built with Streamlit, the chatbot offers a user-friendly interface for real-time, conversational interactions.

- **Scalable & Adaptable:**  
  Designed for easy integration into larger medical knowledge systems and for future enhancements.

## Installation

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Hub](https://huggingface.co/)

### Setup Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   cd <repository-name>
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   Ensure your `requirements.txt` includes all necessary packages such as:
   - streamlit
   - langchain
   - faiss-cpu
   - langchain_community
   - python-dotenv
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   - Create a `.env` file in the root directory.
   - Add your HuggingFace token:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```

## Usage

1. **Indexing Medical Textbooks:**
   - Place your PDF textbooks in the designated data folder.
   - Use the provided data processing script to convert the content into a FAISS vector store.

2. **Running the Chatbot:**
   Launch the chatbot interface using Streamlit:
   ```bash
   streamlit run "d:\gopractice\New folder\chatmodel\chatbot.py"
   ```
   Open the local URL provided by Streamlit in your browser, and start interacting with the chatbot.

## Project Structure

```
├── chatbot.py            # Main Streamlit application file
├── vectorstore/          # Directory for the FAISS vector store
├── data/                 # Directory containing the PDF textbooks
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables file (not tracked by Git)
└── README.md             # This file
```


## Acknowledgements

- [HuggingFace](https://huggingface.co/)
- [LangChain](https://python.langchain.com/)
- [Streamlit](https://streamlit.io/)  
- [FAISS](https://github.com/facebookresearch/faiss)
```

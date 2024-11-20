# Intelligent PDF Query System Using Google’s Gemini-Pro LLM

## Overview

The **Intelligent PDF Query System** allows users to interact with multiple PDF documents through natural language queries. By leveraging Google’s Gemini-Pro Large Language Model (LLM), this system simplifies the process of extracting and querying information from unstructured data in PDF format.

## Features

- **Upload Multiple PDFs:** Users can upload multiple PDF files for processing.
- **Natural Language Queries:** Users can ask questions related to the content of the PDFs.
- **Context-Aware Responses:** The system provides detailed answers based on the content of the uploaded documents.

## Technologies Used

- **Streamlit:** For creating the web interface.
- **PyPDF2:** For reading PDF files.
- **LangChain:** For managing language model interactions.
- **FAISS:** For creating a vector store for fast similarity searches.
- **Google Generative AI:** For leveraging embeddings and conversational capabilities.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Abhiram-23/genAI_Gemini
   cd genAI_Gemini
   ```

2. **Install Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**
   - Create a `.env` file in the project root and add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

1. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF Files:**
   - Use the file uploader in the sidebar to select and upload your PDF documents.

3. **Ask Questions:**
   - Enter your question in the provided text input box and press "Enter" to receive a context-aware answer.

## Code Overview

Here is a brief description of the main components of the code:

### PDF Processing

```python
async def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```
This function reads the uploaded PDF files and extracts their text content.

### Text Chunking

```python
async def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
```
This function splits the extracted text into manageable chunks for processing.

### Vector Store Creation

```python
async def get_vector_store(text_chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
    vector_store.save_local("faiss_index")
```
This function creates a FAISS index from the text chunks for efficient similarity searches.

### Conversational AI

```python
async def get_conversational_chain():
    prompt_template = """ ... """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
```
This function sets up the question-answering chain using the Gemini model.

### User Interaction

```python
async def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    ...
```
This function processes user input, retrieves relevant documents, and generates a response.



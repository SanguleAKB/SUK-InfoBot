from flask import Flask, request, render_template, jsonify
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from typing import List
from langchain_core.documents import Document
import os

# Set Environment Variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Add your API key

# Initialize chat model
llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")

# Initialize embeddings
from langchain_google_vertexai import VertexAIEmbeddings
embeddings = VertexAIEmbeddings(model="text-embedding-004")

# Initialize vector store
vector_store = InMemoryVectorStore(embeddings)

# Load and process documents
loader = AsyncChromiumLoader([
    "https://www.unishivaji.ac.in/",
    "https://www.unishivaji.ac.in/about_suk/about-us",
    "https://www.unishivaji.ac.in/syllabusnew/",
    "https://www.unishivaji.ac.in/mscadmission/For-Student-Information",
    "https://www.unishivaji.ac.in/student/Boys-Hostel",
    "https://www.unishivaji.ac.in/girls_Hostel/",
    "https://www.unishivaji.ac.in/about_suk/About-Kolhapur",
    "https://www.unishivaji.ac.in/Academic_Programs/LIST-OF-PROGRAMS-OFFERED-ON-CAMPUS",
    "https://www.unishivaji.ac.in/about_suk/Organization-Structure#officers"
])
html = loader.load()

# Transform documents
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["body"])

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs_transformed)

# Index chunks
vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    if user_input:
        retrieved_docs = vector_store.similarity_search(user_input)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        messages = prompt.invoke({"question": user_input, "context": docs_content})
        response = llm.invoke(messages)
        answer = response.content
        return jsonify({"bot_response": answer})
    return jsonify({"bot_response": "Please provide a valid question."})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, jsonify, request
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
import pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import CTransformers

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

index_name = "medical-chatbot"

# Load the Pinecone index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Define the prompt template
PROMPT = PromptTemplate(
    template="Use the following context to answer the question: {context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# Load the LLM model
llm = CTransformers(
    model="D:/medical_chatbot/model/llama-2-7b-chat.ggmlv3.q4_0_2.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Build the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        # Log incoming request
        data = request.get_json()
        print("Received request:", data)

        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request. 'message' is required."}), 400

        user_message = data['message']
        print(f"User message: {user_message}")

        # Process message with RetrievalQA
        result = qa({"query": user_message})
        print("QA Input Query:", user_message)
        print("QA Output Result:", result)

        response = result.get("result", "No response available.")
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5004, debug=True)

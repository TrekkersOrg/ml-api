from flask import Flask, request, jsonify
import os
from langchain_community.vectorstores import Pinecone
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceBgeEmbeddings
import json
import pinecone
import argparse
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
wsgi_app = app.wsgi_app
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
LLM_MODEL = os.getenv('LLM_MODEL')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

@app.route('/riskAssessment', methods=['POST'])
def risk_assessment():
    requestBody = request.json
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    vectorstore=Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=embeddings, namespace=request.json['namespace'])
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=LLM_MODEL, temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever())
    response = qa.run(request.json['query'])
    return jsonify({'query': request.json['query'], 'response': response})

@app.route('/')
def main():
    return ""

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
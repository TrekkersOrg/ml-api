from flask import Flask, request, jsonify
from langchain_community.vectorstores import Pinecone
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceBgeEmbeddings
import json
import pinecone
import argparse
from langchain.embeddings.openai import OpenAIEmbeddings

app = Flask(__name__)

wsgi_app = app.wsgi_app

@app.route('/riskAssessment', methods=['POST'])
def risk_assessment():
    requestBody = request.json
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key='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')
    pinecone.init(api_key='3549864b-6436-4d2a-85d8-7c9216f08e0a', environment='gcp-starter')
    vectorstore=Pinecone.from_existing_index(index_name='document-index', embedding=embeddings, namespace=request.json['namespace'])
    llm = ChatOpenAI(openai_api_key='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', model_name='gpt-3.5-turbo-0125', temperature=0.0)
    conv_mem = ConversationBufferWindowMemory(memory_key='history', k=5, return_messages=True)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever())
    response = qa.run(request.json['query'])
    return jsonify({'query': request.json['query'], 'response': response})

@app.route('/')
def hello():
    """Renders a sample page."""
    return "Hello World!"

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)

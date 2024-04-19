from asyncio.windows_events import NULL
from sqlite3 import Date
from xmlrpc.client import DateTime
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
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from docx import Document as DocumentReader
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
wsgi_app = app.wsgi_app


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


def split_docs(documents,chunk_size=150,chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def read_docx(file_path):
    document = DocumentReader(file_path)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text


def create_response_model(statusCode, statusMessage, statusMessageText, elapsedTime, data=None):
    return jsonify({'statusCode': statusCode, 'statusMessage': statusMessage, 'statusMessageText': statusMessageText, 'timestamp': time.time(), 'elapsedTimeSeconds': elapsedTime, 'data': data})


@app.route('/riskAssessment', methods=['POST'])
def risk_assessment():
    start_time = time.time()
    missing_fields = [field for field in ['namespace'] if field not in request.json]
    if missing_fields:
        end_time = time.time()
        response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
        return create_response_model(200, "Success", "Risk assessment did not execute successfully.", end_time-start_time, response)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key='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')
    pinecone.init(api_key='3549864b-6436-4d2a-85d8-7c9216f08e0a', environment='gcp-starter')
    vectorstore=Pinecone.from_existing_index(index_name='document-index', embedding=embeddings, namespace=request.json['namespace'])
    llm = ChatOpenAI(openai_api_key='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', model_name='gpt-3.5-turbo-0125', temperature=0.0)
    conv_mem = ConversationBufferWindowMemory(memory_key='history', k=5, return_messages=True)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever())
    system_query = "What is the legal risk of this document?"
    system_response = qa.run(system_query)
    response = {'response': system_response}
    end_time = time.time()
    return create_response_model(200, "Success", "Risk assessment executed successfully.", end_time-start_time, response)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    start_time = time.time()
    missing_fields = [field for field in ['query', 'namespace'] if field not in request.json]
    if missing_fields:
        end_time = time.time()
        response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
        return create_response_model(200, "Success", "Chatbot did not execute successfully.", end_time-start_time, response)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key='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')
    pinecone.init(api_key='3549864b-6436-4d2a-85d8-7c9216f08e0a', environment='gcp-starter')
    vectorstore=Pinecone.from_existing_index(index_name='document-index', embedding=embeddings, namespace=request.json['namespace'])
    llm = ChatOpenAI(openai_api_key='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', model_name='gpt-3.5-turbo', temperature=0.0)
    conv_mem = ConversationBufferWindowMemory(memory_key='history', k=5, return_messages=True)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever())
    system_response = qa.run(request.json['query'])
    response = {'query': request.json['query'], 'response': system_response}
    end_time = time.time()
    return create_response_model(200, "Success", "Chatbot executed successfully.", end_time-start_time, response)

@app.route('/embedder', methods=['POST'])
def embedder():
    start_time = time.time()
    documents = []
    # Get mongo document text

    # Initialize to doc object

    # Add to documents []

    # Split

    # 


@app.route('/')
def hello():
    return "Hello World!"


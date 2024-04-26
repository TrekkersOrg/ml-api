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
import pymongo
import concurrent.futures

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

def extract_bson_text(file_name, namespace):
    try:
        client = pymongo.MongoClient("mongodb+srv://admin:Qawsaz789!@userfiles.zyeo0rx.mongodb.net/?retryWrites=true&w=majority&appName=UserFiles")
        database = client["Production"]
        collection = database[namespace]
        target_document = collection.find_one({"file_name": file_name})
        if target_document:
            return target_document.get("content")
        else:
            return False
    except Exception as e:
        return False

def ra_system_query(namespace):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key='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')
    pinecone.init(api_key='3549864b-6436-4d2a-85d8-7c9216f08e0a', environment='gcp-starter')
    vectorstore=Pinecone.from_existing_index(index_name='document-index', embedding=embeddings, namespace=namespace)
    llm = ChatOpenAI(openai_api_key='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', model_name='gpt-3.5-turbo', temperature=1.0)
    conv_mem = ConversationBufferWindowMemory(memory_key='history', k=5, return_messages=True)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever())
    operational_query = "Based on the given document text, you will assess the operational risk on a scale of 1 to 5, where 5 is the highest risk. To derive the robustness score for a document of the legal category you will judge across five general sectors: (1) Risk Identification that covers all areas of business in breadth (I.e., financial, legal, IT), along with the potential consequences and causes to potential vulnerabilities; (2) Risk assessment and Prioritization which shall include the probability of each risk occurring and the potential severity of its impact plus an outline on how to allocate resources towards mitigating the most critical risks first; (3) Risk mitigation strategies with defined clear steps that plan to reduce the likelihood or impact of each risk, an accounting for various approaches such as avoidance, reduction, transfer, or acceptance, and finally any mentions of cost for risk mitigation with the potential financial and operational impact of the risk; (4) Contingency plan consisting of alternative plans to respond to disruptions caused by identified risks along with clear assignments of roles and responsibilities for implementing the contingency plan; (5) Communication and monitoring that discusses a clear communication plan to handle relay of identified risks and mitigation plans to relevant stakeholders, including a plan to monitor the effectiveness of the risk management plan, and finally statements of processes to handle any new information, lessons learned, and changes in the business environment. Present the overall score output. Your response should range between 1-5, you can include float integers only up to the first decimal spot. Before you present your answer, double check your scores and ensure you have an accurate assessment for each sector. You must NOT present any explanation on how you found to derive this score, please only present your overall output."
    regulatory_query = "Based on the given document text, you will assess the regulatory risk on a scale of 1 to 5, where 5 is the highest risk. To derive the robustness score for a document of the legal category you will judge across four general sectors: (1) Clarity and specificity, such as use of precise language to outline rights and obligations of parties involved; (2) Comprehensiveness to anticipate future issues, such as mitigation language for potential counter statements or misinterpretations; (3) General Formalities which must include proper signatures by all authorized parties, date and place of signing, and proper formatting; and (4) Governing Law to specify the jurisdiction which will govern the agreement in case of disputes, and clauses accounting for dispute resolutions. Present the overall score output. Your response should range between 1-5, you can include float integers only up to the first decimal spot. Before you present your answer, double check your scores and ensure you have an accurate assessment for each sector. You must NOT present any explanation on how you found to derive this score, please only present your overall output."
    operational_score = None
    regulatory_score = None
    while operational_score is None or not (isinstance(operational_score, float) or (isinstance(operational_score, str) and operational_score.replace('.', '', 1).isdigit())):
        operational_score = qa.run(operational_query)
    operational_score = float(operational_score) if isinstance(operational_score, str) else operational_score
    while regulatory_score is None or not (isinstance(regulatory_score, float) or (isinstance(regulatory_score, str) and regulatory_score.replace('.', '', 1).isdigit())):
        regulatory_score = qa.run(regulatory_query)
    regulatory_score = float(regulatory_score) if isinstance(regulatory_score, str) else regulatory_score
    return {'operationalScore': operational_score, 'regulatoryScore': regulatory_score}

def ra_keywords():
    operational_score = 3
    regulatory_score = 4
    return {'operationalScore': operational_score, 'regulatoryScore': regulatory_score}

def ra_cohere():
    operational_score = 3
    regulatory_score = 1
    return {'operationalScore': operational_score, 'regulatoryScore': regulatory_score}

def ra_custom():
    operational_score = 2
    regulatory_score = 5
    return {'operationalScore': operational_score, 'regulatoryScore': regulatory_score}

def ra_scores(system_query_scores, keywords_scores, cohere_scores, custom_scores):
    operational_score_list = [system_query_scores['operationalScore'], keywords_scores['operationalScore'], cohere_scores['operationalScore'], custom_scores['operationalScore']]
    regulatory_score_list = [system_query_scores['regulatoryScore'], keywords_scores['regulatoryScore'], cohere_scores['regulatoryScore'], custom_scores['regulatoryScore']]
    operational_score = sum(operational_score_list) / len(operational_score_list)
    regulatory_score = sum(regulatory_score_list) / len(regulatory_score_list)
    return {'operationalScore': operational_score, 'regulatoryScore': regulatory_score}


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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        system_query_model = executor.submit(ra_system_query, request.json['namespace'])
        keywords_model = executor.submit(ra_keywords)
        cohere_model = executor.submit(ra_cohere)
        custom_model = executor.submit(ra_custom)
        concurrent.futures.wait([system_query_model, keywords_model, cohere_model, custom_model])
    system_query_scores = system_query_model.result()
    keywords_scores = keywords_model.result()
    cohere_scores =  cohere_model.result()
    custom_scores = custom_model.result()
    risk_assessment_scores = ra_scores(system_query_scores, keywords_scores, cohere_scores, custom_scores)
    response = {'result': risk_assessment_scores, 'system_query': system_query_scores, 'keywords': keywords_scores, 'cohere': cohere_scores, 'classification': custom_scores}
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
    missing_fields = [field for field in ['fileName', 'namespace'] if field not in request.json]
    if missing_fields:
        end_time = time.time()
        response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
        return create_response_model(200, "Success", "Embedder did not execute successfully.", end_time-start_time, response)
    documents = []
    text = extract_bson_text(request.json['fileName'], request.json['namespace'])
    if text != False:
        documents.append(Document(page_content=text))
    else:
        end_time = time.time()
        return create_response_model(200, "Success", "Embedder did not execute successfully.", end_time-start_time, text)
    chunked_documents = split_docs(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key="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")
    pinecone.init(api_key='3549864b-6436-4d2a-85d8-7c9216f08e0a', environment='gcp-starter')
    index = Pinecone.from_documents(chunked_documents, embeddings, index_name='document-index', namespace=request.json['namespace'])
    response = {'fileName': request.json['fileName'], 'namespace': request.json['namespace']}
    end_time = time.time()
    return create_response_model(200, "Success", "Embedder executed successfully.", end_time-start_time, response)


@app.route('/')
def hello():
    return "Hello World"

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)

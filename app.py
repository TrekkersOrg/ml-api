from sqlite3 import Date
from xmlrpc.client import DateTime
from flask import Flask, request, jsonify, render_template
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
from flask_cors import CORS, cross_origin
from collections import Counter
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

load_dotenv()
app = Flask(__name__)
wsgi_app = app.wsgi_app

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
LLM_MODEL = os.environ.get('LLM_MODEL')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
MONGODB_HOST = os.environ.get('MONGODB_HOST')
MONGODB_DATABASE = os.environ.get('MONGODB_DATABASE')

# Risk Assessment System Query Hyperparameters
_rasq_temperature = 1.0
_rasq_operational_query = "Based on the given document text, you will assess the operational risk on a scale of 1 to 5, where 5 is the highest risk. To derive the robustness score for a document of the legal category you will judge across five general sectors: (1) Risk Identification that covers all areas of business in breadth (I.e., financial, legal, IT), along with the potential consequences and causes to potential vulnerabilities; (2) Risk assessment and Prioritization which shall include the probability of each risk occurring and the potential severity of its impact plus an outline on how to allocate resources towards mitigating the most critical risks first; (3) Risk mitigation strategies with defined clear steps that plan to reduce the likelihood or impact of each risk, an accounting for various approaches such as avoidance, reduction, transfer, or acceptance, and finally any mentions of cost for risk mitigation with the potential financial and operational impact of the risk; (4) Contingency plan consisting of alternative plans to respond to disruptions caused by identified risks along with clear assignments of roles and responsibilities for implementing the contingency plan; (5) Communication and monitoring that discusses a clear communication plan to handle relay of identified risks and mitigation plans to relevant stakeholders, including a plan to monitor the effectiveness of the risk management plan, and finally statements of processes to handle any new information, lessons learned, and changes in the business environment. Present the overall score output. Your response should range between 1-5, you can include float integers only up to the first decimal spot. Before you present your answer, double check your scores and ensure you have an accurate assessment for each sector. You must NOT present any explanation on how you found to derive this score, please only present your overall output."
_rasq_regulatory_query = "Based on the given document text, you will assess the regulatory risk on a scale of 1 to 5, where 5 is the highest risk. To derive the robustness score for a document of the legal category you will judge across four general sectors: (1) Clarity and specificity, such as use of precise language to outline rights and obligations of parties involved; (2) Comprehensiveness to anticipate future issues, such as mitigation language for potential counter statements or misinterpretations; (3) General Formalities which must include proper signatures by all authorized parties, date and place of signing, and proper formatting; and (4) Governing Law to specify the jurisdiction which will govern the agreement in case of disputes, and clauses accounting for dispute resolutions. Present the overall score output. Your response should range between 1-5, you can include float integers only up to the first decimal spot. Before you present your answer, double check your scores and ensure you have an accurate assessment for each sector. You must NOT present any explanation on how you found to derive this score, please only present your overall output."
    
# Risk Assessment Keyword Hyperparameters
_rakw_high_regulatory_risk = ["State of California", "Secretary of State", "Articles of Organization", "Registered Agent", "Service of Process", "Statutes"]
_rakw_low_regulatory_risk = ["Limited Liability Company", "Operating Agreement", "Member(s)", "Principal Place of Business", "Registered Agent", "Formation", "Term"]
_rakw_high_operational_risk = ["Ideation", "Product Development", "Software Development", "Business Development", "Management", "Operations", "Budgeting"]
_rakw_low_operational_risk = ["Limited Liability Company", "Operating Agreement", "Member(s)", "Principal Place of Business", "Registered Agent", "Formation", "Statutes", "Term"]
_rakw_high_risk_multiplier = 2
_rakw_n_value = 45

# Chatbot Hyperparameters
_cb_temperature = 0.0
_cb_conversation_memory_template = "I have provided some documents for your reference. Additionally, I've recorded our past conversations, which are organized chronologically with the most recent one being last. You can consider these past interactions if they might be helpful for understanding the context of my question. However, the primary source of knowledge for your answer should be the documents I've provided. PAST 5 CONVERSATIONS: "

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
        client = pymongo.MongoClient(MONGODB_HOST)
        database = client[MONGODB_DATABASE]
        collection = database[namespace]
        target_document = collection.find_one({"file_name": file_name})
        if target_document:
            return target_document.get("content")
        else:
            return False
    except Exception as e:
        return False

def keyword_frequency(keyword_list, target_content):
    keywords_to_search = ' '.join(keyword_list)
    frequency = 0
    for keyword in keyword_list:
        frequency += target_content.count(keyword)
    return frequency

def custom_xgb():
    data = pd.DataFrame()
    X = data.drop(columns=['target'])
    y = data['target']

    # X_train: TF-DF of all training document terms.
    # X_test: TF-IDF of target document terms
    # y_train: Risk score of training documents.
    # y_test: Risk score of testing documents.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    xgb_classifier_model = xgb.XGBClassifier()
    xgb_classifier_model.fit(X_train, y_train)
    predictions = xgb_classifier_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return predictions


def ra_system_query(namespace):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    vectorstore=Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=embeddings, namespace=namespace)
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=LLM_MODEL, temperature=_rasq_temperature)
    conv_mem = ConversationBufferWindowMemory(memory_key='history', k=5, return_messages=True)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever())
    operational_query = _rasq_operational_query
    regulatory_query = _rasq_regulatory_query
    operational_score = None
    regulatory_score = None
    while operational_score is None or not (isinstance(operational_score, float) or (isinstance(operational_score, str) and operational_score.replace('.', '', 1).isdigit())):
        operational_score = qa.run(operational_query)
    operational_score = float(operational_score) if isinstance(operational_score, str) else operational_score
    while regulatory_score is None or not (isinstance(regulatory_score, float) or (isinstance(regulatory_score, str) and regulatory_score.replace('.', '', 1).isdigit())):
        regulatory_score = qa.run(regulatory_query)

    regulatory_score = float(regulatory_score) if isinstance(regulatory_score, str) else regulatory_score
    return {'operationalScore': operational_score, 'regulatoryScore': regulatory_score}

def ra_keywords(file_name, namespace):
    target_content = extract_bson_text(file_name, namespace)
    high_regulatory_risk_list = _rakw_high_regulatory_risk
    low_regulatory_risk_list = _rakw_low_regulatory_risk
    high_regulatory_risk_score = (keyword_frequency(high_regulatory_risk_list, target_content)) * _rakw_high_risk_multiplier
    low_regulatory_risk_score = keyword_frequency(low_regulatory_risk_list, target_content)
    regulatory_score = round((high_regulatory_risk_score + low_regulatory_risk_score) / _rakw_n_value, 1)
    high_operational_list = _rakw_high_operational_risk
    low_operational_list = _rakw_low_operational_risk
    high_operational_risk_score = (keyword_frequency(high_operational_list, target_content)) * _rakw_high_risk_multiplier
    low_operational_risk_score = keyword_frequency(low_operational_list, target_content)
    operational_score = round((high_operational_risk_score + low_operational_risk_score) / _rakw_n_value, 1)
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
    missing_fields = [field for field in ['namespace', 'file_name'] if field not in request.json]
    if missing_fields:
        end_time = time.time()
        response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
        return create_response_model(200, "Success", "Risk assessment did not execute successfully.", end_time-start_time, response)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        system_query_model = executor.submit(ra_system_query, request.json['namespace'])
        keywords_model = executor.submit(ra_keywords, request.json['file_name'], request.json['namespace'])
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
    context = request.json.get('context', None)
    template = ""
    if context is not None:
        template = _cb_conversation_memory_template
        for i, item in enumerate(context, start=1):
            query_key = f"query{i}"
            response_key = f"response{i}"
            query = item.get(query_key, "")
            response = item.get(response_key, "")
            if query:
                template += f"Query {i}: {query},\n"
            if response:
                template += f"Response {i}: {response},\n"
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    vectorstore=Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=embeddings, namespace=request.json['namespace'])
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=LLM_MODEL, temperature=_cb_temperature)
    conv_mem = ConversationBufferWindowMemory(memory_key='history', k=5, return_messages=True)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever())
    if template == "":
        query = request.json['query']
    else:
        query = template + "USER QUERY: " + request.json['query']
    system_response = qa.run(query)
    response = {'query': query, 'response': system_response}
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
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = Pinecone.from_documents(chunked_documents, embeddings, index_name=PINECONE_INDEX, namespace=request.json['namespace'])
    response = {'fileName': request.json['fileName'], 'namespace': request.json['namespace']}
    end_time = time.time()
    return create_response_model(200, "Success", "Embedder executed successfully.", end_time-start_time, response)


@app.route('/')
def main_page():
    return render_template('main_page.html')

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)

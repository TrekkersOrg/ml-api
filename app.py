import sys
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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import fitz
from azure.storage.fileshare import ShareServiceClient, ShareFileClient
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler

# Download dictionaries from NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Load environment
load_dotenv()
app = Flask(__name__)
wsgi_app = app.wsgi_app

# Initialize environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
LLM_MODEL = os.environ.get('LLM_MODEL')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
MONGODB_HOST = os.environ.get('MONGODB_HOST')
MONGODB_DATABASE = os.environ.get('MONGODB_DATABASE')
TRAINING_DOCUMENTS = os.environ.get('TRAINING_COLLECTION')
AZURE_FILES_CONN_STRING = os.environ.get('AZURE_FILES_CONN_STRING')
AZURE_FILES_SHARE_NAME = os.environ.get('AZURE_FILES_SHARE_NAME')
AZURE_FILES_CUSTOM_TRAINING_DIRECTORY = os.environ.get('AZURE_FILES_CUSTOM_TRAINING_DIRECTORY')
AZURE_FILES_KEYWORD_TRAINING_DIRECTORY = os.environ.get('AZURE_FILES_KEYWORD_TRAINING_DIRECTORY')
ENVIRONMENT = os.environ.get('ENVIRONMENT')

for key, value in os.environ.items():
    print(f"{key}: {value}")


# Risk Assessment System Query Hyperparameters
_rasq_temperature = 1.0
_rasq_operational_query = "Based on the given document text, you will assess the operational risk on a scale of 1 to 5, where 5 is the highest risk. To derive the robustness score for a document of the legal category you will judge across five general sectors: (1) Risk Identification that covers all areas of business in breadth (I.e., financial, legal, IT), along with the potential consequences and causes to potential vulnerabilities; (2) Risk assessment and Prioritization which shall include the probability of each risk occurring and the potential severity of its impact plus an outline on how to allocate resources towards mitigating the most critical risks first; (3) Risk mitigation strategies with defined clear steps that plan to reduce the likelihood or impact of each risk, an accounting for various approaches such as avoidance, reduction, transfer, or acceptance, and finally any mentions of cost for risk mitigation with the potential financial and operational impact of the risk; (4) Contingency plan consisting of alternative plans to respond to disruptions caused by identified risks along with clear assignments of roles and responsibilities for implementing the contingency plan; (5) Communication and monitoring that discusses a clear communication plan to handle relay of identified risks and mitigation plans to relevant stakeholders, including a plan to monitor the effectiveness of the risk management plan, and finally statements of processes to handle any new information, lessons learned, and changes in the business environment. Present the overall score output. Your response should range between 1-5, you can include float integers only up to the first decimal spot. Before you present your answer, double check your scores and ensure you have an accurate assessment for each sector. You must NOT present any explanation on how you found to derive this score, please only present your overall output."
_rasq_regulatory_query = "Based on the given document text, you will assess the regulatory risk on a scale of 1 to 5, where 5 is the highest risk. To derive the robustness score for a document of the legal category you will judge across four general sectors: (1) Clarity and specificity, such as use of precise language to outline rights and obligations of parties involved; (2) Comprehensiveness to anticipate future issues, such as mitigation language for potential counter statements or misinterpretations; (3) General Formalities which must include proper signatures by all authorized parties, date and place of signing, and proper formatting; and (4) Governing Law to specify the jurisdiction which will govern the agreement in case of disputes, and clauses accounting for dispute resolutions. Present the overall score output. Your response should range between 1-5, you can include float integers only up to the first decimal spot. Before you present your answer, double check your scores and ensure you have an accurate assessment for each sector. You must NOT present any explanation on how you found to derive this score, please only present your overall output."
    
# Risk Assessment Keyword Hyperparameters
KEYWORD_REWARD = 1
KEYWORD_PENALTY = 0.1

# Chatbot Hyperparameters
_cb_conversation_memory_template = "I have provided some documents for your reference. Additionally, I've recorded our past conversations, which are organized chronologically with the most recent one being last. You can consider these past interactions if they might be helpful for understanding the context of my question. However, the primary source of knowledge for your answer should be the documents I've provided. PAST 5 CONVERSATIONS: "

########## OBJECTS ##########
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

########## AZURE HELPER FUNCTIONS ##########
def upload_file_to_azure_fileshare(file_name, directory):
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    share_client = service_client.get_share_client(AZURE_FILES_SHARE_NAME)
    directory_client = share_client.get_directory_client(directory)
    file_client = directory_client.get_file_client(os.path.basename(file_name))

    print(str(share_client.url))
    print(str(file_client.url))
    print(file_name)
    with open(file_name, "rb") as source_file:
        file_client.upload_file(source_file)
    
def get_df_from_azure_fileshare(filename, directory):
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    print(str(service_client))
    file_client = service_client.get_share_client(AZURE_FILES_SHARE_NAME).get_file_client(directory + "/" + filename)
    download_stream = file_client.download_file()
    file_content = download_stream.readall()
    df = pd.read_csv(BytesIO(file_content))
    print(df)
    return df

def get_list_from_azure_fileshare(filename, directory):
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    print(str(service_client))
    file_client = service_client.get_share_client(AZURE_FILES_SHARE_NAME).get_file_client(directory + "/" + filename)
    download_stream = file_client.download_file()
    file_content = download_stream.readall()
    list_content = eval(file_content.decode('utf-8'))
    print(str(list_content))
    return list_content

########## MONGODB HELPER FUNCTIONS ##########
def insert_document(document, namespace):
    client = pymongo.MongoClient(MONGODB_HOST)
    database = client[MONGODB_DATABASE]
    collection = database[namespace]
    collection.insert_one(document)

def get_all_documents(namespace):
    try:
        client = pymongo.MongoClient(MONGODB_HOST)
        database = client[MONGODB_DATABASE]
        collection = database[namespace]
        return list(collection.find())
    except Exception as e:
        return False

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

########## EMBEDDER HELPER FUNCTIONS ##########
def split_docs(documents,chunk_size=150,chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

########## RISK ASSESSMENT MODELS ##########
##### SYSTEM QUERY MODEL#####
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
    reputational_score = 0
    financial_score = 0
    regulatory_score = float(regulatory_score) if isinstance(regulatory_score, str) else regulatory_score
    return {'operationalScore': int(operational_score), 'regulatoryScore': int(regulatory_score), 'financialScore': int(financial_score), 'reputationalScore': int(reputational_score)}

##### KEYWORD MODEL #####
def keyword_frequency(keyword_list, target_content):
    keywords_to_search = ' '.join(keyword_list)
    frequency = 0
    for keyword in keyword_list:
        frequency += target_content.count(keyword)
    return frequency

def score_scaler(score_unscaled, target_keywords_length):
    min_score_unscaled = -1 * target_keywords_length
    max_score_unscaled = target_keywords_length
    score_scaled = MinMaxScaler(feature_range=(0, 5)).fit_transform(np.array([[min_score_unscaled], [max_score_unscaled], [score_unscaled]]))[-1, 0]
    score_scaled = int(round(score_scaled))
    return score_scaled

def ra_keywords(file_name, namespace):
    target_content = extract_bson_text(file_name, namespace)
    target_keywords = custom_preprocessing(target_content).split()
    target_keywords_length = len(target_keywords)
    operational_keywords = get_list_from_azure_fileshare('operational_keywords.txt', AZURE_FILES_KEYWORD_TRAINING_DIRECTORY)
    regulatory_keywords = get_list_from_azure_fileshare('regulatory_keywords.txt', AZURE_FILES_KEYWORD_TRAINING_DIRECTORY)
    operational_score = 0
    regulatory_score = 0
    reputational_score = 0
    financial_score = 0
    for target in target_keywords:
        if target in operational_keywords:
            operational_score -= KEYWORD_REWARD
        else:
            operational_score += KEYWORD_PENALTY
        if target in regulatory_keywords:
            regulatory_score -= KEYWORD_REWARD
        else:
            regulatory_score += KEYWORD_PENALTY
    operational_score = score_scaler(operational_score, target_keywords_length)
    regulatory_score = score_scaler(regulatory_score, target_keywords_length)
    return {'operationalScore': int(operational_score), 'regulatoryScore': int(regulatory_score), 'financialScore': int(financial_score), 'reputationalScore': int(reputational_score)}

##### COHERE MODEL #####
def ra_cohere():
    operational_score = 3
    regulatory_score = 1
    reputational_score = 3
    financial_score = 5
    return {'operationalScore': int(operational_score), 'regulatoryScore': int(regulatory_score), 'financialScore': int(financial_score), 'reputationalScore': int(reputational_score)}

##### CUSTOM MODEL #####
def custom_training_dataset():
    training_documents = get_all_documents(TRAINING_DOCUMENTS)
    training_document_list = []
    operational_score_list = []
    regulatory_score_list = []
    data = pd.DataFrame()
    for document in training_documents:
        training_document_list.append(custom_preprocessing(document["content"]))
        operational_score_list.append(document["operational_score"])
        regulatory_score_list.append(document["regulatory_score"])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_training_matrix = tfidf_vectorizer.fit_transform(training_document_list)
    data = pd.DataFrame(tfidf_training_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    data['operational_score'] = operational_score_list
    data['regulatory_score'] = regulatory_score_list
    data['operational_score'] = pd.Categorical(data['operational_score'])
    data['operational_score'] = data['operational_score'].cat.codes
    data['regulatory_score'] = pd.Categorical(data['regulatory_score'])
    data['regulatory_score'] = data['regulatory_score'].cat.codes
    return data

def custom_preprocessing(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def custom_xgb(training_data, target_document, risk_category):
    target_column_list = ['operational_score', 'regulatory_score']
    target_column = ''
    if 'operation' in risk_category:
        target_column = 'operational_score'
    elif 'regulatory' in risk_category:
        target_column = 'regulatory_score'
    data = training_data
    X = data.drop(columns=target_column_list)
    y = data[target_column]
    xgb_classifier_model = xgb.XGBClassifier()
    xgb_classifier_model.fit(X, y)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(training_data.columns.drop(target_column_list))
    tfidf_target_matrix = tfidf_vectorizer.transform([target_document])
    target_data = pd.DataFrame(tfidf_target_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    target_data_aligned = target_data.reindex(columns=X.columns, fill_value=0)
    predictions = xgb_classifier_model.predict(target_data_aligned)
    return predictions[0]

def ra_custom(target_text):
    training_data = get_df_from_azure_fileshare('training_data.csv', AZURE_FILES_CUSTOM_TRAINING_DIRECTORY)
    operational_score = custom_xgb(training_data, target_text, 'operational')
    regulatory_score = custom_xgb(training_data, target_text, 'regulatory')
    reputational_score = 0
    financial_score = 0
    return {'operationalScore': int(operational_score), 'regulatoryScore': int(regulatory_score), 'financialScore': int(financial_score), 'reputationalScore': int(reputational_score)}

##### RISK ASSESSMENT SCORE COMPILATION #####
def calculate_final_score(system_query_score, keyword_score, cohere_score, custom_score):
    system_query_weight = 20
    keyword_weight = 10
    cohere_weight = 0
    custom_weight = 70
    final_score = round((system_query_score * system_query_weight + keyword_score * keyword_weight + cohere_score * cohere_weight + custom_score * custom_weight) / 100)
    return final_score

def ra_scores(system_query_scores, keywords_scores, cohere_scores, custom_scores):
    operational_score = calculate_final_score(system_query_scores['operationalScore'], keywords_scores['operationalScore'], cohere_scores['operationalScore'], custom_scores['operationalScore'])
    regulatory_score = calculate_final_score(system_query_scores['regulatoryScore'], keywords_scores['regulatoryScore'], cohere_scores['regulatoryScore'], custom_scores['regulatoryScore'])
    reputational_score = calculate_final_score(system_query_scores['reputationalScore'], keywords_scores['reputationalScore'], cohere_scores['reputationalScore'], custom_scores['reputationalScore'])
    financial_score = calculate_final_score(system_query_scores['financialScore'], keywords_scores['financialScore'], cohere_scores['financialScore'], custom_scores['financialScore'])
    final_score = round((operational_score + regulatory_score + financial_score + reputational_score) / 4)
    return {'operationalScore': int(operational_score), 'regulatoryScore': int(regulatory_score), 'financialScore': int(financial_score), 'reputationalScore': int(reputational_score), 'finalScore': int(final_score)}

########## API HELPER FUNCTIONS ##########
def create_response_model(statusCode, statusMessage, statusMessageText, elapsedTime, data=None):
    return jsonify({'statusCode': int(statusCode), 'statusMessage': statusMessage, 'statusMessageText': statusMessageText, 'timestamp': time.time(), 'elapsedTimeSeconds': float(elapsedTime), 'data': data})

########## API ENDPOINTS ##########
@app.route('/riskAssessment', methods=['POST'])
def risk_assessment():
    start_time = time.time()
    missing_fields = [field for field in ['namespace', 'file_name'] if field not in request.json]
    if missing_fields:
        end_time = time.time()
        response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
        return create_response_model(200, "Success", "Risk assessment did not execute successfully.", end_time-start_time, response)
    target_document = extract_bson_text(request.json['file_name'], request.json['namespace'])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        system_query_model = executor.submit(ra_system_query, request.json['namespace'])
        keywords_model = executor.submit(ra_keywords, request.json['file_name'], request.json['namespace'])
        cohere_model = executor.submit(ra_cohere)
        custom_model = executor.submit(ra_custom, target_document)
        concurrent.futures.wait([system_query_model, keywords_model, cohere_model, custom_model])
    system_query_scores = system_query_model.result()
    keywords_scores = keywords_model.result()
    cohere_scores =  cohere_model.result()
    custom_scores = custom_model.result()
    risk_assessment_scores = ra_scores(system_query_scores, keywords_scores, cohere_scores, custom_scores)
    response = {'result': risk_assessment_scores, 'system_query': system_query_scores, 'keywords': keywords_scores, 'cohere': cohere_scores, 'custom': custom_scores}
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
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=LLM_MODEL, temperature=0.0)
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

@app.route('/updateTrainingData', methods=['POST'])
def update_training_data():
    start_time = time.time()
    if 'file' not in request.files:
        end_time = time.time()
        response = {'error': 'No file part'}
        return create_response_model(200, "Error", "No file part in the request.", end_time - start_time, response)
    missing_fields = [field for field in ['operational_score', 'regulatory_score'] if field not in request.form]
    if missing_fields:
        end_time = time.time()
        response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
        return create_response_model(200, "Success", "Did not execute successfully.", end_time-start_time, response)
    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        document_text = ''
        pdf_document = fitz.open("pdf", file.read())
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            document_text += page.get_text()
        missing_fields = [field for field in ['operational_score', 'regulatory_score'] if field not in request.form]
        if missing_fields:
            end_time = time.time()
            response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
            return create_response_model(400, "Error", f'Missing fields: {", ".join(missing_fields)}', end_time - start_time, response)
        document = {
            'content': document_text,
            'operational_score': int(request.form['operational_score']),
            'regulatory_score': int(request.form['regulatory_score'])
        }
    insert_document(document, TRAINING_DOCUMENTS)
    if (ENVIRONMENT == 'development'):
        training_data = custom_training_dataset()
        training_data_file_name = "training_data.csv"
        training_data.to_csv(training_data_file_name, index=False)
        upload_file_to_azure_fileshare(training_data_file_name, AZURE_FILES_CUSTOM_TRAINING_DIRECTORY)
        end_time = time.time()
        return create_response_model(200, "Success", "Updated training data successfully.", end_time-start_time)
    else:
        training_data = custom_training_dataset()
        os.makedirs('/home/site/wwwroot/data', exist_ok=True)
        training_data_file_name = "training_data.csv"
        training_data_file_path = os.path.join('/home/site/wwwroot/data', training_data_file_name)
        training_data.to_csv(training_data_file_path, index=False)
        print(upload_file_to_azure_fileshare(training_data_file_path, AZURE_FILES_CUSTOM_TRAINING_DIRECTORY))
        end_time = time.time()
        return create_response_model(200, "Success", "Updated training data successfully.", end_time-start_time)

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
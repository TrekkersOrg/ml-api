# TODO: Create calculate final risk score endpoint
# TODO: Deprecate risk assessment API
# TODO: Change regulatory to compliance

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
import openai
import ast

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
AZURE_FILES_CONVERSATION_HISTORY_SHARE_NAME = os.environ.get('AZURE_FILES_CONVERSATION_HISTORY_SHARE_NAME')

# Risk Assessment Code Analysis Hyperparameters
_code_analysis_query = "Use the code as context. Using the following rubric, determine a decimal risk score." + """ 
    This rubric is rated on a scale from 0 (low risk) to 5 (high risk). Based on the criteria below, the score can be any decimal between 0-5.
    Score of 0: Code Vulnerabilities: No known vulnerabilities; code has been thoroughly tested and audited against OWASP Top 10 risks. Regular static and dynamic code analysis performed. Data Protection and Privacy: Full compliance with GDPR and ISO 27001 standards. Data is encrypted in transit and at rest, with strong anonymization techniques. Authentication and Access Control: Implement multi-factor authentication (MFA) and role-based access control (RBAC). Regular audits of access logs. Network Security: Strong firewall rules, intrusion detection systems (IDS), and encryption protocols. Regular penetration testing and security reviews. Third-Party Dependencies: No high-risk dependencies; all are actively maintained and regularly audited. Dependency-checking tools are used to monitor vulnerabilities. Incident Response and Recovery: Comprehensive incident response plan tested regularly; quick recovery assured. Business continuity and disaster recovery plans are in place.
 
    Score of 1: Code Vulnerabilities: Minor vulnerabilities that are unlikely to be exploited; patches are available and scheduled. Regular code reviews and adherence to secure coding practices. Data Protection and Privacy: Good data protection measures; mostly compliant with privacy laws. Minor improvements needed in encryption or data minimization. Authentication and Access Control: Strong mechanisms with minor potential weaknesses; regular password policy updates. Network Security: Good measures with minor potential vulnerabilities; regular updates and monitoring. Third-Party Dependencies: Few dependencies with minor vulnerabilities; updates are available and planned. Incident Response and Recovery: Good incident response plan with minor improvements needed. Regular drills and updates to the response plan.
 
    Score of 2: Code Vulnerabilities: Some vulnerabilities with limited potential impact; fixes are planned and prioritized. Basic implementation of secure coding standards. Data Protection and Privacy: Basic data protection; some non-compliance with GDPR. Lack of data breach notification procedures. Authentication and Access Control: Adequate mechanisms with some weaknesses; need regular reviews. Network Security: Adequate measures with some vulnerabilities; firewalls and IDS need strengthening. Third-Party Dependencies: Some dependencies with known vulnerabilities; updates planned but not immediate. Incident Response and Recovery: Basic plan in place; moderate recovery time. Incident response roles are defined but not regularly trained.
 
    Score of 3: Code Vulnerabilities: Vulnerabilities that could be exploited with moderate impact; partial fixes exist. Inconsistent application of security patches. Data Protection and Privacy: Insufficient data protection; potential non-compliance with privacy laws. Inadequate data retention and deletion policies. Authentication and Access Control: Weak mechanisms; susceptible to unauthorized access. No MFA or RBAC. Network Security: Weak measures; multiple vulnerabilities present. Lack of regular network security assessments. Third-Party Dependencies: Dependencies with significant vulnerabilities; updates delayed. Poor monitoring of third-party risks. Incident Response and Recovery: Limited plan; slow recovery expected. Incident response procedures not tested.
 
    Score of 4: Code Vulnerabilities: Serious vulnerabilities with high potential impact; exploitation is plausible. Little to no security testing or audits. Data Protection and Privacy: Poor data protection; known violations of GDPR. Data subjects' rights not respected or facilitated. Authentication and Access Control: Very weak mechanisms; unauthorized access is likely. No regular reviews or updates to access controls. Network Security: Very weak measures; exploitation of vulnerabilities is likely. No encryption of sensitive data. Third-Party Dependencies: High-risk dependencies with known vulnerabilities; no updates available. No vetting process for new dependencies. Incident Response and Recovery: Minimal plan; untested recovery procedures. No regular updates or improvements to the plan.
 
    Score of 5: Code Vulnerabilities: Critical vulnerabilities that are easily exploitable with severe consequences. No security practices in place. Data Protection and Privacy: No data protection measures; significant privacy law violations. Data breaches are common and unreported. Authentication and Access Control: No authentication or access control mechanisms in place. Unrestricted access to sensitive information. Network Security: No network security measures; extremely vulnerable to attacks. Complete lack of monitoring and response capabilities. Third-Party Dependencies: Critical dependencies with severe vulnerabilities; no plans for updates. No awareness of dependency risks. Incident Response and Recovery: No incident response plan; recovery unlikely. Organization is unprepared for data breaches or attacks.
    
    """ + "Structure your response in a JSON format with the key 'score' being the decimal risk score, the key 'reasoning' that contains the reasoning you had for giving such a score based on the rubric, and the key 'suggestions' that offer suggestions to make the code better and give a lower risk score. For the 'suggestions', please ensure you give suggestions specific to the code as well, such as syntax changes from encoding, hiding API keys, etc. Do not make general suggestions like regular reviews; give suggestions that are specific code changes."


# Risk Assessment System Query Hyperparameters
_rasq_temperature = 1.0
_rasq_operational_query = "Based on the given document text, you will assess the operational risk on a scale of 1 to 5, where 5 is the highest risk. To derive the robustness score for a document of the legal category you will judge across five general sectors: (1) Risk Identification that covers all areas of business in breadth (I.e., financial, legal, IT), along with the potential consequences and causes to potential vulnerabilities; (2) Risk assessment and Prioritization which shall include the probability of each risk occurring and the potential severity of its impact plus an outline on how to allocate resources towards mitigating the most critical risks first; (3) Risk mitigation strategies with defined clear steps that plan to reduce the likelihood or impact of each risk, an accounting for various approaches such as avoidance, reduction, transfer, or acceptance, and finally any mentions of cost for risk mitigation with the potential financial and operational impact of the risk; (4) Contingency plan consisting of alternative plans to respond to disruptions caused by identified risks along with clear assignments of roles and responsibilities for implementing the contingency plan; (5) Communication and monitoring that discusses a clear communication plan to handle relay of identified risks and mitigation plans to relevant stakeholders, including a plan to monitor the effectiveness of the risk management plan, and finally statements of processes to handle any new information, lessons learned, and changes in the business environment. Present the overall score output. Your response should range between 1-5, you can include float integers only up to the first decimal spot. Before you present your answer, double check your scores and ensure you have an accurate assessment for each sector. You must NOT present any explanation on how you found to derive this score, please only present your overall output."
_rasq_regulatory_query = "Based on the given document text, you will assess the regulatory risk on a scale of 1 to 5, where 5 is the highest risk. To derive the robustness score for a document of the legal category you will judge across four general sectors: (1) Clarity and specificity, such as use of precise language to outline rights and obligations of parties involved; (2) Comprehensiveness to anticipate future issues, such as mitigation language for potential counter statements or misinterpretations; (3) General Formalities which must include proper signatures by all authorized parties, date and place of signing, and proper formatting; and (4) Governing Law to specify the jurisdiction which will govern the agreement in case of disputes, and clauses accounting for dispute resolutions. Present the overall score output. Your response should range between 1-5, you can include float integers only up to the first decimal spot. Before you present your answer, double check your scores and ensure you have an accurate assessment for each sector. You must NOT present any explanation on how you found to derive this score, please only present your overall output."
    
# Risk Assessment Keyword Hyperparameters
KEYWORD_REWARD = 1
KEYWORD_PENALTY = 0.1

# Chatbot Hyperparameters
_cb_conversation_memory_template = "I have provided some documents for your reference. Additionally, I've recorded our past conversations, which are organized chronologically with the most recent one being last. You can consider these past interactions if they might be helpful for understanding the context of my question. However, the primary source of knowledge for your answer should be the documents I've provided. PAST 5 CONVERSATIONS: "

########## CLASSES ##########
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}
        
class Chatbot:
    def __init__(self, context):
        self.conversation_history = []
        self.context = context
        
    def chat(self, user_input):
        context_message = {"role": "system", "content": self.context}
        messages = [context_message] + self.conversation_history + [{"role": "user", "content": user_input}]
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=messages
        )
        bot_response = response.choices[0].message.content
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": bot_response})
        return bot_response
    
    def clearMemory(self):
        self.conversation_history = []
        return
    
########## AZURE HELPER FUNCTIONS ##########
def upload_file_to_azure_fileshare(filename: str, directory: str):
    """
        Uploads a file to the Azure file share.

        :param filename: The name of the file.
        :param directory: The name of the Azure file share directory.
    """
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    share_client = service_client.get_share_client(AZURE_FILES_SHARE_NAME)
    directory_client = share_client.get_directory_client(directory)
    file_client = directory_client.get_file_client(os.path.basename(filename))
    with open(filename, "rb") as source_file:
        file_client.upload_file(source_file)
    
def get_df_from_azure_fileshare(filename: str, directory: str) -> pd.DataFrame:
    """
        Returns a dataframe of a CSV in the Azure file share.

        :param file_name: The name of the file.
        :param directory: The name of the Azure file share directory.
        :return: A Pandas Dataframe containing the contents of the CSV.
    """
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    file_client = service_client.get_share_client(AZURE_FILES_SHARE_NAME).get_file_client(directory + "/" + filename)
    download_stream = file_client.download_file()
    file_content = download_stream.readall()
    df = pd.read_csv(BytesIO(file_content))
    return df

def get_list_from_azure_fileshare(filename: str, directory: str) -> list[str]:
    """
        Returns a list of a list in the Azure file share.

        :param file_name: The name of the file.
        :param directory: The name of the Azure file share directory.
        :return: A list containing the contents of the list.
    """
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    file_client = service_client.get_share_client(AZURE_FILES_SHARE_NAME).get_file_client(directory + "/" + filename)
    download_stream = file_client.download_file()
    file_content = download_stream.readall()
    list_content = eval(file_content.decode('utf-8'))
    return list_content

def get_conversation_memory(namespace: str) -> json:
    """
        Returns a JSON conversation history of a user.

        :param namespace: The namespace of the user.
        :return: A JSON containing the contents of the conversation history.
    """
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    file_client = service_client.get_share_client(AZURE_FILES_CONVERSATION_HISTORY_SHARE_NAME).get_file_client(namespace + ".txt")
    download_stream = file_client.download_file()
    file_raw_content = download_stream.readall()
    file_content = file_raw_content.decode('utf-8')
    conversation_history = json.loads(file_content)
    return conversation_history

def upload_conversation_memory(namespace: str, conversation_history) -> json:
    """
        Uploads a user conversation memory.

        :param namespace: The namespace of the user.
        :param conversation_history: The conversation history of the user.
    """
    filename = namespace + ".txt"
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    share_client = service_client.get_share_client(AZURE_FILES_CONVERSATION_HISTORY_SHARE_NAME)
    file_client = share_client.get_file_client(os.path.basename(filename))
    file_content = json.dumps(conversation_history)
    file_stream = BytesIO(file_content.encode('utf-8'))
    file_client.upload_file(file_stream)
    
def delete_conversation_memory(namespace: str):
    """
        Deletes a user conversation memory.

        :param namespace: The namespace of the user.
    """
    filename = namespace + ".txt"
    service_client = ShareServiceClient.from_connection_string(AZURE_FILES_CONN_STRING)
    share_client = service_client.get_share_client(AZURE_FILES_CONVERSATION_HISTORY_SHARE_NAME)
    file_client = share_client.get_file_client(os.path.basename(filename))
    file_client.delete_file()

########## MONGODB HELPER FUNCTIONS ##########
def insert_document(document: str, namespace: str):
    """
        Inserts a document into a MongoDB collection in the production database.

        :param document: The document text.
        :param namespace: The name of the MongoDB destination collection.
    """
    client = pymongo.MongoClient(MONGODB_HOST)
    database = client[MONGODB_DATABASE]
    collection = database[namespace]
    collection.insert_one(document)

def get_all_documents(namespace: str):
    """
        Retrieves all documents from a MongoDB collection in the production database.

        :param namespace: The name of the MongoDB collection.
        :return: List of documents in a namespace. Returns false if unsuccessful.
    """
    try:
        client = pymongo.MongoClient(MONGODB_HOST)
        database = client[MONGODB_DATABASE]
        collection = database[namespace]
        return list(collection.find())
    except Exception as e:
        return False

def extract_bson_text(filename: str, namespace: str):
    """
        Extracts text from a MongoDB document.

        :param filename: The name of the file.
        :param namespace: The name of the MongoDB collection.
        :return: The extracted text. Returns false if unsuccessful.
    """
    try:
        client = pymongo.MongoClient(MONGODB_HOST)
        database = client[MONGODB_DATABASE]
        collection = database[namespace]
        target_document = collection.find_one({"file_name": filename})
        if target_document:
            return target_document.get("content")
        else:
            return False
    except Exception as e:
        return False

########## EMBEDDER HELPER FUNCTIONS ##########
def split_docs(documents: list[Document], chunk_size: int = 150, chunk_overlap: int = 10) -> list[Document]:
    """
        Splits a document into chunks.

        :param documents: List of size 1 containing the target document.
        :param chunk_size: The size of each chunk.
        :param chunk_overlap: The intersection size of each chunk.
        :return: A list of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

########## RISK ASSESSMENT MODELS ##########
##### KEYWORD MODEL #####
def keyword_frequency(keyword_list: list[str], target_content: str) -> int:
    """
        Determines how many times the provided keywords appear in the target text.

        :param keyword_list: The list of keywords.
        :param target_content: The target text.
        :return: The total number of times the keywords appeared in the target document.
    """
    keywords_to_search = ' '.join(keyword_list)
    frequency = 0
    for keyword in keyword_list:
        frequency += target_content.count(keyword)
    return frequency

def score_scaler(score_unscaled: int, target_keywords_length: int) -> int:
    """
        Determines the keyword score depending on the length of the keywords.

        :param keyword_list: The list of keywords.
        :param target_content: The target text.
        :return: The total number of times the keywords appeared in the target document.
    """
    min_score_unscaled = -1 * target_keywords_length
    max_score_unscaled = target_keywords_length
    score_scaled = MinMaxScaler(feature_range=(0, 5)).fit_transform(np.array([[min_score_unscaled], [max_score_unscaled], [score_unscaled]]))[-1, 0]
    score_scaled = int(round(score_scaled))
    return score_scaled

##### CUSTOM MODEL #####
def custom_training_dataset() -> pd.DataFrame:
    """
        Used to create and preprocess training dataset for the custom XGBoost model.

        :return: The Pandas Dataframe of the TF-IDF scores.
    """
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

def custom_preprocessing(text) -> str:
    """
        Preprocessing functionality for the training dataset.

        :return: The Pandas Dataframe of the TF-IDF scores.
    """
    if text == False:
        return False
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def custom_xgb(training_data: pd.DataFrame, target_document: str, risk_category: str) -> int:
    """
        The custom XGBoost model for risk assessment. Returns the risk score depending on the provided category.

        :param training_data: Pandas dataframe of TF-IDF scores of training documents.
        :param target_document: The target text.
        :param risk_category: The risk category, either 'operational' or 'regulatory'.
        :return: The calculated risk score.
    """
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
    if target_document == False:
        return False
    tfidf_target_matrix = tfidf_vectorizer.transform([target_document])
    target_data = pd.DataFrame(tfidf_target_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    target_data_aligned = target_data.reindex(columns=X.columns, fill_value=0)
    predictions = xgb_classifier_model.predict(target_data_aligned)
    return predictions[0]

##### RISK ASSESSMENT SCORE COMPILATION #####
def calculate_final_score(system_query_score: int, keyword_score: int, custom_score: int) -> int:
    """
        Calculates the total risk score depending on the weights of each child model.

        :param system_query_score: The score of the system query model.
        :param keyword_score: The score of the keyword model.
        :param custom_score: The score of the custom XGBoost mdoel.
    """
    system_query_weight = 20
    keyword_weight = 10
    custom_weight = 70
    final_score = round((system_query_score * system_query_weight + keyword_score * keyword_weight + custom_score * custom_weight) / 100)
    return final_score

########## CODE ANALYSIS HELPER ##########
def parse_code_to_ast(code_file):
    with open(code_file, 'r') as f:
        code = f.read()
    return ast.parse(code)

# Runs all checks for a single file
def run_policy_check(filestream):
    # code_raw = filestream.read().decode('utf-8')
    #code_tree = ast.parse(code_raw)
    failed_issue_list = []
    passed_issue_list = []
    # Run all checks here: for node in knowledge graph
    # If failed
    failed_line_number = 10
    failed_policy = 'SQL Injection'
    failed_policy_description = 'SQL injection lets attackers manipulate a database by injecting malicious SQL code.'
    suggested_fix = 'Review SQL queries for proper parameterization.'
    failed_sample = { "line_number": failed_line_number, "policy": failed_policy, "description": failed_policy_description, "fix": suggested_fix }
    failed_issue_list.append(failed_sample)

    # Else
    passed_policy = 'Cross-Site Scripting (XSS)'
    passed_policy_description = 'Allows attackers to inject malicious scripts into web pages viewed by other users.'
    passed_sample = { "policy": passed_policy, "description": passed_policy_description }
    passed_issue_list.append(passed_sample)
    return { 'passed': passed_issue_list, 'failed': failed_issue_list }

def code_analysis(files):
    failed_issue_list = []
    passed_issue_list = []
    for file in files:
        file_result = run_policy_check(file.stream)
        if file_result['failed']:
            failed_entry = { "file_name": file.filename, "issues": file_result['failed']}
            failed_issue_list.append(failed_entry)
        if file_result['passed']:
            passed_issue_list.append(file_result['passed'])
    return { "failed": failed_issue_list, "passed": passed_issue_list}

########## API HELPER FUNCTIONS ##########
def create_response_model(statusCode: int, statusMessage: str, statusMessageText: str, elapsedTime: float, data: object = None) -> json:
    """
        The API wrapper for all API response data.

        :param statusCode: The response status code.
        :param statusMessage: Short-form message.
        :param StatusMessageText: Detailed message.
        :param elapsedTime: The number of seconds it took the API response to reach the client.
    """
    return jsonify({'statusCode': int(statusCode), 'statusMessage': statusMessage, 'statusMessageText': statusMessageText, 'timestamp': time.time(), 'elapsedTimeSeconds': float(elapsedTime), 'data': data})

########## API ENDPOINTS ##########

@app.route('/codeAnalysis', methods=['POST'])
def code_analysis_endpoint():
    start_time = time.time()
    files = []
    for file_name, file in request.files.items():
        files.append(file)
    result = code_analysis(files)
    end_time = time.time()
    return create_response_model(200, "Success", "Code analysis executed successfully.", end_time-start_time, result)


@app.route('/systemQueryModel', methods=['POST'])
def system_query_endpoint():
    """
        Calculates risk score using a system query.

        Request body: 
        {
            "namespace": string,
            "file_name": string
        }

        Response data: 
        {
            "financialScore": int,
            "operationalScore": int,
            "regulatoryScore": int,
            "reputationalScore": int
        }
    """
    start_time = time.time()
    file_name = request.json.get('file_name')
    namespace = request.json.get('namespace')
    file_content = extract_bson_text(file_name, namespace)
    chatbot = Chatbot(file_content)
    operational_query = _rasq_operational_query
    regulatory_query = _rasq_regulatory_query
    operational_score = None
    regulatory_score = None
    while operational_score is None or not (isinstance(operational_score, float) or (isinstance(operational_score, str) and operational_score.replace('.', '', 1).isdigit())):
        operational_score = chatbot.chat(operational_query)
        chatbot.clearMemory()
    operational_score = float(operational_score) if isinstance(operational_score, str) else operational_score
    while regulatory_score is None or not (isinstance(regulatory_score, float) or (isinstance(regulatory_score, str) and regulatory_score.replace('.', '', 1).isdigit())):
        regulatory_score = chatbot.chat(regulatory_query)
        chatbot.clearMemory()
    reputational_score = 0
    financial_score = 0
    regulatory_score = float(regulatory_score) if isinstance(regulatory_score, str) else regulatory_score
    response_data = {'operationalScore': int(operational_score), 'regulatoryScore': int(regulatory_score), 'financialScore': int(financial_score), 'reputationalScore': int(reputational_score)}
    end_time = time.time()
    return create_response_model(200, "Success", "System query model executed successfully.", end_time-start_time, response_data)

@app.route('/keywordsModel', methods=['POST'])
def keywords_endpoint():
    """
        Calculates risk score using a list of keywords.

        Request body: 
        {
            "namespace": string,
            "file_name": string
        }

        Response data: 
        {
            "financialScore": int,
            "operationalScore": int,
            "regulatoryScore": int,
            "reputationalScore": int
        }
    """
    start_time = time.time()
    missing_fields = [field for field in ['namespace', 'file_name'] if field not in request.json]
    if missing_fields:
        end_time = time.time()
        response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
        return create_response_model(200, "Success", "Risk assessment did not execute successfully.", end_time-start_time, response)
    target_content = extract_bson_text(request.json['file_name'], request.json['namespace'])
    target_keywords = custom_preprocessing(target_content)
    if target_keywords == False:
        end_time = time.time()
        return create_response_model(200, "Success", "Risk assessment did not execute successfully.", end_time-start_time)
    else:
        target_keywords = target_keywords.split()
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
    response_data = {'operationalScore': int(operational_score), 'regulatoryScore': int(regulatory_score), 'financialScore': int(financial_score), 'reputationalScore': int(reputational_score)}
    end_time = time.time()
    return create_response_model(200, "Success", "Keywords model executed successfully.", end_time-start_time, response_data)

@app.route('/xgboostModel', methods=['POST'])
def xgboost_endpoint():
    """
        Calculates risk score using a custom XGBoost model.

        Request body: 
        {
            "namespace": string,
            "file_name": string
        }

        Response data: 
        {
            "financialScore": int,
            "operationalScore": int,
            "regulatoryScore": int,
            "reputationalScore": int
        }
    """
    start_time = time.time()
    missing_fields = [field for field in ['namespace', 'file_name'] if field not in request.json]
    if missing_fields:
        end_time = time.time()
        response = {'error': f'Missing fields: {", ".join(missing_fields)}'}
        return create_response_model(200, "Success", "Risk assessment did not execute successfully.", end_time-start_time, response)
    target_document = extract_bson_text(request.json['file_name'], request.json['namespace'])
    training_data = get_df_from_azure_fileshare('training_data.csv', AZURE_FILES_CUSTOM_TRAINING_DIRECTORY)
    operational_score = custom_xgb(training_data, target_document, 'operational')
    regulatory_score = custom_xgb(training_data, target_document, 'regulatory')
    if operational_score is False or regulatory_score is False:
        end_time = time.time()
        return create_response_model(200, "Fail", "XGBoot model did not execute successfully.", end_time-start_time)
    reputational_score = 0
    financial_score = 0
    response_data = {'operationalScore': int(operational_score), 'regulatoryScore': int(regulatory_score), 'financialScore': int(financial_score), 'reputationalScore': int(reputational_score)}
    end_time = time.time()
    return create_response_model(200, "Success", "XGBoost model executed successfully.", end_time-start_time, response_data)

@app.route('/chat', methods=['POST'])
def chat():
    """
        Executes a chatbot query on the MongoDB store.

        Request body:
        {
            "namespace": string,
            "query": string,
            "file_name": string
        }

        Response data:
        {
            "query": string,
            "response": string
        }
    """
    start_time = time.time()
    user_input = request.json.get('query')
    file_name = request.json.get('file_name')
    namespace = request.json.get('namespace')
    file_content = extract_bson_text(file_name, namespace)
    chatbot = Chatbot(file_content)
    try:
        conversation_history = get_conversation_memory(namespace)
    except:
        conversation_history = []
    chatbot.conversation_history = conversation_history
    response = chatbot.chat(user_input)
    conversation_history = chatbot.conversation_history
    upload_conversation_memory(namespace, conversation_history)
    response_data = {'query': user_input, 'response': response}
    end_time = time.time()
    return create_response_model(200, "Success", "Chatbot model executed successfully.", end_time-start_time, response_data)

@app.route('/deleteConversationMemory', methods=['POST'])
def deleteConversationMemory():
    """
        Deletes conversation memory.

        Request body:
        {
            "namespace": string
        }
    """
    start_time = time.time()
    namespace = request.json.get('namespace')
    delete_conversation_memory(namespace)
    end_time = time.time()
    return create_response_model(200, "Success", "Conversation memory deleted successfully.", end_time-start_time)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
        Executes a chatbot query on the vector store.

        Request body:
        {
            "namespace": string,
            "query": string
        }

        Response data:
        {
            "query": string,
            "response": string
        }
    """
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
    """
        Adds training document to MongoDB collection and Azure file share.

        Request form data:
        file: file
        operational_score: text
        regulatory_score: text
    """
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
        upload_file_to_azure_fileshare(training_data_file_path, AZURE_FILES_CUSTOM_TRAINING_DIRECTORY)
        end_time = time.time()
        return create_response_model(200, "Success", "Updated training data successfully.", end_time-start_time)

@app.route('/embedder', methods=['POST'])
def embedder():
    """
        Adds file to the Pinecone index.

        Request body:
        {
            "fileName": string,
            "namespace": string
        }

        Response data:
        {
            "fileName": string,
            "namespace": string
        }
    """
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
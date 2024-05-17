from msilib.schema import File
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


nltk.download('stopwords')
nltk.download('punkt')

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

def custom_preprocessing(file_name, namespace):
    # Extract bson text
    text = extract_bson_text(file_name, namespace)

    # Make all lowercase and remove punctuation 
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # Tokenize using NLTK
    words = word_tokenize(text)

    # Remove redundant/stopwords using nltk
    stop_words = set(stopwords.words('english'))

    # Return a list of words
    words = [word for word in words if word not in stop_words]
    return words

def custom_xgb():
    # Initialize document to calculate risk
    target_document = "You're account has suspicious activity. Please verify location."

    # Initialize training document set
    documents = [
        # Low Risk (30 Examples)
        "Hi there! Hope you're doing well! We wanted to follow up on our conversation about [product name]. Any questions?".encode('utf-8'),
        "Your order #12345 has been shipped! Track it here: [link] (We apologize for any previous delays).".encode('utf-8'),
        "We're having a huge sale on all [product category] items! Check out the amazing deals: [link]".encode('utf-8'),
        "Thank you for being a valued customer! Let us know if you need any assistance with your recent purchase.".encode('utf-8'),
        "We're excited to announce new features in your account. Visit [link] to learn more.".encode('utf-8'),
        "Your subscription renewal was successful! Enjoy our services for another year.".encode('utf-8'),
        "Hi [Name], we have an exclusive offer just for you! Click here to avail: [link]".encode('utf-8'),
        "We'd love to hear your feedback on your recent purchase. Please take our survey: [link]".encode('utf-8'),
        "Reminder: Your upcoming appointment is scheduled for [date]. See you soon!".encode('utf-8'),
        "Hi! Just a friendly reminder to review your account settings: [link]".encode('utf-8'),
        "Your recent payment has been processed successfully. Thank you for your business.".encode('utf-8'),
        "Hi [Name], we've updated our terms of service. Please review them at your convenience.".encode('utf-8'),
        "Enjoy a 10% discount on your next purchase with code: THANKYOU10".encode('utf-8'),
        "We've added new items to our sale! Check them out here: [link]".encode('utf-8'),
        "Your loyalty points are expiring soon. Redeem them now: [link]".encode('utf-8'),
        "Hi [Name], thank you for your recent purchase! We hope you enjoy your new [product name].".encode('utf-8'),
        "Your gift card balance is ready to use. Shop now: [link]".encode('utf-8'),
        "Welcome to [Service Name]! We're glad to have you on board.".encode('utf-8'),
        "Hi [Name], we're offering free shipping on orders over $50! Shop now: [link]".encode('utf-8'),
        "Thank you for updating your account information. If you didn't make this change, please contact us.".encode('utf-8'),
        "We've received your return request. You'll be notified once it's processed.".encode('utf-8'),
        "Hi [Name], your feedback is important to us. Share your thoughts: [link]".encode('utf-8'),
        "Congratulations! You've been selected for an exclusive offer. Click here: [link]".encode('utf-8'),
        "Your order has been confirmed. We'll notify you once it's shipped.".encode('utf-8'),
        "Hi [Name], check out our latest blog post on [topic]: [link]".encode('utf-8'),
        "Your subscription has been successfully upgraded. Enjoy the new features!".encode('utf-8'),
        "We've added new products to our catalog. Browse now: [link]".encode('utf-8'),
        "Thank you for referring a friend! You've earned a reward: [link]".encode('utf-8'),
        "Hi [Name], here's a summary of your recent activity: [link]".encode('utf-8'),
        "Don't miss out on our summer sale! Up to 50% off on select items: [link]".encode('utf-8'),

        # Medium Risk (20 Examples)
        "We detected unusual login attempts from an unrecognized location. Please verify your recent activity: [link] (if it wasn't you)".encode('utf-8'),
        "Your account has been linked to suspicious activity. Please review your recent transactions and change your password if needed.".encode('utf-8'),
        "Your payment method for subscription [service name] has expired. Please update your payment information to avoid service interruptions.".encode('utf-8'),
        "We noticed some unusual activity in your account. Please log in to review: [link]".encode('utf-8'),
        "Your account settings were changed. If this wasn't you, please secure your account immediately.".encode('utf-8'),
        "Hi [Name], we detected multiple failed login attempts on your account. Please reset your password.".encode('utf-8'),
        "Your subscription payment failed. Please update your payment information to continue enjoying our services.".encode('utf-8'),
        "Your account shows unusual behavior. Verify your recent activity here: [link]".encode('utf-8'),
        "Hi [Name], we've detected some suspicious activity on your account. Please review your recent actions.".encode('utf-8'),
        "Your account security is our priority. Please verify your identity by clicking here: [link]".encode('utf-8'),
        "We've noticed an unusual login attempt. Please verify if it was you: [link]".encode('utf-8'),
        "Your recent login was from a new device. If this wasn't you, secure your account immediately.".encode('utf-8'),
        "We've updated your account security settings. If you didn't request this, please contact support.".encode('utf-8'),
        "Your recent activity shows a high number of login attempts. Please verify your identity.".encode('utf-8'),
        "Please confirm your recent transactions to ensure they were authorized by you.".encode('utf-8'),
        "Hi [Name], we've flagged some activity on your account as unusual. Please review here: [link]".encode('utf-8'),
        "Your subscription renewal failed. Update your payment method to continue using our services.".encode('utf-8'),
        "Unusual login detected. Please secure your account by resetting your password.".encode('utf-8'),
        "Hi [Name], we've detected suspicious activity. Review your recent logins here: [link]".encode('utf-8'),
        "Your account may be compromised. Please update your security settings immediately.".encode('utf-8'),

        # High Risk (50 Examples)
        "Your account information has been compromised in a recent data leak. We strongly advise changing your password immediately: [link] (This is a legitimate message from our company)".encode('utf-8'),
        "We noticed a significant increase in unauthorized login attempts originating from a foreign IP address. Please secure your account urgently: [link] (Do not click any links in suspicious emails)".encode('utf-8'),
        "A suspicious document containing malware was recently attached to an email sent from your account. Please scan your device for security threats.".encode('utf-8'),
        "We have identified a fraudulent transaction attempting to purchase high-value items from your account. Please contact us immediately to verify this activity.".encode('utf-8'),
        "This email appears to be a phishing attempt impersonating our company. Do not reply or click on any links. Report this email to us.".encode('utf-8'),
        "Your account has been compromised. Change your password immediately to prevent further unauthorized access.".encode('utf-8'),
        "We've detected malware activity linked to your account. Please scan your device and update your security settings.".encode('utf-8'),
        "Fraudulent transactions detected. Verify your recent activity and secure your account immediately.".encode('utf-8'),
        "Your personal information may have been exposed in a recent data breach. Take action to secure your account.".encode('utf-8'),
        "Multiple unauthorized login attempts detected. Reset your password and enable two-factor authentication.".encode('utf-8'),
        "High-risk login attempt detected from an unknown location. Secure your account now.".encode('utf-8'),
        "A significant data leak has affected your account. Change your password and monitor for suspicious activity.".encode('utf-8'),
        "Phishing attempt detected. Do not click any links or reply to this email. Report this incident to us.".encode('utf-8'),
        "Your account has been flagged for suspicious activity. Review and secure your account immediately.".encode('utf-8'),
        "Security alert: Unauthorized access detected. Reset your password and review your account activity.".encode('utf-8'),
        "We have identified potential fraud on your account. Verify your recent transactions to confirm their legitimacy.".encode('utf-8'),
        "High-risk alert: Your account information has been leaked. Take immediate action to secure your account.".encode('utf-8'),
        "Urgent: Your account is at risk. Update your security settings and scan for malware.".encode('utf-8'),
        "Suspicious activity detected. Review your recent logins and transactions for any unauthorized actions.".encode('utf-8'),
        "Critical security alert: Immediate action required to secure your account from potential threats.".encode('utf-8'),
        "High-priority notice: Update your security settings to prevent unauthorized access.".encode('utf-8'),
        "We detected multiple unauthorized transactions. Contact us immediately to resolve this issue.".encode('utf-8'),
        "Urgent security update required. Reset your password and enable additional security measures.".encode('utf-8'),
        "High-risk login attempt detected. Verify if this was you and update your account security.".encode('utf-8'),
        "Your account is at high risk due to recent suspicious activity. Secure your account immediately.".encode('utf-8'),
        "Multiple fraud attempts detected. Review your account activity and change your password.".encode('utf-8'),
        "Security breach detected. Immediate action required to protect your personal information.".encode('utf-8'),
        "Your account has been targeted in a recent cyber attack. Secure your account to prevent further damage.".encode('utf-8'),
        "High-risk alert: Unauthorized login attempts detected. Reset your password now.".encode('utf-8'),
        "Urgent: Review your account activity for potential security threats.".encode('utf-8'),
        "Immediate action required: Your account has been compromised. Secure your account now.".encode('utf-8'),
        "We've detected a security breach affecting your account. Change your password immediately.".encode('utf-8'),
        "Fraudulent activity detected on your account. Verify your recent transactions and secure your account.".encode('utf-8'),
        "Your account is at risk due to a recent data leak. Update your security settings now.".encode('utf-8'),
        "Multiple suspicious login attempts detected. Reset your password to prevent unauthorized access.".encode('utf-8'),
        "Critical alert: Unauthorized access detected. Secure your account and review your activity.".encode('utf-8'),
        "High-priority security alert: Take immediate action to protect your account from potential threats.".encode('utf-8'),
        "We've identified potential fraud. Verify your recent activity and update your security settings.".encode('utf-8'),
        "Urgent: Your account has been compromised. Change your password and review your security settings.".encode('utf-8'),
        "Suspicious document detected. Scan your device for malware and secure your account.".encode('utf-8'),
        "High-risk alert: Your account has been targeted. Update your security settings immediately.".encode('utf-8'),
        "Multiple unauthorized transactions detected. Contact us immediately to resolve this issue.".encode('utf-8'),
        "Immediate action required: Your personal information has been exposed in a data breach.".encode('utf-8'),
        "Security threat detected. Take immediate action to secure your account from potential risks.".encode('utf-8'),
        "High-priority alert: Unauthorized login attempts detected. Secure your account now.".encode('utf-8'),
        "Your account is at high risk due to recent suspicious activity. Update your security settings.".encode('utf-8'),
        "Critical security alert: Unauthorized access detected. Reset your password immediately.".encode('utf-8'),
        "Urgent: Your account has been compromised. Take immediate action to secure it.".encode('utf-8'),
        "We've identified potential fraud on your account. Verify your recent activity and update your security settings.".encode('utf-8')
    ]

    # Add the target document to document library
    documents.append(target_document)

    # Calculate TF-IDF scores and translate to dataframe
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_training_matrix = tfidf_vectorizer.fit_transform(documents)
    data = pd.DataFrame(tfidf_training_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Initialize risk scores of calculated documents and categorize
    data['target'] = [1] * 30 + [2] * 20 + [3] * 50
    data['target'] = pd.Categorical(data['target'])
    data['target'] = data['target'].cat.codes

    # Training data of documents (X is the TF-DF scores and y are the respective risk scores)
    X = data.drop(columns=['target'])
    y = data['target']
    X_train = data.drop(columns=['target']).iloc[0:len(data) - 1]
    y_train = data['target'].iloc[0:len(data) - 1]

    # Set prediction data (TF-IDF scores of target document)
    target_prediction_features = X.iloc[len(data) - 1:len(data)]

    # Train model and calculate prediction
    xgb_classifier_model = xgb.XGBClassifier()
    xgb_classifier_model.fit(X_train, y_train)
    predictions = xgb_classifier_model.predict(target_prediction_features)
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

# Pre trained with training documents beforehand, will not be retrained upon every execution -> tf-idf scores calculated -> model is fitted/trained here
def ra_custom():
    # Consume target document text from MongoDB -> Preprocessed (cleaned), tf idf scores are calculated

    # Trained model inputs target document TF-IDF scores and makes predictions
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
    print('STARTING XGB TESTING')
    print(custom_xgb())
    return render_template('main_page.html')

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)

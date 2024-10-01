from flask import Flask, request, jsonify, render_template
import os
import json
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from unit_checks import *
from linter_checks import *

# Load environment
load_dotenv()
app = Flask(__name__)
wsgi_app = app.wsgi_app

# Initialize environment variables
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

def get_checkId_list():
    result = []
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("unit_checks").select("id").execute()
    for row in response.data:
        result.append(row["id"])
    return result

def get_policyIds_by_checkId(checkId):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("policy_check_mapping").select("*").eq("check_id", checkId).execute()
    result = {
        "owasp": response.data[0].get("owasp_id"),
        "soc2": response.data[0].get("soc2_id")
    }
    return {k: v for k, v in result.items() if v is not None}

def get_policies_by_checkId(checkId):
    result = {}
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("policy_check_mapping").select("*").eq("check_id", checkId).execute()
    if response.data[0].get("owasp_id"):
        owasp = []
        for id in response.data[0].get("owasp_id"):
            owasp.append(get_policy_by_policyId(id, "owasp"))
        result["owasp"] = owasp
    if response.data[0].get("soc2_id"):
        soc2 = []
        for id in response.data[0].get("soc2_id"):
            soc2.append(get_policy_by_policyId(id, "soc2"))
        result["soc2"] = soc2
    return result

def get_function_by_checkId(checkId):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("unit_checks").select("function").eq("id", checkId).execute()
    return response.data[0]['function']

def get_checkIds_by_policyId(policyId, policyType):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    column_name = f"{policyType}_id"
    response = supabase.table("policy_check_mapping").select("*").execute()
    return [row["check_id"] for row in response.data if policyId in row.get(column_name, "")]

def get_policy_by_policyId(policyId, policyType):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table(policyType).select("policy").eq("id", policyId).execute()
    return response.data[0]['policy']

def get_fix_by_checkId(checkId):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("unit_checks").select("fix").eq("id", checkId).execute()
    return response.data[0]['fix']

def run_policy_check(filestream, language):
    code_raw = filestream.read().decode('utf-8')
    # code_tree = ast.parse(code_raw)
    failed_check_list = []
    passed_check_list = []
    checkId_list = get_checkId_list()
    for check in checkId_list:

        # Whether you are running once or all at once
        # result = check_function(code_tree, language)
        try:
            check_function = globals()[get_function_by_checkId(check)]
            result = check_function(code_raw, language)
        except:
            continue
        if result != [] and result != False and result != None:
            failed_policies = get_policyIds_by_checkId(check)
            for line in result:
                entry = { "failed_check": get_function_by_checkId(check), "line_number": line, "policies": get_policies_by_checkId(check), "fix": get_fix_by_checkId(check) }
                failed_check_list.append(entry)
        elif result == [] or result == False:
            entry = { "passed_check": get_function_by_checkId(check) }
            passed_check_list.append(entry)
    failed_check_list = sorted(failed_check_list, key=lambda x: x["line_number"])
    return { 'passed': passed_check_list, 'failed': failed_check_list }

def code_analysis(files):
    failed_check_list = []
    passed_check_list = []
    for file in files:
        print(f'Checking {file.filename}------------------------------------------------------------------')
        filename, file_extension = os.path.splitext(file.filename)
        language = ''
        if file_extension == '.py':
            language = 'python'
        elif file_extension == '.js':
            language = 'javascript'
        elif file_extension == '.java':
            language = 'java'
        elif file_extension == '.cs':
            language = 'c#'
        elif file_extension == '.cpp':
            language = 'c++'
        elif file_extension == '.ts':
            language = 'typescript'
        elif file_extension == '.ps1':
            language = 'powershell'
        file_result = run_policy_check(file.stream, language)
        if file_result['failed'] != []:
            failed_entry = { "file_name": file.filename, "issues": file_result['failed']}
            failed_check_list.append(failed_entry)
        else:
            passed_entry = { "file_name": file.filename }
            passed_check_list.append(passed_entry)
        print(f'Finished checking {file.filename}------------------------------------------------------------------')
    return { "failed": failed_check_list, "passed": passed_check_list}

def linter_analysis(files):
    result = {}
    for file in files:
        filename, file_extension = os.path.splitext(file.filename)
        language = ''
        if file_extension == '.py':
            result[file.filename] = python_linter_check(file)
            continue
        elif file_extension == '.js':
            language = 'javascript'
            continue
        elif file_extension == '.java':
            language = 'java'
            continue
        elif file_extension == '.cs':
            language = 'c#'
            continue
        elif file_extension == '.cpp':
            language = 'c++'
            continue
        elif file_extension == '.ts':
            language = 'typescript'
            continue
        elif file_extension == '.ps1':
            language = 'powershell'
            continue
    return result

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
@app.route('/linterAnalysis', methods=['POST'])
def linter_analysis_endpoint():
    MAX_FILE_SIZE_KB = 1000000
    start_time = time.time()
    files = request.files.getlist('file')
    total_size = 0
    for file in files:
        file.seek(0, 2)
        total_size += file.tell()
        file.seek(0) 
    if total_size > (MAX_FILE_SIZE_KB * 1024):
        return create_response_model(400, "Error", "Total file size exceeds the 1GB limit.", time.time() - start_time, None)
    result = linter_analysis(files)
    end_time = time.time()
    return create_response_model(200, "Success", "Linter code analysis executed successfully.", end_time-start_time, result)

@app.route('/codeAnalysis', methods=['POST'])
def code_analysis_endpoint():
    MAX_FILE_SIZE_KB = 1000000
    start_time = time.time()
    files = request.files.getlist('file')
    total_size = 0
    for file in files:
        file.seek(0, 2)
        total_size += file.tell()
        file.seek(0) 
    if total_size > (MAX_FILE_SIZE_KB * 1024):
        return create_response_model(400, "Error", "Total file size exceeds the 1GB limit.", time.time() - start_time, None)
    result = code_analysis(files)
    end_time = time.time()
    return create_response_model(200, "Success", "Code analysis executed successfully.", end_time-start_time, result)

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
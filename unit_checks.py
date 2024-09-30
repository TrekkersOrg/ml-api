import ast
from unit_subchecks import *
from unit_helpers import *

def check_sql_injection(codestream, language, deep_response=True):
    print('SQL Injection Check: Started')
    r1 = r1_extract_sql_strings(codestream)
    if r1 == False:
        return False
    r2 = r2_check_sql_concatenation(r1, language)
    r3 = r3_check_unprepared_sql(r1, language)
    result = False
    if r2 is not False and r3 is not False:
        result = r2 + r3
        return pick_response(result, deep_response, 'SQL Injection Check')
    elif r2 is not False:
        return pick_response(r2, deep_response, 'SQL Injection Check')
    elif r3 is not False:
        return pick_response(r3, deep_response, 'SQL Injection Check')
    return False

def check_xss(codestream, language, deep_response=True):
    print("Cross-Site Scripting Check: Started")
    r4 = r4_check_output_concatenation(codestream, language)
    r5 = r5_check_htmljs_concatenation(codestream, language)
    if r4 is not False and r5 is not False:
        result = r4 + r5
        return pick_response(result, deep_response, 'Cross-Site Scripting Check')
    if r4 is not False:
        return pick_response(r4, deep_response, 'Cross-Site Scripting Check')
    if r5 is not False:
        return pick_response(r5, deep_response, 'Cross-Site Scripting Check')
    return False

def check_insecure_deserialization(codestream, language, deep_response=True):
    print("Insecure Deserialization Check: Started")
    r6 = r6_check_invalid_deserialization(codestream, language)
    if r6 is not False:
        return pick_response(r6, deep_response, 'Insecure Deserialization Check')
    return

def check_security_misconfiguration(codestream, language, deep_response=True):
    print("Security Misconfiguration Check: Started")
    r7 = r7_check_harcoded_sensitive_information(codestream, language)
    r8 = r8_check_unauthorized_endpoints(codestream, language)
    r9 = r9_check_dev_env_misconfigurations(codestream, language)
    if r7 is not False and r8 is not False and r9 is not False:
        result = r7 + r8 + r9
        return pick_response(result, deep_response, 'Security Misconfiguration Check')
    if r8 is not False and r9 is not False:
        result = r8 + r9
        return pick_response(result, deep_response, 'Security Misconfiguration Check')
    if r7 is not False and r9 is not False:
        result = r7 + r9
        return pick_response(result, deep_response, 'Security Misconfiguration Check')
    if r7 is not False and r8 is not False:
        result = r7 + r8
        return pick_response(result, deep_response, 'Security Misconfiguration Check')
    if r7 is not False:
        return pick_response(r7, deep_response, 'Security Misconfiguration Check')
    if r8 is not False:
        return pick_response(r8, deep_response, 'Security Misconfiguration Check')
    if r9 is not False:
        return pick_response(r9, deep_response, 'Security Misconfiguration Check')
    return False

def check_sensitive_data_exposure(codestream, language, deep_response=True):
    print("Sensitive Data Exposure Check: Started")
    r7 = r7_check_harcoded_sensitive_information(codestream, language)
    r10 = r10_check_imported_plaintext(codestream, language)
    if r7 is not False and r10 is not False:
        result = r7 + r10
        return pick_response(result, deep_response, 'Sensitive Data Exposure Check')
    if r7 is not False:
        return pick_response(r7, deep_response, 'Sensitive Data Exposure Check')
    if r10 is not False:
        return pick_response(r10, deep_response, 'Sensitive Data Exposure Check')
    return False

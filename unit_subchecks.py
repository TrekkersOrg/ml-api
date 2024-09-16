import re

def r1_extract_sql_strings(code: str):
    print("R1: Started")
    SQL_KEYWORDS = [
        r'\bSELECT\b', r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b',
        r'\bFROM\b', r'\bWHERE\b', r'\bJOIN\b', r'\bINTO\b',
        r'\bVALUES\b', r'\bSET\b', r'\bAND\b', r'\bOR\b'
    ]
    SQL_PATTERN = (
        r'\'(?:[^\']*?\b(?:' + '|'.join(SQL_KEYWORDS) + r')\b[^\']*?)\''
        r'|"[^"]*?\b(?:' + '|'.join(SQL_KEYWORDS) + r')\b[^"]*?"'
        r'|`[^`]*?\b(?:' + '|'.join(SQL_KEYWORDS) + r')\b[^`]*?`'
        r'|"""(?:[^"]*?\b(?:' + '|'.join(SQL_KEYWORDS) + r')\b[^"]*?)"""'
        r'|\'\'\'(?:[^\'\'\'*?\b(?:' + '|'.join(SQL_KEYWORDS) + r')\b[^\'\'\'*?])*?\'\'\''
        r'|@\s*"[^"]*?\b(?:' + '|'.join(SQL_KEYWORDS) + r')\b[^"]*?"'
        r'|@\s*\'[^\'\'*?\b(?:' + '|'.join(SQL_KEYWORDS) + r')\b[^\'\'*?]*?\''
    )
    CONCATENATION_PATTERN = r'(\+\s*[\w\d_]+)'
    CONCATENATION_STRING_PATTERN = r'([\w\s]*\+\s*[\w\d_]+)'
    results = []
    current_line = 1
    code_lines = code.splitlines()
    combined_code = ""
    in_multiline_string = False
    multiline_start_line = None
    in_ps_here_string = False
    ps_here_string_start_line = None
    for line in code_lines:
        if '"""' in line or "'''" in line:
            if not in_multiline_string:
                in_multiline_string = True
                multiline_start_line = current_line
                combined_code += line
            else:
                in_multiline_string = False
                combined_code += line
                sql_queries = re.findall(SQL_PATTERN, combined_code, re.IGNORECASE | re.DOTALL)
                for query in sql_queries:
                    cleaned_query = re.sub(r'\s+', ' ', combined_code.replace('\n', ' ')).strip()
                    if cleaned_query:
                        concatenation_matches = re.findall(CONCATENATION_STRING_PATTERN, cleaned_query)
                        variables = []
                        for match in concatenation_matches:
                            variables.extend(re.findall(CONCATENATION_PATTERN, match))
                        results.append((multiline_start_line, cleaned_query, variables))
                combined_code = ""
        elif '@"' in line or '@\n"' in line or "@'" in line or '@\n\'' in line:
            if not in_ps_here_string:
                in_ps_here_string = True
                ps_here_string_start_line = current_line
                combined_code += line
            else:
                in_ps_here_string = False
                combined_code += line
                sql_queries = re.findall(SQL_PATTERN, combined_code, re.IGNORECASE | re.DOTALL)
                for query in sql_queries:
                    cleaned_query = re.sub(r'\s+', ' ', combined_code.replace('\n', ' ')).strip()
                    if cleaned_query:
                        concatenation_matches = re.findall(CONCATENATION_STRING_PATTERN, cleaned_query)
                        variables = []
                        for match in concatenation_matches:
                            variables.extend(re.findall(CONCATENATION_PATTERN, match))
                        results.append((ps_here_string_start_line, cleaned_query, variables))
                combined_code = ""
        else:
            if in_multiline_string or in_ps_here_string:
                combined_code += '\n' + line
            else:
                sql_queries = re.findall(SQL_PATTERN, line, re.IGNORECASE | re.DOTALL)
                for query in sql_queries:
                    cleaned_query = re.sub(r'\s+', ' ', line.replace('\n', ' ')).strip()
                    if cleaned_query:
                        concatenation_matches = re.findall(CONCATENATION_STRING_PATTERN, cleaned_query)
                        variables = []
                        for match in concatenation_matches:
                            variables.extend(re.findall(CONCATENATION_PATTERN, match))
                        results.append((current_line, cleaned_query, variables))
        current_line += 1
    if combined_code:
        sql_queries = re.findall(SQL_PATTERN, combined_code, re.IGNORECASE | re.DOTALL)
        for query in sql_queries:
            cleaned_query = re.sub(r'\s+', ' ', combined_code.replace('\n', ' ')).strip()
            if cleaned_query:
                concatenation_matches = re.findall(CONCATENATION_STRING_PATTERN, cleaned_query)
                variables = []
                for match in concatenation_matches:
                    variables.extend(re.findall(CONCATENATION_PATTERN, match))
                results.append((multiline_start_line if in_multiline_string else ps_here_string_start_line, cleaned_query, variables))
    if not results:
        print('R1: Failed')
        return False
    print('R1: Passed')
    return results

def r2_check_sql_concatenation(sql_queries, language):
    print('R2: Started')
    results = []
    patterns = {
        'python': [r'f"[^"]*\{[^}]*\}[^"]*"', r'\+\s*[\'"].*?\s*\+\s*\w+.*?[\'"]'],
        'javascript': [r'\$\{[^}]*\}', r'\+\s*["\']\s*(?:\s*\+\s*\w+)+\s*["\']'],
        'java': [r'\+\s*["\'].*?\s*\+\s*\w+.*?["\']'],
        'c#': [r'\$\{[^}]*\}'],
        'c++': [r'\+\s*["\'].*?\s*\+\s*\w+.*?["\']'],
        'typescript': [r'\$\{[^}]*\}'],
        'powershell': [r'\$[a-zA-Z_]\w*\s*=\s*".*?\$[a-zA-Z_]\w+.*?"']
    }
    for line_number, query, variables in sql_queries:
        if len(variables) > 0:
            results.append((line_number, query))
        for pattern in patterns[language]:
            if re.search(pattern, query, re.IGNORECASE):
                results.append((line_number, query))
                break
    if not results:
        print('R2: Failed')
        return False
    print('R2: Passed')
    return results

def r3_check_unprepared_sql(sql_queries, language):
    print('R3: Started')
    patterns = {
        'python': [
            r'f"[^"]*{[^}]*}[^"]*"',  # Detect f-strings containing variables
            r'cursor\.execute\(f"[^"]*{[^}]*}[^"]*"\)',  # Detect f-strings with cursor.execute
            r'cursor\.execute\(["\'].*\+\s*\w+.*["\']\)',  # Using + for concatenation
            r'cursor\.execute\(["\'].*%\w+.*["\']\)',  # Using % for string formatting
        ],
        'javascript': [
            r'\bdb\.query\(["\'].*\+\s*\w+.*["\']\)',  # Using + for concatenation
            r'\bdb\.query\(["\'].*\$\{[^}]*\}.*["\']\)',  # Using ${} for template literals
        ],
        'typescript': [
            r'\bdb\.query\(["\'].*\+\s*\w+.*["\']\)',  # Using + for concatenation
            r'\bdb\.query\(["\'].*\$\{[^}]*\}.*["\']\)',  # Using ${} for template literals
        ],
        'java': [
            r'Statement\s+\w+\s*=\s*connection\.createStatement\(\);\s*\w+\.executeQuery\(["\'].*\+\s*\w+.*["\']\)',  # Using + for concatenation
        ],
        'c#': [
            r'SqlCommand\s+\w+\s*=\s*new\s+SqlCommand\(["\'].*\+\s*\w+.*["\']\)',  # Using + for concatenation
        ],
        'c++': [
            r'std::string\s+\w+\s*=\s*"SELECT\s+.*\+\s*\w+.*"',  # Using + for concatenation
        ],
        'powershell': [
            r'Invoke-Sqlcmd\s+-Query\s+"SELECT\s+\*\s+FROM\s+.*\$\(\$\w+\)"',  # Direct concatenation
        ]
    }
    language = language.lower()
    relevant_patterns = patterns[language]
    results = []
    for line_number, query, variables in sql_queries:
        for pattern in relevant_patterns:
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                results.append((line_number, query, variables))
                break
    if not results:
        print('R3: Failed')
        return False
    print('R3: Passed')
    return results

def r4_check_output_concatenation(code, language):
    print('R4: Started')
    patterns = {
        'python': [
            r'return\s+f?"[^"]*\{[^}]*\}[^"]*"',  # Detect f-strings with user input
            r'return\s+.*\+\s*\w+.*',  # Concatenation using + operator
        ],
        'javascript': [
            r'return\s+.*\+\s*\w+',  
            r'return\s+`[^`]*\$\{[^}]*\}[^`]*`' 
        ],
        'typescript': [
            r'return\s+.*\+\s*\w+',  
            r'return\s+`[^`]*\$\{[^}]*\}[^`]*`'   
        ],
        'java': [
            r'return\s*".*"\s*\+\s*\w+',  
            r'return\s*\w+\s*\+\s*".*"', 
            r'return\s*".*"\s*\+\s*".*"', 
            r'return\s*String\s+\w+\s*=\s*".*"\s*\+\s*\w+', 
            r'System\.out\.(println|print)\s*\(.*\+\s*\w+.*\)' 
        ],
        'c#': [
            r'Response\.Write\(".*"\s*\+\s*\w+',
            r'return\s+["\'].*\s*\+\s*\w+.*["\']',   
            r'SqlCommand\s+\w+\s*=\s*new\s+SqlCommand\(["\'].*\+\s*\w+.*["\']\)',   
        ],
        'c++': [
            r'std::cout\s*<<\s*".*"\s*\+\s*\w+',
            r'std::cout\s*<<\s*".*"\s*<<\s*\w+',  
        ],
        'powershell': [
            r'Write-Host\s+"[^"]*\$[^"]*"',
            r'Write-Host\s*".*"\s*\+\s*\$\w+',  
            r'Write-Output\s*".*"\s*\+\s*\$\w+', 
            r'\$\w+\s*=\s*".*"\s*\+\s*\$\w+',    
        ]
    }

    results = []
    language = language.lower()
    if language not in patterns:
        print(f"R4: Language '{language}' not supported")
        return False

    relevant_patterns = patterns[language]
    code_lines = code.splitlines()
    current_line = 1

    for line in code_lines:
        for pattern in relevant_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                results.append((current_line, line.strip()))
                break
        current_line += 1

    if not results:
        print('R4: Failed')
        return False

    print('R4: Passed')
    return results


def r5_check_htmljs_concatenation(code, language):
    print('R5: Started')
    patterns = {
        'python': [
            r'\bhtml\s*=\s*".*"\s*\+\s*\w+',  # Concatenation in variable assignment
            r'\bdocument\.write\s*\(\s*".*"\s*\+\s*\w+',  # Concatenation in document.write
            r'\breturn\s*".*"\s*\+\s*\w+',  # Return statement with concatenation
            r'\bprint\s*\(\s*".*"\s*\+\s*\w+',  # Print statement with concatenation
            r'\b"\s*\+\s*\w+',  # General concatenation pattern
            r'\bhtml\s*=\s*".*"\s*\+\s*\w+',  # Another variable assignment pattern
            r'\b".*"\s*\+\s*".*"\s*\+\s*\w+',  # Multiple concatenation cases
            r'\bstr\s*=\s*".*"\s*\+\s*\w+',  # Concatenation in string assignments
            r'\b".*"\s*\+\s*\w+\s*\+\s*".*"',  # Complex concatenation cases
        ],
        'javascript': [
            r'document\.write\s*\(\s*".*"\s*\+\s*\w+.*\)',  # HTML/JavaScript concatenation with document.write
            r'document\.getElementById\s*\(\s*".*"\s*\)\.innerHTML\s*=\s*".*"\s*\+\s*\w+.*',  # HTML/JavaScript concatenation with innerHTML
            r'return\s*".*"\s*\+\s*\w+',   # Concatenation in return statements
            r'return\s+`[^`]*\$\{[^}]*\}[^`]*`',  # Template literals with variables
        ],
        'typescript': [
            r'let\s+\w+\s*=\s*".*"\s*\+\s*\w+',  # Concatenation when defining a variable
            r'(\.innerHTML|document\.write)\s*=\s*".*"\s*\+\s*\w+',  # HTML/JavaScript concatenation with innerHTML or document.write
            r'(\.innerHTML|document\.write)\s*\(\s*".*"\s*\+\s*\w+',  # HTML/JavaScript concatenation in function calls
            r'return\s*".*"\s*\+\s*\w+',   # Concatenation in return statements
            r'return\s+`[^`]*\$\{[^}]*\}[^`]*`',  # Template literals with variables
        ],
        'java': [
            r'return\s*".*"\s*\+\s*\w+',   # Concatenation in return statements
            r'return\s*\w+\s*\+\s*".*"',   # Concatenation with variables and strings
            r'return\s*".*"\s*\+\s*".*"',   # General string concatenation
            r'return\s*String\s+\w+\s*=\s*".*"\s*\+\s*\w+',   # Concatenation with String variables
            r'System\.out\.(println|print)\s*\(.*\+\s*\w+.*\)'  # Output with concatenation
        ],
        'c#': [
            r'return\s*["\'].*?\s*\+\s*\w+.*?["\']',   # Concatenation in return statements
            r'return\s*["\'].*?\s*\+\s*["\'].*?["\']',  # Concatenation of strings in return statements
            r'Console\.WriteLine\s*\(".*?\+\s*\w+.*?"\)',  # Concatenation in Console.WriteLine
            r'Console\.Write\s*\(".*?\+\s*\w+.*?"\)',      # Concatenation in Console.Write
            r'SqlCommand\s+\w+\s*=\s*new\s+SqlCommand\(["\'].*?\+\s*\w+.*?["\']\)',  # SQL command concatenation
            r'return\s*".*?\s*\+\s*\w+.*?"',  # General concatenation pattern in return statements
            r'[\s\S]*\+\s*\w+.*'  # General concatenation with any variable
        ],
        'c++': [
            r'std::cout\s*<<\s*["\'].*?\s*\+\s*\w+.*?',  # Concatenation using << operator
            r'std::cout\s*<<\s*["\'].*?\s*<<\s*\w+',    # Concatenation using << operator
        ],
        'powershell': [
            r'Write-Host\s+".*\+\s*\$\w+.*?"',   # Concatenation in Write-Host
            r'Write-Host\s*".*"\s*\+\s*\$\w+',   # Concatenation in Write-Host with variables
            r'Write-Output\s*".*"\s*\+\s*\$\w+', # Concatenation in Write-Output
            r'\$\w+\s*=\s*".*"\s*\+\s*\$\w+',   # Concatenation in variable assignments
        ]
    }

    results = []
    language = language.lower()
    if language not in patterns:
        print(f"R5: Language '{language}' not supported")
        return False

    relevant_patterns = patterns[language]
    code_lines = code.splitlines()
    current_line = 1

    for line in code_lines:
        line = line.strip()
        for pattern in relevant_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                results.append((current_line, line))
                break
        current_line += 1

    if not results:
        print('R5: Failed')
        return False

    print('R5: Passed')
    return results

def r6_check_invalid_deserialization(code, language):
    print('R6: Started')
    patterns = {
        "python": [r"\b(pickle\.loads|pickle\.load)\b"],
        "javascript": [r"\bJSON\.parse\b"],
        "typescript": [r"\bJSON\.parse\b"],
        "java": [r"\bObjectInputStream\b.*\.readObject\b"],
        "csharp": [r"\bBinaryFormatter\b.*\.Deserialize\b"],
        "cpp": [r"\bifstream\b.*\.read\b"],
        "powershell": [r"\bConvertFrom-Json\b"]
    }

    results = []
    language = language.lower()
    if language not in patterns:
        print(f"R6: Language '{language}' not supported")
        return False

    relevant_patterns = patterns[language]
    code_lines = code.splitlines()
    current_line = 1

    for line in code_lines:
        line = line.strip()
        for pattern in relevant_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                results.append((current_line, line))
                break
        current_line += 1
    print(results)

    if not results:
        print('R6: Failed')
        return False

    print('R6: Passed')
    return results
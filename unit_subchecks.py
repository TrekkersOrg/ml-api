import re

def r1_extract_sql_strings(code: str):
    print("R1 (Extract SQL Strings): Started")
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
        print('R1 (Extract SQL Strings): No SQL strings found')
        print('R1 (Extract SQL Strings): Finished')
        return False
    print('R1 (Extract SQL Strings): SQL string(s) found')
    print('R1 (Extract SQL Strings): Finished')
    return results

def r2_check_sql_concatenation(sql_queries, language):
    print('R2 (SQL Concatenation): Started')
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
        print('R2 (SQL Concatenation): No SQL concatenation errors')
        print('R2 (SQL Concatenation): Finished')
        return False
    print('R2 (SQL Concatenation): SQL concatenation error(s) found')
    print('R2 (SQL Concatenation): Finished')
    return results

def r3_check_unprepared_sql(sql_queries, language):
    print('R3 (Unprepared SQL): Started')
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
        print('R3 (Unprepared SQL): No unprepared SQL statements found')
        print('R3 (Unprepared SQL): Finished')
        return False
    print('R3 (Unprepared SQL): Unprepared SQL statement(s) found')
    print('R3 (Unprepared SQL): Finished')
    return results

def r4_check_output_concatenation(code, language):
    print('R4 (Output Concatenation): Started')
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
        print(f"R4 (Output Concatenation): Language '{language}' not supported")
        print(f"R4 (Output Concatenation): Finished")
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
        print('R4 (Output Concatenation): No output concatenations found')
        print('R4 (Output Concatenation): Finished')
        return False
    print('R4 (Output Concatenation): Output concatenation(s) found')
    print('R4 (Output Concatenation): Finished')
    return results


def r5_check_htmljs_concatenation(code, language):
    print('R5 (HTML/JS Concatenation): Started')
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
        print(f"R5 (HTML/JS Concatenation): Language '{language}' not supported")
        print(f"R5 (HTML/JS Concatenation): Finished")
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
        print('R5 (HTML/JS Concatenation): No HTML/JS concatenations found')
        print('R5 (HTML/JS Concatenation): Finished')
        return False
    print('R5 (HTML/JS Concatenation): HTML/JS concatenation(s) found')
    print('R5 (HTML/JS Concatenation): Finished')
    return results

def r6_check_invalid_deserialization(code, language):
    print('R6 (Invalid Deserialization): Started')
    patterns = {
        "python": [r"\b(pickle\.loads|pickle\.load)\b"],
        "javascript": [r"\bJSON\.parse\b"],
        "typescript": [r"\bJSON\.parse\b"],
        "java": [r"\bObjectInputStream\b.*\.readObject\b"],
        "c#": [r"\bBinaryFormatter\b.*\.Deserialize\b"],
        "c++": [r"\bifstream\b.*\.read\b"],
        "powershell": [r"\bConvertFrom-Json\b"]
    }

    results = []
    language = language.lower()
    if language not in patterns:
        print(f"R6 (Invalid Deserialization): Language '{language}' not supported")
        print(f"R6 (Invalid Deserialization): Finished")
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
        print(f"R6 (Invalid Deserialization): No invalid deserializations found")
        print('R6 (Invalid Deserialization): Finished')
        return False
    print(f"R6 (Invalid Deserialization): Invalid deserialization(s) found")
    print('R6 (Invalid Deserialization): Finished')
    return results

def r7_check_harcoded_sensitive_information(code, language):
    print('R7 (Hardcoded Sensitive Info): Started')    
    patterns = {
        "python": [
            r"(api_key|access_key|secret|token|password)\s*=\s*['\"].+['\"]",
            r"['\"](api_key|access_key|secret|token|password)['\"]\s*:\s*['\"].+['\"]"  # JSON-like structures
        ],
        "javascript": [
            r"(const|let|var)\s+(apiKey|accessKey|secret|token|password)\s*=\s*['\"].+['\"]",
            r"['\"](apiKey|accessKey|secret|token|password)['\"]\s*:\s*['\"].+['\"]"  # JSON-like structures
        ],
        "typescript": [
            r"(const|let|var)\s+(apiKey|accessKey|secret|token|password)\s*=\s*['\"].+['\"]",
            r"['\"](apiKey|accessKey|secret|token|password)['\"]\s*:\s*['\"].+['\"]"  # JSON-like structures
        ],
        "java": [
            r"(String|final)\s+(apiKey|accessKey|secret|token|password)\s*=\s*['\"].+['\"]",
            r"['\"](apiKey|accessKey|secret|token|password)['\"]\s*:\s*['\"].+['\"]"  # JSON-like structures
        ],
        "c#": [
            r"(string|var)\s+(connectionString|apiKey|accessKey|secret|token|password)\s*=\s*['\"].*Password=.*['\"]",
            r"(string|var)\s+(apiKey|accessKey|secret|token|password)\s*=\s*['\"].+['\"]"
        ],
        "c++": [
            r"std::string\s+(apiKey|accessKey|secret|token|password)\s*=\s*['\"].+['\"]"
        ],
        "powershell": [
            r"\$(apiKey|accessKey|secret|token|password)\s*=\s*['\"].+['\"]",
            r"\$(apiKey|accessKey|secret|token|password)\s*:\s*['\"].+['\"]"  # HashTable or JSON-like structures
        ]
    }
    
    results = []
    language = language.lower()
    
    if language not in patterns:
        print(f"R7 (Hardcoded Sensitive Info): Language '{language}' not supported")
        print(f"R7 (Hardcoded Sensitive Info): Finished")
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
        print(f"R7 (Hardcoded Sensitive Info): No hardcoded sensitive information found")
        print('R7 (Hardcoded Sensitive Info): Finished')
        return False
    print(f"R7 (Hardcoded Sensitive Info): Hardcoded sensitive information found")
    print('R7 (Hardcoded Sensitive Info): Finished')
    return results

def r8_check_unauthorized_endpoints(code, language):
    print('R8 (Unauthorized Endpoints): Started')

    # Regex patterns for detecting endpoint definitions
    patterns = {
        "python": [
            r"@app\.route\((?:'|\")(/[^'\"]*)(?:'|\")\)",  # Flask route
        ],
        "javascript": [
            r"app\.(get|post|put|delete)\((?:'|\")(/[^'\"]+)(?:'|\")",  # Express.js get/post/put/delete
            r"router\.(get|post|put|delete)\((?:'|\")(/[^'\"]+)(?:'|\")",  # Express router
        ],
        "typescript": [
            r"app\.(get|post|put|delete)\((?:'|\")(/[^'\"]+)(?:'|\")",  # TypeScript app routes
            r"router\.(get|post|put|delete)\((?:'|\")(/[^'\"]+)(?:'|\")",  # TypeScript router
            r"@(Get|Post|Put|Delete)\((?:'|\")(/[^'\"]+)(?:'|\")\)",      # More generic route for TypeScript
        ],
        "java": [
            r"@GetMapping\((?:'|\")(/[^'\"]+)(?:'|\")\)",  # Spring GetMapping
            r"@RequestMapping\((?:'|\")(/[^'\"]+)(?:'|\")\)",  # Spring RequestMapping
        ],
        "c#": [
            r"\[Http(Get|Post|Put|Delete)\]",  # ASP.NET Core Http methods
            r"\[Route\((?:'|\")(/[^'\"]+)(?:'|\")\)\]",  # ASP.NET with Route attribute
        ],
        "c++": [
            r"void\s+\w+\(.*\)\s*{",  # C++ function definitions (simple case)
        ],
        "powershell": [
            r"function\s+\w+\s*{",  # PowerShell function definitions
        ]
    }

    # Patterns to check for missing authorization checks
    auth_check_patterns = {
        "python": r"@login_required|@requires_auth",  # Ensure authorization is present
        "javascript": r"req\.isAuthenticated\(\)",  # Ensure isAuthenticated is present
        "typescript": r"req\.isAuthenticated\(\)|@UseGuards\(\)",  # isAuthenticated() or UseGuards() for Nest.js, TypeScript
        "java": r"@PreAuthorize|@Secured|@RolesAllowed",  # Ensure @PreAuthorize or @Secured is present
        "c#": r"\[Authorize\]",  # Ensure [Authorize] is present
        "c++": r"isAuthenticated\(\)",  # Ensure isAuthenticated() check is present
        "powershell": r"\$User\.IsAuthenticated"  # Ensure $User.IsAuthenticated check is present
    }

    results = []
    language = language.lower()

    # Check if the language is supported
    if language not in patterns:
        print(f"R8 (Unauthorized Endpoints): Language '{language}' not supported")
        print(f"R8 (Unauthorized Endpoints): Finished")
        return False

    relevant_patterns = patterns[language]
    auth_check_pattern = auth_check_patterns[language]
    code_lines = code.splitlines()
    current_line = 1

    for line in code_lines:
        line = line.strip()
        # Check if the line contains an endpoint definition
        for pattern in relevant_patterns:
            if re.search(pattern, line):


                # Check if the line does NOT contain the necessary authorization check
                if not re.search(auth_check_pattern, line):
                    results.append((current_line, line))
                break  # Stop checking once we find a matching pattern
        current_line += 1

    if not results:
        print(f"R8 (Unauthorized Endpoints): No unauthorized endpoints found")
        print('R8 (Unauthorized Endpoints): Finished')
        return False
    
    print(f"R8 (Unauthorized Endpoints): Unauthorized endpoints found")
    print('R8 (Unauthorized Endpoints): Finished')
    return results

def r9_check_dev_env_misconfigurations(code, language):
    print('R9 (Dev Env Misconfigurations): Started')

    # Regex patterns for detecting insecure development settings
    patterns = {
        "python": [
            r"DEBUG\s*=\s*True",  # Python debug mode
        ],
        "javascript": [
            r"app\.use\(\s*morgan\('dev'\)\s*\)",  # JavaScript/TypeScript morgan dev mode
        ],
        "typescript": [
            r"app\.use\(\s*morgan\('dev'\)\s*\)",  # TypeScript morgan dev mode
        ],
        "java": [
            r"@Bean\s*\n\s*public\s+ServerEndpointExporter\s*\(.*\)",  # Match @Bean and ServerEndpointExporter method declaration
            r"return\s+new\s+ServerEndpointExporter\s*\(\);",  # Match return statement for ServerEndpointExporter
        ],
        "c#": [
            r"app\.UseDeveloperExceptionPage\(\)",  # C# developer exception page
        ],
        "c++": [
            r"#ifdef\s+DEBUG",  # C++ debug macros
        ],
        "powershell": [
            r"\$DebugPreference\s*=\s*['\"]Continue['\"]",  # PowerShell DebugPreference set to continue
        ]
    }

    results = []
    language = language.lower()

    # Check if the language is supported
    if language not in patterns:
        print(f"R9 (Dev Env Misconfigurations): Language '{language}' not supported")
        print(f"R9 (Dev Env Misconfigurations): Finished")
        return False

    relevant_patterns = patterns[language]
    code_lines = code.splitlines()
    current_line = 1

    # Iterate through the lines of code and check for patterns
    for line in code_lines:
        line = line.strip()
        # Check if the line contains an insecure development setting
        for pattern in relevant_patterns:
            if re.search(pattern, line, re.MULTILINE):
                results.append((current_line, line))
                break  # Stop checking once we find a matching pattern
        current_line += 1

    if not results:
        print(f"R9 (Dev Env Misconfigurations): No dev env misconfigurations found")
        print('R9 (Dev Env Misconfigurations): Finished')
        return False
    
    print(f"R9 (Dev Env Misconfigurations): Dev env misconfigurations found")
    print('R9 (Dev Env Misconfigurations): Finished')
    return results

def r10_check_imported_plaintext(code, language):
    print(f"R10 (Sensitive Data Stored Without Encryption): Started")
    
    patterns = {
        "python": [
            r"open\s*\(.*['\"]w['\"].*\)\s*\.write\s*\(.*(?:password|secret).*",
            r"with\s+open\s*\(.*['\"]w['\"].*\)\s+as\s+.*:\s*.*(?:password|secret).*"
        ],
        "javascript": [
            r"fs\.writeFileSync\s*\(.*['\"](?:password|secret).*",
            r"fs\.writeFile\s*\(.*['\"](?:password|secret).*"
        ],
        "typescript": [
            r"fs\.writeFileSync\s*\(.*['\"](?:password|secret).*",
            r"fs\.writeFile\s*\(.*['\"](?:password|secret).*"
        ],
        "java": [
            r"Files\.write\s*\(.*(?:password|secret).*\.getBytes\(\)",
            r"new\s+FileOutputStream\s*\(.*['\"].*(?:password|secret).*"
        ],
        "c#": [
            r"File\.WriteAllText\s*\(.*['\"].*(?:password|secret).*",
            r"File\.WriteAllBytes\s*\(.*(?:password|secret).*"
        ],
        "c++": [
            r"std::ofstream\s+\w+\s*\(.*(?:password|secret).*",
            r"std::ofstream\s+\w+\s*;\s*.*\s*<<\s*.*(?:password|secret).*"
        ],
        "powershell": [
            r"Set-Content\s*\-Path\s*.*(?:password|secret).*",
            r"Out-File\s*-FilePath\s*.*(?:password|secret).*"
        ]
    }
    
    results = []
    language = language.lower()

    if language not in patterns:
        print(f"R10 (Sensitive Data Stored Without Encryption): Language '{language}' not supported")
        print(f"R10 (Sensitive Data Stored Without Encryption): Finished")
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
        print(f"R10 (Sensitive Data Stored Without Encryption): No insecure data storage found")
        print(f"R10 (Sensitive Data Stored Without Encryption): Finished")
        return False

    print(f"R10 (Sensitive Data Stored Without Encryption): Found instances of sensitive data stored without encryption")
    print(f"R10 (Sensitive Data Stored Without Encryption): Finished")
    return results

def r11_weak_hashing(code, language):
    print('R11 (Weak Hashing Algorithms): Started')    
    patterns = {
        "python": [
            r"hashlib\.md5\s*\(.*\)\s*\.hexdigest\(\)",
            r"hashlib\.sha1\s*\(.*\)\s*\.hexdigest\(\)"
        ],
        "javascript": [
            r"crypto\.createHash\(['\"]md5['\"]\)\.update\(.+\)\.digest\(['\"]hex['\"]\)",
            r"crypto\.createHash\(['\"]sha1['\"]\)\.update\(.+\)\.digest\(['\"]hex['\"]\)"
        ],
        "typescript": [
            r"crypto\.createHash\(['\"]md5['\"]\)\.update\(.+\)\.digest\(['\"]hex['\"]\)",
            r"crypto\.createHash\(['\"]sha1['\"]\)\.update\(.+\)\.digest\(['\"]hex['\"]\)"
        ],
        "java": [
            r"MessageDigest\s*\.\s*getInstance\s*\(\s*['\"]MD5['\"]\s*\)",  # MD5 usage
            r"MessageDigest\s*\.\s*getInstance\s*\(\s*['\"]SHA-?1['\"]\s*\)",  # SHA1 usage
            r"\.digest\s*\(.*?\)",  # .digest function
            r"\.update\s*\(.*?\)"  # .update function
        ],
        "c#": [
            r"(?i)(MD5|SHA1)\.Create\(\)",  # Detects `MD5.Create()` or `SHA1.Create()`
            r"(?i)HashAlgorithm\.(Create|Create\(.*?(MD5|SHA1).*\))",  # General HashAlgorithm usage
            r"(?i)(MD5|SHA1)\.Create\(\)\.ComputeHash\(",  # `ComputeHash()` method for MD5 or SHA1
            r"(?i)(MD5|SHA1)\.Create\(\)\.(TransformBlock|TransformFinalBlock)\(",  # Other transformation methods
            r"(?i)(MD5|SHA1)\.Create\(\)\.Initialize\(",  # `Initialize` method for MD5 or SHA1
            r"(?i)new\s+(MD5|SHA1)CryptoServiceProvider\(\)",  # Direct usage of `CryptoServiceProvider`
            r"(?i)new\s+HMAC(MD5|SHA1)\(\)"  # HMAC variants
        ],
        "c++": [
            r"MD5\s*\(\s*.*\s*\)",  # Match MD5 function calls
            r"SHA1\s*\(\s*.*\s*\)"   # Match SHA1 function calls
        ],
        "powershell": [
            r"\[System\.Security\.Cryptography\.MD5\]::Create\(\)\.ComputeHash\(\[System\.Text\.Encoding\]::UTF8\.GetBytes\(.+\)\)",  # MD5
            r"\[System\.Security\.Cryptography\.SHA1\]::Create\(\)\.ComputeHash\(\[System\.Text\.Encoding\]::UTF8\.GetBytes\(.+\)\)"  # SHA1
        ]
    }
    
    results = []
    language = language.lower()
    
    if language not in patterns:
        print(f"R11 (Weak Hashing Algorithms): Language '{language}' not supported")
        print(f"R11 (Weak Hashing Algorithms): Finished")
        return False

    relevant_patterns = patterns[language]
    code_lines = code.splitlines()
    current_line = 1

    for line in code_lines:
        line = line.strip()
        for pattern in relevant_patterns:
            if re.search(pattern, line):
                results.append((current_line, line))
                break
        current_line += 1
    
    if not results:
        print(f"R11 (Weak Hashing Algorithms): No weak hashing algorithms found")
        print('R11 (Weak Hashing Algorithms): Finished')
        return False
    
    print(f"R11 (Weak Hashing Algorithms): Weak hashing algorithms found")
    print('R11 (Weak Hashing Algorithms): Finished')
    return results


def r19_check_no_csrf_token(code, language):
    print('R19 (Missing CSRF Token): Started')    

    patterns = {
        "python": [
            r"@app\.route\(.+\)\s*def\s+\w+\(.+\):",  # Matches Flask route definitions
            r"@csrf_exempt",  # Exemptions from CSRF
            r"@app\.view_functions\[\w+\]",  # Matches Flask view functions
            r"request\.post",  # Matches direct POST request handling
            r"request\.form",  # Matches access to form data
            r"request\.get_json",  # Matches JSON requests
            r"request\.data",  # Matches raw request data
            r"request\.files",  # Matches file uploads
            r"def\s+\w+\(\s*.*\s*\):",  # Matches function definitions (general)
        ],
        "javascript": [
            r"app\.(post|put|delete)\s*\(\s*['\"]\S+['\"]\s*,",  # Matches Express.js request handlers
            r"app\.use\(\s*csrfProtection\s*,",  # Middleware usage
        ],
        "typescript": [
            r"app\.(post|put|delete)\s*\(\s*['\"]\S+['\"]\s*,",  # Matches Express.js request handlers
            r"app\.use\(\s*csrfProtection\s*,",  # Middleware usage
        ],
        "java": [
            r"@RequestMapping\(.+RequestMethod.POST.+\)",  # Matches Spring POST request mappings
            r"@PostMapping\(.+\)",  # Matches PostMapping annotations
        ],
        "c#": [
            r"\[HttpPost\]",  # Matches ASP.NET HttpPost attribute
            r"\[IgnoreAntiforgeryToken\]",  # Exemptions from CSRF
            r"Post\(\w+\)",  # General post handling method
        ],
        "c++": [
            r"if\s*\(.+POST.+\)\s*{",  # Hypothetical request handling in C++
            r"if\s*\(method\s*==\s*\"POST\"\)",  # General check for POST method
        ],
        "powershell": [
            # POST Request Patterns (with various styles)
            r"Invoke-RestMethod\s+.+-Method\s+Post",  # Detects PowerShell Invoke-RestMethod POST requests
            r"Invoke-WebRequest\s+.+-Method\s+Post",  # Detects PowerShell Invoke-WebRequest POST requests
            r"Invoke-RestMethod\s+-Uri\s+.+-Method\s+Post",  # POST requests with URI
            r"Invoke-WebRequest\s+-Uri\s+.+-Method\s+Post",  # POST requests with URI
            r"\[System.Net.Http.HttpClient\]\::PostAsync",  # Using HttpClient's PostAsync
            r"New-Object\s+System.Net.Http.HttpClient",  # Direct instantiation of HttpClient
            r"Start-Job\s+\{.+Invoke-RestMethod.+Post.+\}",  # Background jobs with POST

            # CSRF Token Patterns (Assignment/Check)
            r"\$csrfToken\s*=\s*.*",  # Assignment of CSRF token variable
            r"if\s*\(\s*\$csrfToken\s*-ne\s*\"\"\s*\)",  # Check if CSRF token is non-empty
            r"-Headers\s+\@{.*\$csrfToken.*}",  # Sending CSRF token in headers
            r"Add-Member\s+-MemberType\s+NoteProperty\s+-Name\s+'csrfToken'\s+-Value",  # Add a CSRF token as a property
        ]
    }

    csrf_patterns = {
        "python": [
            r"csrf_token",  # Check for csrf_token in form handlers
            r"request\.csrf_token",  # Check for CSRF token in requests
            r"flask_wtf\.csrf\.CSRFProtect",  # Flask-WTF CSRF protection
        ],
        "javascript": [
            r"csrfProtection",  # Middleware used in Express
            r"csrf_token",  # General CSRF token check
        ],
        "typescript": [
            r"csrfProtection",  # Middleware used in Express
            r"csrf_token",  # General CSRF token check
        ],
        "java": [
            r"CsrfToken",  # Check for CSRF token usage
        ],
        "c#": [
            r"ValidateAntiForgeryToken",  # Attribute for CSRF validation in ASP.NET
        ],
        "c++": [],  # No specific pattern for C++ in this example
        "powershell": [
            r"csrfToken",  # General pattern for CSRF token variable
            r"\$csrfToken\s*=\s*.*",  # Assignment of CSRF token
            r"if\s*\(\$.*\s*-ne\s*\"\"\)",  # Check if token is non-empty
            r"-Headers\s+\@{.*csrfToken.*}",  # CSRF token sent in headers
        ]
    }

    results = []
    language = language.lower()

    if language not in patterns:
        print(f"R19 (Missing CSRF Token): Language '{language}' not supported")
        print(f"R19 (Missing CSRF Token): Finished")
        return False

    relevant_patterns = patterns[language]
    csrf_token_patterns = csrf_patterns[language]
    
    code_lines = code.splitlines()
    current_line = 1

    for line in code_lines:
        line = line.strip()
        for pattern in relevant_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Check if CSRF token is missing in the following lines
                has_csrf = any(re.search(csrf_pattern, line, re.IGNORECASE) for csrf_pattern in csrf_token_patterns)
                if not has_csrf:
                    results.append((current_line, line))
                break
        current_line += 1
    
    if not results:
        print(f"R19 (Missing CSRF Token): No issues found")
        print('R19 (Missing CSRF Token): Finished')
        return False

    print(f"R19 (Missing CSRF Token): Found request handlers without CSRF token")
    print('R19 (Missing CSRF Token): Finished')
    return results
    
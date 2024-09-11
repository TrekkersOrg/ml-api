import ast

def check_sql_injection(code_ast, language):
    failed_lines = []
    if language == 'python':
        def is_sql_injection_pattern(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
                if isinstance(node.left, ast.Str) and ('%' in node.left.s or '{' in node.left.s):
                    return True
                if isinstance(node.right, ast.Str) and ('%' in node.right.s or '{' in node.right.s):
                    return True 
            elif isinstance(node, ast.JoinedStr):
                return True
            return False
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Call) and hasattr(node.func, 'attr') and node.func.attr in ('execute', 'executemany'):
                for arg in node.args:
                    if is_sql_injection_pattern(arg):
                        failed_lines.append(node.lineno)
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, (ast.Mod, ast.Add)):
                    if isinstance(node.value.left, ast.Str) and ('%' in node.value.left.s or '{' in node.value.left.s):
                        failed_lines.append(node.lineno)
                    if isinstance(node.value.right, ast.Str) and ('%' in node.value.right.s or '{' in node.value.right.s):
                        failed_lines.append(node.lineno)
    return failed_lines

def check_xss(code_ast, language):
    failed_lines = []
    if language == 'python':
        reported_lines = set()
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'render_template':
                for arg in node.args:
                    if isinstance(arg, ast.Str) and ("<" in arg.s or "{{" in arg.s):
                        if node.lineno not in reported_lines:
                            failed_lines.append(node.lineno)
                            reported_lines.add(node.lineno)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                if (isinstance(node.left, ast.Str) and "<" in node.left.s) or (isinstance(node.right, ast.Str) and "<" in node.right.s):
                    if node.lineno not in reported_lines:
                        failed_lines.append(node.lineno)
                        reported_lines.add(node.lineno)
    return failed_lines

def check_insecure_deserialization(code_ast, language):
    failed_lines = []
    if language == 'python':
        deserialization_functions = ['pickle.load', 'pickle.loads', 'marshal.load', 'marshal.loads', 'yaml.load', 'yaml.unsafe_load', 'json.load']
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        full_func_name = f"{module_name}.{func_name}"
                        if full_func_name in deserialization_functions:
                            failed_lines.append(node.lineno)
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in deserialization_functions:
                        failed_lines.append(node.lineno)
    return failed_lines

def check_security_misconfiguration(code_ast, language):
    failed_lines = []
    if language == 'python':
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'DEBUG':
                    if isinstance(node.value, ast.Constant) and node.value.value == True:
                        failed_lines.append(node.lineno)
    return failed_lines

def check_sensitive_data_exposure(code_ast, language):
    failed_lines = []
    if language == 'python':
        sensitive_keywords = ['password', 'secret', 'apikey', 'token', 'credential', 'private_key', 'ssl_key', 'api_key']
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name):
                    if any(kw in node.targets[0].id.lower() for kw in sensitive_keywords):
                        if isinstance(node.value, ast.Str):
                            failed_lines.append(node.lineno)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(kw in node.value.lower() for kw in sensitive_keywords):
                    failed_lines.append(node.lineno)
    return failed_lines

def check_broken_authentication(code_ast, language):
    failed_lines = []
    if language == 'python':
        weak_algorithms = ['md5', 'sha1', 'des', 'rc4']
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.attr, str) and node.func.attr.lower() in weak_algorithms:
                        failed_lines.append(node.lineno)
                elif isinstance(node.func, ast.Name):
                    if node.func.id.lower() in weak_algorithms:
                        failed_lines.append(node.lineno)
    return failed_lines

def check_broken_access_control(code_ast, language):
    failed_lines = []
    if language == 'python':
        for node in ast.walk(code_ast):
            if isinstance(node, ast.If):
                if not any(isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id in ['is_authenticated', 'has_permission'] for test in ast.walk(node.test)):
                    failed_lines.append(node.lineno)
    return failed_lines

def check_csrf(code_ast, language):
    failed_lines = []
    if language == 'python':
        form_handlers = {'form', 'post', 'request', 'submit'}
        for node in ast.walk(code_ast):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                is_form_handler = any(
                    isinstance(decorator, ast.Call) and
                    isinstance(decorator.func, ast.Name) and
                    decorator.func.id in form_handlers
                    for decorator in node.decorator_list
                )
                if not is_form_handler:
                    csrf_token_check_found = False
                    for body_node in node.body:
                        if isinstance(body_node, ast.If):
                            for test in ast.walk(body_node.test):
                                if isinstance(test, ast.Compare):
                                    if any(
                                        isinstance(test.left, ast.Name) and
                                        test.left.id == 'csrf_token'
                                        for test in ast.walk(body_node.test)
                                    ):
                                        csrf_token_check_found = True
                                        break
                            if csrf_token_check_found:
                                break
                    if not csrf_token_check_found:
                        failed_lines.append(node.lineno)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in form_handlers:
                    csrf_token_present = any(
                        isinstance(arg, ast.Str) and 'csrf_token' in arg.s
                        for arg in node.args
                    )
                    if not csrf_token_present:
                        failed_lines.append(node.lineno)
    return failed_lines

def check_known_vulnerabilities(code_ast, language):
    failed_lines = []
    if language == 'python':
        vulnerable_components = {
            'urllib3': '0.9',
            'requests': '2.0',
            'django': '1.8'
        }
        vulnerable_modules = set(vulnerable_components.keys())
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in vulnerable_modules:
                        failed_lines.append(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                if node.module in vulnerable_modules:
                    failed_lines.append(node.lineno)
    return failed_lines

def check_logging_monitoring(code_ast, language):
    failed_lines = []
    if language == 'python':
        for node in ast.walk(code_ast):
            if isinstance(node, ast.Try):
                if not any(
                    isinstance(handler, ast.ExceptHandler) and
                    any(
                        isinstance(sub_node, ast.Call) and
                        isinstance(sub_node.func, ast.Name) and
                        sub_node.func.id in ['logging', 'log']
                        for sub_node in ast.walk(stmt)
                    )
                    for handler in node.handlers for stmt in handler.body
                ):
                    failed_lines.append(node.lineno)
    return failed_lines
import re

from nltk import RangeFeature

def clean_ruff_output(ruff_output):
    message = ruff_output.get("message", b"").decode('utf-8') if isinstance(ruff_output.get("message", b""), bytes) else ruff_output.get("message", "")
    errors = []
    error_lines = message.splitlines()
    
    # Adjusted regex pattern to capture the relevant error information
    error_pattern = re.compile(
        r'^(?P<file>.*?):(?P<line>\d+):(?P<column>\d+): (?P<type>\w+)\s+\[\*\]\s*(?P<message>.+?)$',
        re.MULTILINE
    )
    current_error = None
    
    for line in error_lines:
        match = error_pattern.match(line)
        if match:
            if current_error:
                errors.append(current_error)  # Save the previous error before starting a new one
            line_number = int(match.group('line').strip())
            error_type = match.group('type').strip()
            error_message = match.group('message').strip()
            current_error = {
                "file": match.group('file'),
                "line": line_number,
                "column": int(match.group('column')),
                "type": error_type,
                "message": error_message
            }
        elif current_error and re.match(r'^\s*\|\s*', line):
            context_line = line.strip().lstrip('|').strip()
            if context_line:
                current_error["message"] += "\n" + context_line  # Append context lines to the current error message
        elif "Found" in line:  # Check for summary lines that are not errors
            continue  # Skip summary lines
        
    if current_error:
        errors.append(current_error)  # Append the last error if exists
    
    for error in errors:
        error["message"] = error["message"].replace('\n^', '').strip()  # Clean up the error message
    
    if errors:
        return {
            "status": "fail",
            "errors": errors
        }
    else:
        return {
            "status": "success",
            "message": "No linting errors found."
        }

def clean_pylint_output(pylint_output):
    message = pylint_output.get("message", "")
    errors = []
    error_lines = message.splitlines()
    error_pattern = re.compile(
        r'^(?P<file>.*?):(?P<line>\d+):(?P<type>\w+): (?P<message>.+)$',
        re.MULTILINE
    )
    for line in error_lines:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        match = error_pattern.match(line)
        if match:
            file_path = match.group('file')
            line_number = int(match.group('line'))
            error_type = match.group('type').strip()
            error_message = match.group('message').strip()
            errors.append({
                "line": line_number,
                "type": error_type,
                "message": error_message
            })
    if len(errors) > 0:
        return {
            "status": "fail",
            "errors": errors
        }
    else:
        return {
            "status": "success",
            "message": "No linting errors found."
        }
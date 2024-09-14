import subprocess
import os
import re

LINTER_FOLDER = 'linter'
os.makedirs(LINTER_FOLDER, exist_ok=True)

def run_ruff(file_path):
    try:
        result = subprocess.run(["py", "-m", "ruff", "check", file_path], capture_output=True)
        if result.returncode == 0:
            return {"status": "success", "message": result.stdout}
        return {"status": "fail", "message": result.stdout}
    except FileNotFoundError:
        return {"status": "error", "message": "Ruff is not installed. Please install it using 'pip install ruff'."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return

def clean_ruff_output(ruff_output):
    # Extract the message part of the ruff_output
    message = ruff_output.get("message", b"").decode('utf-8') if isinstance(ruff_output.get("message", b""), bytes) else ruff_output.get("message", "")
    
    # Initialize a list to hold the cleaned errors
    errors = []

    # Split the message into individual lines
    error_lines = message.splitlines()

    # Define a more detailed regex pattern for error extraction
    error_pattern = re.compile(
        r'^(?P<file>.*?):(?P<line>\d+):(?P<column>\d+):\s*(?P<code>\w+)\s*(?P<message>.+?)\s*(?:\n\s*\|\s*\d+\s*\|.*)?(?:\n\s*\|\s*\d+\s*\|.*)*$'
    )

    current_error = None

    for line in error_lines:
        # Match the error line
        match = error_pattern.match(line)
        if match:
            # Save the current error if it exists
            if current_error:
                errors.append(current_error)
            
            # Start a new error entry
            file_path = match.group('file').strip()
            line_number = int(match.group('line').strip())
            column_number = int(match.group('column').strip())
            error_code = match.group('code').strip()
            error_message = match.group('message').strip()
            
            current_error = {
                "line": line_number,
                "message": error_message
            }
        elif current_error and re.match(r'^\s*\|\s*', line):
            # Append multi-line context to the current error's message
            context_line = line.strip().lstrip('|').strip()
            if context_line:
                current_error["message"] += "\n" + context_line
    
    # Don't forget to add the last error if it exists
    if current_error:
        errors.append(current_error)
    
    # Return a fail status with errors if any were found, else success
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



def python_linter_check(file):
    file_path = os.path.join(LINTER_FOLDER, file.filename)
    file.save(file_path)
    linter_result = {}
    ruff_result = run_ruff(file_path)

    os.remove(file_path)
    print(ruff_result)
    print('BREAKBREAKBREAJREAJAERNIEARBO+++++++++++++++++++++++++++++++++++++++++++')
    cleaned_result = clean_ruff_output(ruff_result)
    linter_result['ruff'] = cleaned_result
    return linter_result

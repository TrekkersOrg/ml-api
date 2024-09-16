import subprocess
import os
import re
from linter_helpers import *

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

def python_linter_check(file):
    file_path = os.path.join(LINTER_FOLDER, file.filename)
    file.save(file_path)
    linter_result = {}
    ruff_result = run_ruff(file_path)
    os.remove(file_path)
    print(ruff_result)
    cleaned_result = clean_ruff_output(ruff_result)
    linter_result['ruff'] = cleaned_result
    return linter_result

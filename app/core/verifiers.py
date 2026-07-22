import ast
import json
import re
from typing import Optional

class ExecutionVerifier:
    def verify(self, code: str) -> bool:
        if not code or not code.strip(): return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def extract_code(self, response: str) -> str:
        match = re.search(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
        if match: return match.group(1).strip()
        match = re.search(r'```(.*?)```', response, re.DOTALL)
        if match: return match.group(1).strip()
        return response.strip()

class SchemaVerifier:
    def verify(self, json_str: str, required_keys: Optional[list] = None) -> bool:
        if not json_str or not json_str.strip(): return False
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            match = re.search(r'```(?:json)?\n?(.*?)```', json_str, re.DOTALL)
            if match:
                try: parsed = json.loads(match.group(1).strip())
                except: return False
            else: return False
        if required_keys:
            return isinstance(parsed, dict) and all(k in parsed for k in required_keys)
        return True

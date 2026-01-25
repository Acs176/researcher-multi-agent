from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def require_json(text: str) -> Dict[str, Any]:
    data = extract_json(text)
    if data is None:
        raise ValueError("Expected valid JSON block in agent response.")
    return data


def to_json_text(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)

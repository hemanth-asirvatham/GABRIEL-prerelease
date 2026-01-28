from __future__ import annotations

import base64
import os
from typing import Dict, Optional


def encode_pdf(pdf_path: str) -> Optional[Dict[str, str]]:
    """Return PDF contents as a dict suitable for OpenAI input_file."""

    try:
        with open(pdf_path, "rb") as pdf_file:
            payload = base64.b64encode(pdf_file.read()).decode("utf-8")
        filename = os.path.basename(pdf_path) or "document.pdf"
        return {
            "filename": filename,
            "file_data": f"data:application/pdf;base64,{payload}",
        }
    except Exception:
        return None

"""
kpe/ai/gemini_client.py

Lightweight integration layer for Google Gemini.
Designed to fail gracefully if the environment or dependencies are missing,
ensuring zero disruption to existing pipelines.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Attempt to import the generative AI library
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logger.warning("google-generativeai package is not installed. Gemini integration disabled.")


class GeminiClient:
    """
    Client for interacting with Google Gemini models.
    """

    def __init__(self):
        self.is_active = False
        self.model = None

        if not HAS_GENAI:
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.info("GEMINI_API_KEY not found in environment. Gemini integration disabled.")
            return

        try:
            genai.configure(api_key=api_key)
            # Using the free tier model (fast and cost-effective)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.is_active = True
            logger.info("GeminiClient successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiClient: {e}")

    def analyze_signature(self, signature_json: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Send the signature data to Gemini to get a human-readable interpretation.
        Returns None if the client is not active or if the API call fails.
        """
        if not self.is_active or not self.model:
            return None

        prompt = (
            "You are an expert computer vision analyst. "
            "I will provide you with a JSON payload representing a motion signature extracted from a video clip "
            "using sparse optical flow. The data contains 10-dimensional vectors per frame.\n\n"
            "Please provide a very brief (1-2 sentences) semantic description of the motion and flag any "
            "obvious anomalies (like sudden drastic changes in motion vectors that might indicate a scene cut or tampering).\n\n"
            f"Metadata: {metadata}\n"
            f"Signature Payload: {signature_json}\n"
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return None

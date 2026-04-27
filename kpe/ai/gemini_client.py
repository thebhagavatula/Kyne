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
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logger.warning("google-genai package is not installed. Gemini integration disabled.")


class GeminiClient:
    """
    Client for interacting with Google Gemini models.
    """

    def __init__(self):
        self.is_active = False
        self.client = None
        self.model_name = "gemini-2.5-flash"

        if not HAS_GENAI:
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.info("GEMINI_API_KEY not found in environment. Gemini integration disabled.")
            return

        try:
            self.client = genai.Client(api_key=api_key)
            self.is_active = True
            logger.info("GeminiClient successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiClient: {e}")

    def analyze_signature(self, signature_json: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Send the signature data to Gemini to get a human-readable interpretation.
        Returns None if the client is not active or if the API call fails.
        """
        if not self.is_active or not self.client:
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return None

    def generate_forensic_report(self, query_sig: list, ref_sig: Optional[list], match_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Generate a comprehensive Markdown forensic report explaining the verification result.
        """
        if not self.is_active or not self.client:
            return None

        prompt = (
            "You are an expert digital forensics analyst specializing in kinetic video verification. "
            "I will provide you with the motion energy (global motion magnitude) extracted from a suspect clip, "
            "and optionally the reference broadcast it was matched against. I will also provide the DTW matching metadata.\n\n"
            "Your job is to generate a concise Markdown report (2-3 paragraphs) that:\n"
            "1. Explains the verification verdict and confidence score.\n"
            "2. Analyzes the motion signals to point out any anomalies (like scene cuts, drops, or recording artifacts).\n"
            "3. Concludes on the authenticity of the suspect clip.\n\n"
            f"Matching Metadata: {match_metadata}\n"
            f"Suspect Motion Energy: {query_sig}\n"
        )
        
        if ref_sig:
            prompt += f"Reference Motion Energy: {ref_sig}\n"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return None

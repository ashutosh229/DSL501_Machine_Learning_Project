from logger import logger
from datetime import datetime
import spacy
from typing import List, Dict, Any, Tuple, Optional
from models.states import SelfState, ProcessedPost, ClassificationResult, StateType
import re


class DataProcessingAgent:
    """Agent 1: Handles data preprocessing and preparation"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        logger.info("Data Processing Agent initialized")

    def process_post(self, post_data: Dict[str, Any]) -> ProcessedPost:
        """Process raw Reddit post data"""
        start_time = datetime.now()

        # Extract text and metadata
        text = post_data.get("text", "")
        user_id = post_data.get("user_id", "unknown")
        post_id = post_data.get("post_id", "unknown")
        timeline_pos = post_data.get("timeline_position", 0)

        # Privacy protection - basic anonymization
        text = self._anonymize_text(text)

        # Sentence segmentation
        doc = self.nlp(text)
        sentences = [
            sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10
        ]

        # Create context windows (2-3 sentence chunks)
        context_windows = self._create_context_windows(sentences)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processed post {post_id} in {processing_time:.2f}s")

        return ProcessedPost(
            original_text=text,
            sentences=sentences,
            context_windows=context_windows,
            user_id=user_id,
            post_id=post_id,
            timeline_position=timeline_pos,
        )

    def _anonymize_text(self, text: str) -> str:
        """Basic privacy protection"""
        # Remove usernames, emails, phone numbers
        text = re.sub(r"@\w+", "[USER]", text)
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
        )
        text = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", "[PHONE]", text)
        return text

    def _create_context_windows(
        self, sentences: List[str], window_size: int = 3
    ) -> List[str]:
        """Create overlapping context windows"""
        windows = []
        for i in range(len(sentences)):
            start = max(0, i - window_size // 2)
            end = min(len(sentences), start + window_size)
            window = " ".join(sentences[start:end])
            windows.append(window)
        return windows

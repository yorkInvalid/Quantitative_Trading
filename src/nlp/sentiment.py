"""
FinBERT-based sentiment analysis module for Chinese financial text.

Uses yiyanghkust/finbert-tone-chinese model to compute sentiment scores.
Score = Prob(Positive) - Prob(Negative)
"""

import os
from typing import List, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Default model and cache settings
DEFAULT_MODEL_NAME = "yiyanghkust/finbert-tone-chinese"
DEFAULT_CACHE_DIR = "/app/data/models"

# Label mapping for yiyanghkust/finbert-tone-chinese
# Actual: 0=Neutral, 1=Positive, 2=Negative
LABEL_NEUTRAL = 0
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 2


class SentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT for Chinese financial text.

    Computes sentiment score as: Prob(Positive) - Prob(Negative)
    Range: [-1, 1] where positive values indicate bullish sentiment.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[str] = None,
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Directory to cache downloaded model files.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
        """
        self.model_name = model_name
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"SentimentAnalyzer initialized on {self.device}")

    def predict(
        self,
        text_list: Union[str, List[str]],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> List[float]:
        """
        Compute sentiment scores for a list of texts.

        Args:
            text_list: Single text or list of texts to analyze.
            batch_size: Batch size for inference.
            max_length: Maximum token length for truncation.

        Returns:
            List of sentiment scores in range [-1, 1].
            Score = Prob(Positive) - Prob(Negative)
        """
        # Handle single text input
        if isinstance(text_list, str):
            text_list = [text_list]

        if not text_list:
            return []

        scores = []

        # Process in batches
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Calculate score: P(positive) - P(negative)
            batch_scores = (
                (probs[:, LABEL_POSITIVE] - probs[:, LABEL_NEGATIVE]).cpu().tolist()
            )

            scores.extend(batch_scores)

        return scores

    def predict_single(self, text: str) -> float:
        """
        Compute sentiment score for a single text.

        Args:
            text: Text to analyze.

        Returns:
            Sentiment score in range [-1, 1].
        """
        return self.predict([text])[0]


if __name__ == "__main__":
    # Quick test
    analyzer = SentimentAnalyzer()

    test_texts = [
        "公司业绩大幅增长，净利润翻倍",
        "公司涉嫌财务造假，被立案调查",
        "今日公司召开例行会议",
    ]

    scores = analyzer.predict(test_texts)
    for text, score in zip(test_texts, scores):
        print(f"Text: {text}")
        print(f"Score: {score:.4f}")
        print("-" * 40)

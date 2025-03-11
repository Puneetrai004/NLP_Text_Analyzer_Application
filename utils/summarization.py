# utils/summarization.py
from nltk.tokenize import sent_tokenize
import numpy as np

def summarize_text(text, ratio=0.3):
    """
    Summarize text using a simple extractive approach
    
    Args:
        text (str): Input text
        ratio (float): Ratio of the original text to keep in the summary
        
    Returns:
        str: Summarized text
    """
    if not text or len(text) < 100:
        return text
    
    try:
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 3:
            return text
        
        # Calculate number of sentences for summary
        n_sentences = max(1, int(len(sentences) * ratio))
        
        # Simple approach: take first n sentences
        # For a more sophisticated approach, you could implement sentence scoring
        summary = ' '.join(sentences[:n_sentences])
        
        return summary
    except Exception as e:
        print(f"Error in summarization: {e}")
        return text[:int(len(text) * ratio)]

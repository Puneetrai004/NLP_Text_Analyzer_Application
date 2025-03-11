from gensim.summarization import summarize as gensim_summarize
from nltk.tokenize import sent_tokenize
import traceback

def summarize_text(text, ratio=0.3):
    """
    Summarize text using gensim's TextRank algorithm
    
    Args:
        text (str): Input text
        ratio (float): Ratio of the original text to keep in the summary
        
    Returns:
        str: Summarized text
    """
    if not text or len(sent_tokenize(text)) < 3:
        return ""
    
    try:
        # Use gensim's summarize function
        return gensim_summarize(text, ratio=ratio)
    except Exception as e:
        # Fallback to a simple extractive summarization
        try:
            sentences = sent_tokenize(text)
            n_sentences = max(1, int(len(sentences) * ratio))
            return ' '.join(sentences[:n_sentences])
        except:
            return ""

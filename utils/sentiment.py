from textblob import TextBlob
from nltk.tokenize import sent_tokenize

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary containing polarity and subjectivity scores
    """
    # Create TextBlob object
    blob = TextBlob(text)
    
    # Get overall sentiment
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Analyze sentiment of individual sentences
    sentences = sent_tokenize(text)
    sentence_analysis = []
    
    for sentence in sentences:
        sentence_blob = TextBlob(sentence)
        sentence_analysis.append({
            'text': sentence,
            'polarity': sentence_blob.sentiment.polarity,
            'subjectivity': sentence_blob.sentiment.subjectivity
        })
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentence_analysis': sentence_analysis
    }

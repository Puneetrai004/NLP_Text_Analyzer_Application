def extract_entities(text, nlp):
    """
    Extract named entities from text using spaCy
    
    Args:
        text (str): Input text
        nlp: spaCy language model
        
    Returns:
        list: List of dictionaries containing entity text, label, and start/end positions
    """
    if not text:
        return []
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract entities
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    
    return entities

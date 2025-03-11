# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# import spacy
# from textblob import TextBlob
# from wordcloud import WordCloud
# import time

# # Set page configuration as the first Streamlit command
# st.set_page_config(
#     page_title="NLP Text Analysis App",
#     page_icon="📊",
#     layout="wide"
# )

# # Download necessary NLTK resources
# @st.cache_resource
# def download_nltk_resources():
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     nltk.download('wordnet')

# # Load spaCy model
# @st.cache_resource
# def load_spacy_model():
#     return spacy.load('en_core_web_sm')

# # Text preprocessing function
# def preprocess_text(text, remove_stopwords=True, remove_punctuation=True, lemmatize=True):
#     if not text:
#         return ""
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Tokenize
#     tokens = word_tokenize(text)
    
#     # Remove stopwords
#     if remove_stopwords:
#         stop_words = set(stopwords.words('english'))
#         tokens = [token for token in tokens if token not in stop_words]
    
#     # Remove punctuation
#     if remove_punctuation:
#         tokens = [token for token in tokens if token.isalnum()]
    
#     # Lemmatize
#     if lemmatize:
#         lemmatizer = nltk.stem.WordNetLemmatizer()
#         tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
#     # Join tokens back into text
#     return ' '.join(tokens)

# # Sentiment analysis function
# def analyze_sentiment(text):
#     blob = TextBlob(text)
#     return {
#         'polarity': blob.sentiment.polarity,
#         'subjectivity': blob.sentiment.subjectivity,
#     }

# # Named entity recognition function
# def extract_entities(text, nlp):
#     if not text:
#         return []
    
#     doc = nlp(text)
#     entities = []
#     for ent in doc.ents:
#         entities.append({
#             'text': ent.text,
#             'label': ent.label_
#         })
#     return entities

# # Text summarization function (alternative to Gensim)
# def summarize_text(text, ratio=0.3):
#     if not text or len(text) < 100:
#         return text
    
#     try:
#         # Split into sentences
#         sentences = sent_tokenize(text)
        
#         if len(sentences) <= 3:
#             return text
        
#         # Calculate number of sentences for summary
#         n_sentences = max(1, int(len(sentences) * ratio))
        
#         # Simple approach: take first n sentences
#         summary = ' '.join(sentences[:n_sentences])
        
#         return summary
#     except Exception as e:
#         print(f"Error in summarization: {e}")
#         return text[:int(len(text) * ratio)]

# def main():
#     # Initialize resources
#     download_nltk_resources()
#     nlp = load_spacy_model()
    
#     # Application title and description
#     st.title("NLP Text Analysis Application")
#     st.markdown("""
#     This application provides various Natural Language Processing tools to analyze your text.
#     Enter your text and select the analysis you want to perform.
#     """)
    
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     analysis_option = st.sidebar.selectbox(
#         "Choose Analysis Type",
#         ["Text Preprocessing", "Sentiment Analysis", "Named Entity Recognition", "Text Summarization", "Word Cloud"]
#     )
    
#     # Text input area
#     text_input = st.text_area("Enter your text here:", height=200)
    
#     # Process based on selected option
#     if text_input:
#         if analysis_option == "Text Preprocessing":
#             show_text_preprocessing(text_input)
        
#         elif analysis_option == "Sentiment Analysis":
#             show_sentiment_analysis(text_input)
        
#         elif analysis_option == "Named Entity Recognition":
#             show_ner(text_input, nlp)
        
#         elif analysis_option == "Text Summarization":
#             show_text_summarization(text_input)
        
#         elif analysis_option == "Word Cloud":
#             show_word_cloud(text_input)
#     else:
#         st.info("Please enter some text to analyze.")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("Developed with ❤️ using Streamlit and Python NLP libraries")

# def show_text_preprocessing(text):
#     st.header("Text Preprocessing")
    
#     with st.spinner("Processing text..."):
#         # Get preprocessing options
#         remove_stopwords = st.checkbox("Remove Stopwords", value=True)
#         remove_punctuation = st.checkbox("Remove Punctuation", value=True)
#         lemmatize = st.checkbox("Lemmatize", value=True)
        
#         # Process text
#         processed_text = preprocess_text(
#             text, 
#             remove_stopwords=remove_stopwords,
#             remove_punctuation=remove_punctuation,
#             lemmatize=lemmatize
#         )
        
#         # Display results
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Original Text")
#             st.write(text)
            
#             st.subheader("Token Count")
#             tokens = word_tokenize(text)
#             st.write(f"Number of tokens: {len(tokens)}")
            
#             st.subheader("Sentence Count")
#             sentences = sent_tokenize(text)
#             st.write(f"Number of sentences: {len(sentences)}")
        
#         with col2:
#             st.subheader("Processed Text")
#             st.write(processed_text)
            
#             st.subheader("Processed Token Count")
#             processed_tokens = word_tokenize(processed_text)
#             st.write(f"Number of tokens after processing: {len(processed_tokens)}")
            
#             # Show sample tokens
#             if processed_tokens:
#                 st.subheader("Sample Tokens")
#                 sample_size = min(10, len(processed_tokens))
#                 st.write(processed_tokens[:sample_size])

# def show_sentiment_analysis(text):
#     st.header("Sentiment Analysis")
    
#     with st.spinner("Analyzing sentiment..."):
#         # Analyze sentiment
#         sentiment_results = analyze_sentiment(text)
        
#         # Display results
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Overall Sentiment")
            
#             # TextBlob sentiment
#             polarity = sentiment_results['polarity']
#             subjectivity = sentiment_results['subjectivity']
            
#             # Determine sentiment label
#             if polarity > 0.1:
#                 sentiment_label = "Positive 😊"
#                 color = "green"
#             elif polarity < -0.1:
#                 sentiment_label = "Negative 😞"
#                 color = "red"
#             else:
#                 sentiment_label = "Neutral 😐"
#                 color = "blue"
            
#             st.markdown(f"<h3 style='color: {color};'>{sentiment_label}</h3>", unsafe_allow_html=True)
            
#             # Display metrics
#             st.metric("Polarity", f"{polarity:.2f}", f"{polarity:.2f}")
#             st.metric("Subjectivity", f"{subjectivity:.2f}")
            
#             st.info("Polarity ranges from -1 (negative) to 1 (positive). Subjectivity ranges from 0 (objective) to 1 (subjective).")
        
#         with col2:
#             st.subheader("Sentence-level Analysis")
            
#             # Analyze individual sentences
#             sentences = sent_tokenize(text)
#             sentence_sentiments = []
            
#             for sentence in sentences:
#                 blob = TextBlob(sentence)
#                 sentence_sentiments.append({
#                     'sentence': sentence,
#                     'polarity': blob.sentiment.polarity,
#                     'subjectivity': blob.sentiment.subjectivity
#                 })
            
#             # Create DataFrame
#             df = pd.DataFrame(sentence_sentiments)
            
#             # Color code based on sentiment
#             def color_sentiment(val):
#                 if val > 0.1:
#                     return 'background-color: rgba(0, 255, 0, 0.2)'
#                 elif val < -0.1:
#                     return 'background-color: rgba(255, 0, 0, 0.2)'
#                 return 'background-color: rgba(0, 0, 255, 0.1)'
            
#             # Display styled DataFrame
#             st.dataframe(df.style.applymap(color_sentiment, subset=['polarity']))
            
#             # Plot sentiment distribution
#             if len(sentences) > 1:
#                 st.subheader("Sentiment Distribution")
#                 fig, ax = plt.subplots(figsize=(10, 4))
#                 ax.bar(range(len(sentences)), df['polarity'], color=[
#                     'green' if x > 0.1 else 'red' if x < -0.1 else 'blue' for x in df['polarity']
#                 ])
#                 ax.set_xlabel('Sentence Index')
#                 ax.set_ylabel('Polarity')
#                 ax.set_ylim(-1, 1)
#                 st.pyplot(fig)

# def show_ner(text, nlp):
#     st.header("Named Entity Recognition")
    
#     with st.spinner("Extracting entities..."):
#         # Extract entities
#         entities = extract_entities(text, nlp)
        
#         # Group entities by type
#         entity_groups = {}
#         for entity in entities:
#             if entity['label'] not in entity_groups:
#                 entity_groups[entity['label']] = []
#             entity_groups[entity['label']].append(entity['text'])
        
#         # Display results
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             st.subheader("Entities in Text")
            
#             # Create DataFrame
#             df = pd.DataFrame(entities)
            
#             if not df.empty:
#                 st.dataframe(df)
#             else:
#                 st.info("No entities found in the text.")
        
#         with col2:
#             st.subheader("Entity Types")
            
#             if entity_groups:
#                 for entity_type, entity_list in entity_groups.items():
#                     with st.expander(f"{entity_type} ({len(entity_list)})"):
#                         st.write(", ".join(set(entity_list)))
#             else:
#                 st.info("No entities found in the text.")
        
#         # Visualize entities in text
#         st.subheader("Visualized Entities")
        
#         if entities:
#             # Create HTML with highlighted entities
#             html = text
#             for entity in sorted(entities, key=lambda x: len(x['text']), reverse=True):
#                 color = {
#                     'PERSON': 'rgba(166, 226, 45, 0.4)',
#                     'ORG': 'rgba(67, 198, 252, 0.4)',
#                     'GPE': 'rgba(253, 151, 31, 0.4)',
#                     'LOC': 'rgba(253, 151, 31, 0.4)',
#                     'DATE': 'rgba(255, 102, 255, 0.4)',
#                     'TIME': 'rgba(255, 102, 255, 0.4)',
#                     'MONEY': 'rgba(153, 255, 153, 0.4)',
#                     'PERCENT': 'rgba(153, 255, 153, 0.4)',
#                 }.get(entity['label'], 'rgba(180, 180, 180, 0.4)')
                
#                 html = html.replace(entity['text'], f"<span style='background-color: {color};'>{entity['text']} <small>[{entity['label']}]</small></span>")
            
#             st.markdown(f"<div style='background-color: white; padding: 10px; border-radius: 5px;'>{html}</div>", unsafe_allow_html=True)
#         else:
#             st.info("No entities found to visualize.")

# def show_text_summarization(text):
#     st.header("Text Summarization")
    
#     with st.spinner("Generating summary..."):
#         # Get summarization options
#         ratio = st.slider("Summary Length (% of original text)", 10, 50, 30) / 100
        
#         # Generate summary
#         summary = summarize_text(text, ratio=ratio)
        
#         # Display results
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Original Text")
#             st.write(text)
#             st.info(f"Original Length: {len(text)} characters, {len(word_tokenize(text))} words")
        
#         with col2:
#             st.subheader("Summary")
#             if summary:
#                 st.write(summary)
#                 st.success(f"Summary Length: {len(summary)} characters, {len(word_tokenize(summary))} words")
#                 st.metric("Compression Ratio", f"{len(summary)/len(text):.1%}")
#             else:
#                 st.warning("Could not generate summary. The text might be too short or not suitable for summarization.")

# def show_word_cloud(text):
#     st.header("Word Cloud Visualization")
    
#     with st.spinner("Generating word cloud..."):
#         # Preprocess text for word cloud
#         processed_text = preprocess_text(text, remove_stopwords=True, remove_punctuation=True)
        
#         if processed_text:
#             # Generate word cloud
#             wordcloud = WordCloud(
#                 width=800, 
#                 height=400, 
#                 background_color='white',
#                 max_words=150,
#                 contour_width=3,
#                 contour_color='steelblue'
#             ).generate(processed_text)
            
#             # Display word cloud
#             fig, ax = plt.subplots(figsize=(10, 5))
#             ax.imshow(wordcloud, interpolation='bilinear')
#             ax.axis('off')
#             st.pyplot(fig)
            
#             # Display most common words
#             tokens = word_tokenize(processed_text.lower())
#             word_freq = {}
#             for token in tokens:
#                 if len(token) > 1:  # Filter out single-character tokens
#                     word_freq[token] = word_freq.get(token, 0) + 1
            
#             # Sort by frequency
#             sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
#             st.subheader("Most Common Words")
            
#             # Create DataFrame
#             df = pd.DataFrame(sorted_words[:20], columns=['Word', 'Frequency'])
            
#             # Display as bar chart
#             fig, ax = plt.subplots(figsize=(10, 5))
#             ax.barh(df['Word'], df['Frequency'])
#             ax.set_xlabel('Frequency')
#             ax.invert_yaxis()  # Display the most frequent at the top
#             st.pyplot(fig)
#         else:
#             st.warning("Not enough meaningful words to generate a word cloud.")

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.express as px
import base64
from io import BytesIO
import time
import os

# Try to import shadcn UI components if available
try:
    import streamlit_shadcn_ui as ui
    has_shadcn = True
except ImportError:
    has_shadcn = False

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="NLP Text Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4527A0;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5E35B1;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-text {
        color: #555;
        font-size: 0.9rem;
    }
    .stButton>button {
        background-color: #6200EA;
        color: white;
        border-radius: 0.3rem;
    }
    .stButton>button:hover {
        background-color: #3700B3;
        border-color: #3700B3;
    }
</style>
""", unsafe_allow_html=True)

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_sm')

# Text preprocessing function
def preprocess_text(text, remove_stopwords=True, remove_punctuation=True, lemmatize=True):
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Remove punctuation
    if remove_punctuation:
        tokens = [token for token in tokens if token.isalnum()]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

# Sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
    }

# Named entity recognition function
def extract_entities(text, nlp):
    if not text:
        return []
    
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_
        })
    return entities

# Text summarization function
def summarize_text(text, ratio=0.3):
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
        summary = ' '.join(sentences[:n_sentences])
        
        return summary
    except Exception as e:
        print(f"Error in summarization: {e}")
        return text[:int(len(text) * ratio)]

# Function to create download link
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # Create a bytes object for a file download
    b64 = base64.b64encode(object_to_download.encode()).decode()
    
    # Generate download link
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    return href

def main():
    # Initialize resources
    download_nltk_resources()
    nlp = load_spacy_model()
    
    # Application header with logo and title in the same row
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/data/logo.png", width=80)
    with col2:
        st.markdown('<h1 class="main-header">NLP Text Analysis Application</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application provides various Natural Language Processing tools to analyze your text.
    Enter your text and select the analysis you want to perform.
    """)
    
    # Create a better layout with columns for the main interface
    main_col1, main_col2 = st.columns([3, 1])
    
    with main_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Analysis Options</h3>', unsafe_allow_html=True)
        
        # Use shadcn UI if available, otherwise use standard Streamlit components
        if has_shadcn:
            analysis_option = ui.select(
                options=["Text Preprocessing", "Sentiment Analysis", "Named Entity Recognition", 
                         "Text Summarization", "Word Cloud"],
                label="Choose Analysis Type",
                key="analysis_select"
            )
        else:
            analysis_option = st.selectbox(
                "Choose Analysis Type",
                ["Text Preprocessing", "Sentiment Analysis", "Named Entity Recognition", 
                 "Text Summarization", "Word Cloud"]
            )
        
        st.markdown('<h3 class="sub-header">Input Method</h3>', unsafe_allow_html=True)
        text_input_method = st.radio("Select Input Method", ["Text Area", "File Upload"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add information card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">About</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">This app uses various NLP techniques to analyze text:</p>
        <ul class="info-text">
            <li>Text preprocessing (tokenization, stopword removal)</li>
            <li>Sentiment analysis with TextBlob</li>
            <li>Named Entity Recognition with spaCy</li>
            <li>Text summarization</li>
            <li>Word cloud visualization</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with main_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Text Input</h3>', unsafe_allow_html=True)
        
        # Text input based on selected method
        if text_input_method == "Text Area":
            text_input = st.text_area("Enter your text here:", height=200)
        else:
            uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
            if uploaded_file is not None:
                text_input = uploaded_file.read().decode("utf-8")
                st.text_area("File content:", text_input, height=200)
            else:
                text_input = ""
        
        # Add analyze button
        if has_shadcn:
            analyze_button = ui.button(text="Analyze Text", key="analyze_btn")
        else:
            analyze_button = st.button("Analyze Text")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process based on selected option
    if text_input and analyze_button:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="sub-header">📊 {analysis_option} Results</h2>', unsafe_allow_html=True)
        
        if analysis_option == "Text Preprocessing":
            show_text_preprocessing(text_input)
        
        elif analysis_option == "Sentiment Analysis":
            show_sentiment_analysis(text_input)
        
        elif analysis_option == "Named Entity Recognition":
            show_ner(text_input, nlp)
        
        elif analysis_option == "Text Summarization":
            show_text_summarization(text_input)
        
        elif analysis_option == "Word Cloud":
            show_word_cloud(text_input)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add download results option
        if analysis_option != "Word Cloud":  # Word cloud is visual, not easily downloadable as text
            results_text = generate_results_text(text_input, analysis_option, nlp)
            st.markdown(
                download_link(results_text, f"{analysis_option.lower().replace(' ', '_')}_results.txt", "📥 Download Results"),
                unsafe_allow_html=True
            )
    elif not text_input:
        st.info("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit and Python NLP libraries")

def generate_results_text(text, analysis_type, nlp):
    """Generate text results for download based on analysis type"""
    if analysis_type == "Text Preprocessing":
        processed = preprocess_text(text)
        return f"Original Text:\n{text}\n\nProcessed Text:\n{processed}"
    
    elif analysis_type == "Sentiment Analysis":
        sentiment = analyze_sentiment(text)
        polarity = sentiment['polarity']
        subjectivity = sentiment['subjectivity']
        
        sentiment_label = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
        
        return f"Text:\n{text}\n\nSentiment: {sentiment_label}\nPolarity: {polarity:.2f}\nSubjectivity: {subjectivity:.2f}"
    
    elif analysis_type == "Named Entity Recognition":
        entities = extract_entities(text, nlp)
        result = f"Text:\n{text}\n\nEntities Found:\n"
        
        for entity in entities:
            result += f"{entity['text']} - {entity['label']}\n"
        
        return result
    
    elif analysis_type == "Text Summarization":
        summary = summarize_text(text)
        return f"Original Text:\n{text}\n\nSummary:\n{summary}"
    
    return "No results available for download."

def show_text_preprocessing(text):
    st.markdown('<h3 class="sub-header">Text Preprocessing</h3>', unsafe_allow_html=True)
    
    with st.spinner("Processing text..."):
        # Get preprocessing options
        remove_stopwords = st.checkbox("Remove Stopwords", value=True)
        remove_punctuation = st.checkbox("Remove Punctuation", value=True)
        lemmatize = st.checkbox("Lemmatize", value=True)
        
        # Process text
        processed_text = preprocess_text(
            text, 
            remove_stopwords=remove_stopwords,
            remove_punctuation=remove_punctuation,
            lemmatize=lemmatize
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 class="sub-header">Original Text</h4>', unsafe_allow_html=True)
            st.write(text)
            
            st.markdown('<h4 class="sub-header">Token Count</h4>', unsafe_allow_html=True)
            tokens = word_tokenize(text)
            st.write(f"Number of tokens: {len(tokens)}")
            
            st.markdown('<h4 class="sub-header">Sentence Count</h4>', unsafe_allow_html=True)
            sentences = sent_tokenize(text)
            st.write(f"Number of sentences: {len(sentences)}")
        
        with col2:
            st.markdown('<h4 class="sub-header">Processed Text</h4>', unsafe_allow_html=True)
            st.write(processed_text)
            
            st.markdown('<h4 class="sub-header">Processed Token Count</h4>', unsafe_allow_html=True)
            processed_tokens = word_tokenize(processed_text)
            st.write(f"Number of tokens after processing: {len(processed_tokens)}")
            
            # Show sample tokens
            if processed_tokens:
                st.markdown('<h4 class="sub-header">Sample Tokens</h4>', unsafe_allow_html=True)
                sample_size = min(10, len(processed_tokens))
                st.write(processed_tokens[:sample_size])

def show_sentiment_analysis(text):
    st.markdown('<h3 class="sub-header">Sentiment Analysis</h3>', unsafe_allow_html=True)
    
    with st.spinner("Analyzing sentiment..."):
        # Analyze sentiment
        sentiment_results = analyze_sentiment(text)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 class="sub-header">Overall Sentiment</h4>', unsafe_allow_html=True)
            
            # TextBlob sentiment
            polarity = sentiment_results['polarity']
            subjectivity = sentiment_results['subjectivity']
            
            # Determine sentiment label
            if polarity > 0.1:
                sentiment_label = "Positive 😊"
                color = "green"
            elif polarity < -0.1:
                sentiment_label = "Negative 😞"
                color = "red"
            else:
                sentiment_label = "Neutral 😐"
                color = "blue"
            
            st.markdown(f"<h3 style='color: {color};'>{sentiment_label}</h3>", unsafe_allow_html=True)
            
            # Display metrics
            st.metric("Polarity", f"{polarity:.2f}", f"{polarity:.2f}")
            st.metric("Subjectivity", f"{subjectivity:.2f}")
            
            st.info("Polarity ranges from -1 (negative) to 1 (positive). Subjectivity ranges from 0 (objective) to 1 (subjective).")
        
        with col2:
            st.markdown('<h4 class="sub-header">Sentence-level Analysis</h4>', unsafe_allow_html=True)
            
            # Analyze individual sentences
            sentences = sent_tokenize(text)
            sentence_sentiments = []
            
            for sentence in sentences:
                blob = TextBlob(sentence)
                sentence_sentiments.append({
                    'sentence': sentence,
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
            
            # Create DataFrame
            df = pd.DataFrame(sentence_sentiments)
            
            # Color code based on sentiment
            def color_sentiment(val):
                if val > 0.1:
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                elif val < -0.1:
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                return 'background-color: rgba(0, 0, 255, 0.1)'
            
            # Display styled DataFrame
            st.dataframe(df.style.applymap(color_sentiment, subset=['polarity']))
            
            # Plot sentiment distribution with Plotly
            if len(sentences) > 1:
                st.markdown('<h4 class="sub-header">Sentiment Distribution</h4>', unsafe_allow_html=True)
                
                # Create interactive Plotly chart
                fig = px.bar(
                    df, 
                    x=df.index, 
                    y='polarity',
                    color='polarity',
                    color_continuous_scale=['red', 'blue', 'green'],
                    labels={'polarity': 'Polarity', 'index': 'Sentence Index'},
                    title="Sentiment Distribution by Sentence"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def show_ner(text, nlp):
    st.markdown('<h3 class="sub-header">Named Entity Recognition</h3>', unsafe_allow_html=True)
    
    with st.spinner("Extracting entities..."):
        # Extract entities
        entities = extract_entities(text, nlp)
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            if entity['label'] not in entity_groups:
                entity_groups[entity['label']] = []
            entity_groups[entity['label']].append(entity['text'])
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h4 class="sub-header">Entities in Text</h4>', unsafe_allow_html=True)
            
            # Create DataFrame
            df = pd.DataFrame(entities)
            
            if not df.empty:
                # Create interactive table with Plotly
                fig = px.bar(
                    df.groupby('label').count().reset_index(),
                    x='label',
                    y='text',
                    color='label',
                    labels={'text': 'Count', 'label': 'Entity Type'},
                    title="Entity Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display dataframe
                st.dataframe(df)
            else:
                st.info("No entities found in the text.")
        
        with col2:
            st.markdown('<h4 class="sub-header">Entity Types</h4>', unsafe_allow_html=True)
            
            if entity_groups:
                for entity_type, entity_list in entity

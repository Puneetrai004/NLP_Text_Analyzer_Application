import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from textblob import TextBlob
import time

from utils.preprocessing import preprocess_text
from utils.sentiment import analyze_sentiment
from utils.ner import extract_entities
from utils.summarization import summarize_text

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

def main():
    # Initialize resources
    download_nltk_resources()
    nlp = load_spacy_model()
    
    # Set page configuration
    st.set_page_config(
        page_title="NLP Text Analysis App",
        page_icon="📊",
        layout="wide"
    )
    
    # Application title and description
    st.title("NLP Text Analysis Application")
    st.markdown("""
    This application provides various Natural Language Processing tools to analyze your text.
    Enter your text and select the analysis you want to perform.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_option = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Text Preprocessing", "Sentiment Analysis", "Named Entity Recognition", "Text Summarization", "Word Cloud"]
    )
    
    # Text input area
    text_input = st.text_area("Enter your text here:", height=200)
    
    # Process based on selected option
    if text_input:
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
    else:
        st.info("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit and Python NLP libraries")

def show_text_preprocessing(text):
    st.header("Text Preprocessing")
    
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
            st.subheader("Original Text")
            st.write(text)
            
            st.subheader("Token Count")
            tokens = word_tokenize(text)
            st.write(f"Number of tokens: {len(tokens)}")
            
            st.subheader("Sentence Count")
            sentences = sent_tokenize(text)
            st.write(f"Number of sentences: {len(sentences)}")
        
        with col2:
            st.subheader("Processed Text")
            st.write(processed_text)
            
            st.subheader("Processed Token Count")
            processed_tokens = word_tokenize(processed_text)
            st.write(f"Number of tokens after processing: {len(processed_tokens)}")
            
            # Show sample tokens
            if processed_tokens:
                st.subheader("Sample Tokens")
                sample_size = min(10, len(processed_tokens))
                st.write(processed_tokens[:sample_size])

def show_sentiment_analysis(text):
    st.header("Sentiment Analysis")
    
    with st.spinner("Analyzing sentiment..."):
        # Analyze sentiment
        sentiment_results = analyze_sentiment(text)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Sentiment")
            
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
            st.subheader("Sentence-level Analysis")
            
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
            
            # Plot sentiment distribution
            if len(sentences) > 1:
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(sentences)), df['polarity'], color=[
                    'green' if x > 0.1 else 'red' if x < -0.1 else 'blue' for x in df['polarity']
                ])
                ax.set_xlabel('Sentence Index')
                ax.set_ylabel('Polarity')
                ax.set_ylim(-1, 1)
                st.pyplot(fig)

def show_ner(text, nlp):
    st.header("Named Entity Recognition")
    
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
            st.subheader("Entities in Text")
            
            # Create DataFrame
            df = pd.DataFrame(entities)
            
            if not df.empty:
                st.dataframe(df)
            else:
                st.info("No entities found in the text.")
        
        with col2:
            st.subheader("Entity Types")
            
            if entity_groups:
                for entity_type, entity_list in entity_groups.items():
                    with st.expander(f"{entity_type} ({len(entity_list)})"):
                        st.write(", ".join(set(entity_list)))
            else:
                st.info("No entities found in the text.")
        
        # Visualize entities in text
        st.subheader("Visualized Entities")
        
        if entities:
            # Create HTML with highlighted entities
            html = text
            for entity in sorted(entities, key=lambda x: len(x['text']), reverse=True):
                color = {
                    'PERSON': 'rgba(166, 226, 45, 0.4)',
                    'ORG': 'rgba(67, 198, 252, 0.4)',
                    'GPE': 'rgba(253, 151, 31, 0.4)',
                    'LOC': 'rgba(253, 151, 31, 0.4)',
                    'DATE': 'rgba(255, 102, 255, 0.4)',
                    'TIME': 'rgba(255, 102, 255, 0.4)',
                    'MONEY': 'rgba(153, 255, 153, 0.4)',
                    'PERCENT': 'rgba(153, 255, 153, 0.4)',
                }.get(entity['label'], 'rgba(180, 180, 180, 0.4)')
                
                html = html.replace(entity['text'], f"<span style='background-color: {color};'>{entity['text']} <small>[{entity['label']}]</small></span>")
            
            st.markdown(f"<div style='background-color: white; padding: 10px; border-radius: 5px;'>{html}</div>", unsafe_allow_html=True)
        else:
            st.info("No entities found to visualize.")

def show_text_summarization(text):
    st.header("Text Summarization")
    
    with st.spinner("Generating summary..."):
        # Get summarization options
        ratio = st.slider("Summary Length (% of original text)", 10, 50, 30) / 100
        
        # Generate summary
        summary = summarize_text(text, ratio=ratio)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Text")
            st.write(text)
            st.info(f"Original Length: {len(text)} characters, {len(word_tokenize(text))} words")
        
        with col2:
            st.subheader("Summary")
            if summary:
                st.write(summary)
                st.success(f"Summary Length: {len(summary)} characters, {len(word_tokenize(summary))} words")
                st.metric("Compression Ratio", f"{len(summary)/len(text):.1%}")
            else:
                st.warning("Could not generate summary. The text might be too short or not suitable for summarization.")

def show_word_cloud(text):
    st.header("Word Cloud Visualization")
    
    with st.spinner("Generating word cloud..."):
        # Preprocess text for word cloud
        processed_text = preprocess_text(text, remove_stopwords=True, remove_punctuation=True)
        
        if processed_text:
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=150,
                contour_width=3,
                contour_color='steelblue'
            ).generate(processed_text)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Display most common words
            tokens = word_tokenize(processed_text.lower())
            word_freq = {}
            for token in tokens:
                if len(token) > 1:  # Filter out single-character tokens
                    word_freq[token] = word_freq.get(token, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            st.subheader("Most Common Words")
            
            # Create DataFrame
            df = pd.DataFrame(sorted_words[:20], columns=['Word', 'Frequency'])
            
            # Display as bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(df['Word'], df['Frequency'])
            ax.set_xlabel('Frequency')
            ax.invert_yaxis()  # Display the most frequent at the top
            st.pyplot(fig)
        else:
            st.warning("Not enough meaningful words to generate a word cloud.")

if __name__ == "__main__":
    main()

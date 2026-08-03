import streamlit as st
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download NLTK data (first time only)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

st.title("NLP Pipeline Demo")
st.write("Enter text and explore NLP steps")

# User input
text = st.text_area("Enter your text here:")

if text:

    st.subheader("1. SentenceTokenization")
    sentences = sent_tokenize(text)
    st.write(sentences)

    st.subheader("2. Word Tokenization")
    tokens = word_tokenize(text)
    st.write(tokens)

    st.subheader("3. Stemming")
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]
    st.write(stemmed)

    st.subheader("4. Lemmatization")
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    st.write(lemmatized)

    st.subheader("5. Stopword Removal")
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word.lower() not in stop_words]
    st.write(filtered)

    st.subheader("6. POS Tagging")
    pos_tags = nltk.pos_tag(tokens)
    st.write(pos_tags)

    st.subheader("7. Named Entity Recognition (NER)")
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    st.write(entities)

    st.subheader("8. chunking")
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    st.chunk_parser = nltk.RegexpParser(grammar)
    tree = st.chunk_parser.parse(pos_tags)
    st.write(tree)
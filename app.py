import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# Function to translate documents
def translate_documents(documents, target_language='en'):
    translated_documents = []
    for document in documents:
        translation = GoogleTranslator(source='auto', target=target_language).translate(document)
        translated_documents.append(translation)
    return translated_documents

# Function for semantic search
def semantic_search(query, translated_documents, vectorizer, X, threshold=0.3):
    translated_query = GoogleTranslator(source='auto', target='en').translate(query)
    query_vector = vectorizer.transform([translated_query])
    cosine_similarities = cosine_similarity(query_vector, X).flatten()
    related_documents = []
    for i, score in enumerate(cosine_similarities):
        if score > threshold:
            related_documents.append((score, translated_documents[i]))
    related_documents.sort(reverse=True)
    return related_documents

# Streamlit app
def main():
    st.title("Semantic Search")

    # Input for documents
    st.subheader("Input Documents")
    documents_input = st.text_area("Enter documents (one per line)", height=150)

    # Input for query
    st.subheader("Query")
    query = st.text_input("Enter your query:")

    # Button to perform search
    if st.button("Search"):
        if documents_input and query:
            documents = documents_input.split('\n')
            # Translate documents to English
            translated_documents = translate_documents(documents)
            
            # Vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(translated_documents)
            
            # Perform semantic search
            results = semantic_search(query, translated_documents, vectorizer, X, threshold=0.3)
            
            # Display search results
            if results:
                st.subheader("Search Results:")
                for score, document in results:
                    st.markdown(f'<div style="border: 1.5px solid white; padding: 10px;">{document} [{score * 100:.2f}%]</div>', unsafe_allow_html=True)
            else:
                st.write("No relevant documents found.")
        else:
            st.write("Please input both documents and query.")

if __name__ == "__main__":
    main()


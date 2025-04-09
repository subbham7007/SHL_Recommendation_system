## pip install streamlit 
##  To RUN this File :-  "streamlit run app.py"

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
from newspaper import Article
# from urllib.parse import urlparse
# from google import genai

# Load catalog with only Unique courses scraped from product-catalog page
newdf = pd.read_csv('Cleaned_SHL_catalog.csv')

# Load model and FAISS
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
assessment_embeddings = model.encode(newdf['Description'])
index = faiss.IndexFlatL2(assessment_embeddings.shape[1])
index.add(assessment_embeddings)

# Util
def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def extract_text_from_query(query):
    urls = []
    text = []
    for word in query.split():
        if word.startswith('http'):
            urls.append(word)
        else:
            text.append(word)
    return urls, ' '.join(text)



def recommend_assessments(query_text, k=10):
    urls, text = extract_text_from_query(query_text)
    query_text = text
    if urls:
        for url in urls:
            query_text += ' ' + extract_text_from_url(url)

    query_vec = model.encode([query_text])
    distances, indices = index.search(query_vec, k)
    indices=np.unique(indices[0])
    return newdf.iloc[indices]

    #     # Use Google Gemini API to get embeddings
    # result = client.models.embed_content(
    #     model="gemini-embedding-exp-03-07",
    #     contents=query_text)
    # query_vec = np.array(result.embeddings)
    
    # distances, indices = index.search(query_vec.reshape(1, -1), k)
    # indices=np.unique(indices[0])
    # return newdf.iloc[indices]


    

# Streamlit UI
st.title("🔍 SHL Assessment Recommendation System")
input_text = st.text_area("Paste job description text or URL", height=200)

if st.button("Get Recommendations"):
    if input_text.startswith("http"):
        input_text = extract_text_from_url(input_text)
    
    # recommendations = recommend_assessments(input_text)
    # st.write("### Recommended Assessments:")
    # st.dataframe(recommendations[['name', 'url', 'remote', 'adaptive', 'duration', 'type']])
    
    recommendations = recommend_assessments(input_text)
    st.write("### Recommended Assessments:")
    for index, row in recommendations.iterrows():
        st.write(f"**{row['Assessment Name']}**")
        st.write(f"URL: {row['URL']}")
        st.write(f"Remote: {row['Remote Testing Support']}")
        st.write(f"Adaptive: {row['Adaptive/IRT Support']}")
        st.write(f"Duration: {row['Assessment Length']}")
        st.write(f"Type: {row['Test Type']}")
        st.write("----")

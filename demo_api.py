from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import faiss
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load your dataframe (assuming it's a CSV file)
newdf = pd.read_csv('Cleaned_SHL_catalog.csv')

# Create embeddings for the 'Description' column
embeddings = model.encode(newdf['Description'])

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Add the encoded embeddings to the FAISS index
index.add(embeddings)


@app.route('/search', methods=['POST'])
def handle_post():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON payload'}), 400

    # Process the JSON data here
    print(data)

     # Get the query text from the request body
    query_text = data['query']

    # Encode the query text
    query_embedding = model.encode([query_text])

    # Search for similar embeddings
    D, I = index.search(query_embedding, k=20)

    # Remove duplicates from the indices array
    I = np.unique(I[0])

    # Select the top 10 most similar embeddings
    most_relevant_indices = I[:10]

    # Get the most similar courses
    most_similar_courses = newdf.iloc[most_relevant_indices]

    # Convert the result to a JSON response
    response = most_similar_courses.to_dict(orient='records')

    return jsonify(response)
    # Return a response
    # return jsonify({'message': 'JSON payload received successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)

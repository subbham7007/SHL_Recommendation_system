# SHL Assessment Recommendation System

This app recommends SHL assessments based on job description or query text which uses FAISS search to retrieve relevant courses based on the query provided. It can also work with link which points to the JD for a particular role.

## Features
- Web Scrapping using bs4, 
- Web app using Streamlit
- REST API using Flask
- Semantic search using Sentence-BERT(3 options and many morecould be added) + FAISS

## How to run
```bash
1. Create your own Environment using the code "python -m venv <your_env_name"> . Run this on your Terminal/Command prompt (Keep in mind to switch to the root directory where your file is stored using "cd <Directory_path>")
2. RUN(the following command)-- to install all the dependencies
This ->>pip install -r requirements.txt

3.RUN(the following command) --For Streamlit file i.e., main.py and main2.py
This :--> "streamlit run <your_file_name>.py "

4. RUN -- For FLASK API this is a POST API JSON request body {"query" : "Your_query(may involve links starting with "https:")"}
This :-->  "uvicorn demo_api:app --reload"  or "python demo_api.py

**Note:** if using any Personnel Credentials use python load_dotenv() technique to safeguard them from miss use and do not UPLOAD the .env file along these on your GitHub profile

Cricket Intelligence Assistant (CIA)
--------------------------------------------
LLM-powered Cricket Analytics System that can:
1. Ingest & process cricket data (player stats, team performances, match records).
2. Answer natural language queries related to cricket history, records, and player stats.
3. Summarize key insights using an LLM-powered retrieval system.
4. Deploy an interactive API or chatbot for cricket queries.

1. Data Processing & Engineering
--------------------------------------

Data source - https://cricsheet.org/matches/   Test matches, One-day internationals, T20 internationals, CSA T20 Challenge, 
              Indian Premier League data are colleccted, transformed and stored in src/data/cricket_data.
The data after transformation are stored in csv format files - match_data.csv , player_data.csv

Data transformation code are available in jupyter notebook and python script in locations utils/Data_transformation.ipynb 
        and utils/feature_collection.py
json_to_csv.py is to convert the json file directly to csv without any transformations.

2. Large Language Model (LLM) Integration
--------------------------------------------
RAG pipeline is used for getting response for the user queries.
LLM used is Mistral-7B-Instruct-v0.1-GGUF
VectorDB FAISS

the script and implimations are available in src/main.py and RAG_pipeline.ipynb

3. API Deployment & Chatbot Integration
-------------------------------------------
The code available in Main.py and Collections are available in CricketAI.postman_collection.json

The possible Q&A are tested and got result.

The Project is Scuccessfull

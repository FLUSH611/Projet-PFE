# Creates folders and installs base requirements
mkdir data, data\raw, data\interim, data\processed, vectors, out, logs -ErrorAction SilentlyContinue
mkdir src\{config,utils,ingestion,indexing,rag,generation,scraping,ui} -ErrorAction SilentlyContinue
pip install --upgrade pip
pip install -r requirements.txt
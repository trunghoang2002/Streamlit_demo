#!/bin/bash

# Run FastAPI app in the background
nohup uvicorn fa_lenet:app --host 127.0.0.1 --port 8000 &

# Run Streamlit app
streamlit run st_lenet.py

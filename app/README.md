# Contextual Retrieval Chatbot App

This is a small Flask + HTML app that implements the assignment chatbot requirement
using your Contextual Retrieval pipeline.

## Book Reference

- Source textbook: Speech and Language Processing (Jurafsky and Martin, 3rd edition)
- Chapter used in this app: Chapter 4 - Logistic Regression and Text Classification

## What It Uses

- Retriever: sentence-transformers/all-MiniLM-L6-v2
- Generator: translategemma:latest via Ollama
- Retrieval: contextual retrieval over enriched chunks
- Citation: chunk index, rank, distance, and chunk text are shown for each answer

## Prerequisites

1. Keep `contextual_chunks.json` in the `app` folder.
2. Start Ollama with `translategemma:latest` available.
3. Install dependencies from `requirements.txt`.

## Run

From the `app` folder:

python app.py

Then open:

http://127.0.0.1:5000

## Optional Environment Variables

- `OLLAMA_URL` (default: `http://localhost:11434/api/generate`)
- `OLLAMA_MODEL` (default: `translategemma:latest`)
- `TOP_K` (default: `5`)

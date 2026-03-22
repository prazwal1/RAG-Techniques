# AT82.03 Machine Learning - A6

## Naive RAG vs Contextual Retrieval

This repository contains a complete implementation of Assignment 6: Naive RAG vs Contextual Retrieval.
The work includes data preparation, RAG pipeline implementation, evaluation on 20 QA pairs, analysis of
ROUGE metrics, and a web application that uses Contextual Retrieval for interactive QA.

## Student and Scope

- Student ID: st125974
- Assigned chapter: Chapter 4 - Logistic Regression and Text Classification
- Source textbook: Speech and Language Processing (Jurafsky and Martin, 3rd edition)

## Repository Structure

- Notebook solution: A6_RAG_Techniques.ipynb
- QA dataset (20 pairs): data/qa.json
- Submission JSON: answer/response-st125974-chapter-4.json
- Web app entrypoint: app/app.py
- Web app template: app/templates/index.html
- App dependencies: app/requirements.txt
- App documentation: app/README.md
- Demo video: demo-video/web-demo.mp4

## Task Coverage

### Task 1 - Source Discovery and Data Preparation

1. Extracted and cleaned chapter text from the assigned chapter.
2. Built a chunking pipeline with overlap for retrieval.
3. Prepared 20 chapter-grounded QA pairs in data/qa.json.

### Task 2 - Naive RAG vs Contextual Retrieval

1. Implemented Naive RAG.
2. Implemented Contextual Retrieval with chunk enrichment.
3. Ran both methods on the same 20 QA pairs.
4. Computed ROUGE-1, ROUGE-2, and ROUGE-L and discussed findings.

### Task 3 - Chatbot Web Application

1. Implemented a simple Flask + HTML chat interface.
2. Backend uses Contextual Retrieval from Task 2.
3. App displays generated answer and cited source chunks (index, rank, distance, text).

## Models Used

- Retriever model: sentence-transformers/all-MiniLM-L6-v2
- Generator model: translategemma:latest via Ollama
- Vector store: ChromaDB (cosine distance)

Both Naive and Contextual pipelines use the same retriever and generator. This keeps the comparison focused
on retrieval strategy differences.

## Architectures

### A. Evaluation Architecture (Task 2)

```text
PDF Chapter -> Clean Text -> Chunking
								 |
			+---------------+----------------+
			|                                |
		Naive RAG                    Contextual Retrieval
	Embed raw chunks                Enrich chunk with context sentence
	ChromaDB index                  Embed enriched chunks
	Top-k retrieval                 ChromaDB index
	Gemma-translate answer          Top-k retrieval
			|                         Gemma-translate answer
			+---------------+----------------+
								 |
				  ROUGE-1 / ROUGE-2 / ROUGE-L
```

### B. Web Application Architecture (Task 3)

```text
User Browser (Flask HTML UI)
				|
				v
		Flask app/app.py
				|
				v
Contextual Retrieval Backend
  - load contextual_chunks.json
  - embed with all-MiniLM-L6-v2
  - retrieve top-k from ChromaDB
				|
				v
Ollama (translategemma:latest)
				|
				v
Answer + Source Chunk Citations
```

## Evaluation Findings

Average ROUGE scores across 20 QA pairs:

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Naive RAG | 0.3882 | 0.1971 | 0.3062 |
| Contextual Retrieval | 0.3772 | 0.1819 | 0.3007 |

Best performing method in this run: Naive RAG.

Interpretation:

1. Under this specific setup and QA set, Naive RAG achieved slightly higher overlap metrics.
2. This result indicates contextual enrichment is sensitive to enrichment quality and prompt behavior.
3. Since both methods share the same generator, the difference mainly comes from retrieval content quality.

## Demo

<video src="demo-video/web-demo.mp4" controls width="600"></video>
- Demo video (download): [web-demo.mp4](demo-video/web-demo.mp4)


## How To Run

### Notebook

1. Open A6_RAG_Techniques.ipynb.
2. Run cells in order to reproduce extraction, indexing, and evaluation.
3. Confirm submission JSON exists in answer/response-st125974-chapter-4.json.

### Web App

1. Create and activate a Python virtual environment.
2. Install dependencies:
	pip install -r app/requirements.txt
3. Run the Flask app:
	python app/app.py
4. Open:
	http://127.0.0.1:5000

## Submission Checklist

- GitHub repository contains notebook, README, and app folder.
- Notebook includes Task 1, Task 2 implementation, evaluation, and discussion.
- JSON submission file is present in answer folder with required schema.
- Web app uses Contextual Retrieval and shows source chunk citations.
- Demo video is included and linked.

## Credits and References

1. Jurafsky, D., and Martin, J. H. Speech and Language Processing (3rd ed. draft). 
	https://web.stanford.edu/~jurafsky/slp3/
2. Anthropic Engineering. Contextual Retrieval.
	https://www.anthropic.com/engineering/contextual-retrieval

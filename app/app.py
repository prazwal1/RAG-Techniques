import json
import os
import random
from dataclasses import dataclass
from typing import Any

import chromadb
import requests
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    chunk_index: int
    distance: float
    text: str


class ContextualRAGBackend:
    """Contextual Retrieval backend adapted from the notebook implementation."""

    def __init__(self, contextual_chunks_path: str) -> None:
        if not os.path.exists(contextual_chunks_path):
            raise FileNotFoundError(
                f"contextual_chunks.json not found at: {contextual_chunks_path}. "
                "Run the notebook contextual enrichment cells first."
            )

        with open(contextual_chunks_path, "r", encoding="utf-8") as f:
            contextual_chunks = json.load(f)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client(
            chromadb.config.Settings(anonymized_telemetry=False)
        )

        try:
            self.chroma_client.delete_collection("contextual_chat_app")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name="contextual_chat_app",
            metadata={"hnsw:space": "cosine"},
        )

        embeddings = self.embedder.encode(contextual_chunks, show_progress_bar=False)
        self.collection.add(
            documents=contextual_chunks,
            embeddings=embeddings.tolist(),
            ids=[f"ctx_chunk_{i}" for i in range(len(contextual_chunks))],
            metadatas=[
                {"chunk_index": i, "source": "chapter4"}
                for i in range(len(contextual_chunks))
            ],
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

        retrieved: list[RetrievedChunk] = []
        for i in range(len(results["documents"][0])):
            retrieved.append(
                RetrievedChunk(
                    chunk_index=int(results["metadatas"][0][i]["chunk_index"]),
                    distance=float(results["distances"][0][i]),
                    text=results["documents"][0][i],
                )
            )
        return retrieved

    @staticmethod
    def generate_answer(
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        ollama_url: str,
        ollama_model: str,
    ) -> str:
        context = "\n\n---\n\n".join([c.text for c in retrieved_chunks])

        prompt = f"""You are an NLP textbook assistant answering questions about Chapter 4: Logistic Regression.
Answer ONLY using the provided context.
Be concise (2-4 sentences).
If the answer is not in the context, say 'Not found in context.'

Context:
{context}

Question: {question}
Answer:"""

        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 300, "temperature": 0},
        }

        response = requests.post(ollama_url, json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def answer(
        self,
        question: str,
        top_k: int,
        ollama_url: str,
        ollama_model: str,
    ) -> dict[str, Any]:
        retrieved = self.retrieve(query=question, top_k=top_k)
        answer = self.generate_answer(
            question=question,
            retrieved_chunks=retrieved,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )
        return {"answer": answer, "retrieved": retrieved}


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONTEXTUAL_PATH = os.path.join(APP_DIR, "..", "contextual_chunks.json")
DEFAULT_QA_PATH = os.path.join(APP_DIR, "..", "data", "qa.json")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "translategemma:latest")
TOP_K = int(os.getenv("TOP_K", "5"))

backend: ContextualRAGBackend | None = None
question_suggestions_cache: list[str] | None = None


def get_backend() -> ContextualRAGBackend:
    global backend
    if backend is None:
        backend = ContextualRAGBackend(DEFAULT_CONTEXTUAL_PATH)
    return backend


def get_question_suggestions() -> list[str]:
    global question_suggestions_cache
    if question_suggestions_cache is not None:
        return question_suggestions_cache

    suggestions: list[str] = []
    try:
        with open(DEFAULT_QA_PATH, "r", encoding="utf-8") as f:
            qa_items = json.load(f)
        for item in qa_items:
            q = item.get("question", "").strip()
            if q:
                suggestions.append(q)
    except Exception:
        suggestions = []

    question_suggestions_cache = suggestions
    return question_suggestions_cache


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    answer_text = ""
    sources: list[RetrievedChunk] = []
    question = ""
    error = ""
    selected_top_k = TOP_K
    all_question_suggestions = get_question_suggestions()
    question_suggestions = random.sample(
        all_question_suggestions,
        k=min(2, len(all_question_suggestions)),
    )

    try:
        rag_backend = get_backend()
    except Exception as exc:
        error = f"Backend init failed: {exc}"
        return render_template(
            "index.html",
            question=question,
            answer_text=answer_text,
            sources=sources,
            error=error,
            top_k=selected_top_k,
            ollama_model=OLLAMA_MODEL,
            ollama_url=OLLAMA_URL,
            question_suggestions=question_suggestions,
        )

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        try:
            selected_top_k = int(request.form.get("top_k", str(TOP_K)))
        except ValueError:
            selected_top_k = TOP_K

        if question:
            try:
                result = rag_backend.answer(
                    question=question,
                    top_k=max(1, min(selected_top_k, 10)),
                    ollama_url=OLLAMA_URL,
                    ollama_model=OLLAMA_MODEL,
                )
                answer_text = result.get("answer", "") or "No answer generated."
                sources = result.get("retrieved", [])
            except Exception as exc:
                error = f"Error while generating answer: {exc}"

    return render_template(
        "index.html",
        question=question,
        answer_text=answer_text,
        sources=sources,
        error=error,
        top_k=selected_top_k,
        ollama_model=OLLAMA_MODEL,
        ollama_url=OLLAMA_URL,
        question_suggestions=question_suggestions,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

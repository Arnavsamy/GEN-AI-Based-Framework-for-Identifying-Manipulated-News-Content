# GEN-AI-Based-Framework-for-Identifying-Manipulated-News-Content

A news manipulation detection system built using FastAPI, Streamlit, and Groq LLaMA. The project analyzes news articles for bias, emotional manipulation, propaganda patterns, and AI-generated content using a lightweight RAG pipeline with TF-IDF retrieval.

Features:

* News bias detection
* Emotional heatmap visualization
* Manipulation insights
* Neutral news rewriting
* RAG-based evidence retrieval
* Final verdict classification


Tech Stack:
Python, FastAPI, Streamlit, Groq API, LLaMA, Plotly, Pandas, TF-IDF Retrieval

How It Works:
The user submits a news article through the Streamlit interface. The backend processes the text and retrieves the most relevant trusted news chunks from a local knowledge base using TF-IDF similarity scoring. These retrieved results are then passed to the Groq LLaMA model, which analyzes the content for bias, emotional tone, propaganda techniques, and possible AI-generated patterns. Finally, the system displays visual insights such as the emotional heatmap, bias meter, retrieved evidence, and a final verdict indicating whether the content is real, manipulated, fake, or unverified.

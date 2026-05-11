import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="GEN AI Based Framework for Identifying Manipulated News Content",
    layout="wide",
)

with st.sidebar:
    st.subheader("Simple RAG Knowledge Base")
    try:
        kb_response = requests.get(f"{API_URL}/knowledge-base", timeout=3)
        if kb_response.status_code == 200:
            kb = kb_response.json()
            st.caption(f"{kb.get('chunks', 0)} chunks indexed")
            for source_name in kb.get("sources", [])[:5]:
                st.caption(f"- {source_name}")
        else:
            st.caption("Backend is running, but RAG status is unavailable.")
    except Exception:
        st.caption("Start groq_backend.py to use RAG.")

    with st.expander("Add trusted source"):
        source_name = st.text_input("Source name", placeholder="Example: official news source")
        source_text = st.text_area("Trusted source text", height=130)
        source_file = st.file_uploader("Upload trusted .txt file", type=["txt"])

        if st.button("Add Source"):
            final_source_text = source_text
            if source_file is not None:
                final_source_text += "\n\n" + source_file.getvalue().decode(
                    "utf-8",
                    errors="ignore",
                )

            if not final_source_text.strip():
                st.warning("Please paste trusted source text or upload a .txt file.")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/ingest",
                        data={
                            "text": final_source_text,
                            "source": source_name
                            or (source_file.name if source_file else "Manual source"),
                        },
                        timeout=30,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(
                            f"Added {result.get('chunks_added', 0)} chunks. "
                            f"Total indexed: {result.get('total_chunks', 0)}."
                        )
                        st.rerun()
                    else:
                        detail = response.json().get("detail", response.text)
                        st.error(f"Error {response.status_code}: {detail}")
                except Exception as e:
                    st.error(f"Could not connect to backend. {e}")

    st.divider()
    st.subheader("Emotional Heatmap")
    if "results" in st.session_state:
        emo = st.session_state.results.get("emotions", {})
        df = pd.DataFrame(dict(r=list(emo.values()), theta=list(emo.keys())))
        fig = px.line_polar(df, r="r", theta="theta", line_close=True)
        fig.update_traces(fill="toself", line_color="#007bff")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Analyze content to see heatmap.")

    st.divider()
    st.subheader("Neutral Rewrite")
    if "results" in st.session_state:
        st.write(st.session_state.results.get("neutral_rewrite", "N/A"))

    st.divider()
    st.subheader("Feedback")
    st.text_area("Observations", height=70)
    st.button("Submit Feedback")

st.title("GEN AI Based Framework for Identifying Manipulated News Content")

tabs = st.tabs(
    [
        "Input News",
        "Manipulation Insights",
        "Fact-Check and Sentiment",
        "Media and Resource",
        "Summary Verdict",
    ]
)

with tabs[0]:
    st.subheader("Paste the news content below")
    text_input = st.text_area("Enter News Content", height=250)
    news_file = st.file_uploader("Upload news .txt file", type=["txt"])

    if st.button("Analyze News"):
        final_text = text_input
        if news_file is not None:
            final_text += "\n\n" + news_file.getvalue().decode("utf-8", errors="ignore")

        if not final_text.strip():
            st.warning("Please enter news text or upload a .txt file.")
        else:
            with st.spinner("Analyzing with Groq LLaMA and RAG..."):
                try:
                    response = requests.post(
                        f"{API_URL}/analyze",
                        data={"text": final_text},
                        timeout=60,
                    )
                    if response.status_code == 200:
                        st.session_state.results = response.json()
                        st.rerun()
                    else:
                        detail = response.json().get("detail", response.text)
                        st.error(f"Error {response.status_code}: {detail}")
                except Exception as e:
                    st.error(f"Error: Could not connect to backend. {e}")

if "results" in st.session_state:
    res = st.session_state.results

    with tabs[1]:
        st.subheader("Manipulation Insights")
        tactics = res.get("manipulation_tactics", [])
        st.write(f"Techniques Detected: {', '.join(tactics) if tactics else 'None detected'}")
        st.write(f"AI Detection: {res.get('ai_detection', 'N/A')}% probability of AI generation.")
        st.write(f"Historical Match: {res.get('propaganda_era', 'N/A')} style propaganda.")

    with tabs[2]:
        st.subheader("Bias Meter and Fact Check")
        bias = res.get("bias_score", 0)
        fig_bias = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=bias,
                title={"text": "Bias Score"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "red"}},
            )
        )
        st.plotly_chart(fig_bias, use_container_width=True)

        st.subheader("Emotion Intensity Map")
        emotions = res.get("emotions", {})
        if emotions:
            emotion_df = pd.DataFrame(
                {"Emotion": list(emotions.keys()), "Intensity": list(emotions.values())}
            )
            fig_emotions = px.bar(
                emotion_df,
                x="Emotion",
                y="Intensity",
                color="Emotion",
                range_y=[0, 1],
            )
            st.plotly_chart(fig_emotions, use_container_width=True)

        st.subheader("RAG Assessment")
        st.write(res.get("rag_assessment", "N/A"))
        for point in res.get("evidence_summary", []):
            st.write(f"- {point}")

    with tabs[3]:
        st.subheader("Resource and Article Links")
        st.write(f"Story Mutation Tracker: {res.get('mutation_tracker', 'N/A')}")
        st.write("- Primary Source Verification: https://news.google.com")

        st.subheader("Retrieved RAG Evidence")
        evidence = res.get("rag_evidence", [])
        if evidence:
            max_score = max(item.get("score", 0) for item in evidence)
            match_percent = min(round(max_score * 100), 100)
            fig_match = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=match_percent,
                    title={"text": "RAG Match Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1f77b4"},
                        "steps": [
                            {"range": [0, 25], "color": "#ffcccc"},
                            {"range": [25, 60], "color": "#fff2b2"},
                            {"range": [60, 100], "color": "#c7f5d9"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig_match, use_container_width=True)

            for index, item in enumerate(evidence, start=1):
                title = f"Evidence {index}: {item.get('source', 'Unknown source')}"
                with st.expander(f"{title} | score {item.get('score', 0)}"):
                    if item.get("match_type") == "low_keyword_match":
                        st.warning("This source is indexed, but keyword similarity is low.")
                    st.write(item.get("text", ""))
        else:
            st.info("No RAG sources are indexed yet. Add a trusted source and click Add Source.")

    with tabs[4]:
        st.subheader("Summary Verdict")
        st.write(f"Verdict: {res.get('verdict', 'N/A')}")

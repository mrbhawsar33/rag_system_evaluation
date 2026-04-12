import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.title("🚗 RAG System Evaluation Dashboard")

# Load data
df = pd.read_csv("data/output/evaluation_results.csv")

# Sidebar filters
st.sidebar.header("Filters")
retriever = st.sidebar.selectbox("Retriever", df["retriever"].unique())
llm = st.sidebar.selectbox("LLM", df["llm"].unique())
prompt = st.sidebar.selectbox("Prompt", df["prompt"].unique())

filtered_df = df[
    (df["retriever"] == retriever) &
    (df["llm"] == llm) &
    (df["prompt"] == prompt)
]

st.subheader("Filtered Results")
st.dataframe(filtered_df)

# Aggregation
summary = df.groupby(["retriever", "llm", "prompt"]).agg({
    "recall_at_k": "mean",
    "mrr": "mean",
    "hallucination": "mean",
    "latency": "mean"
}).reset_index()

st.subheader("Aggregated Metrics")
st.dataframe(summary)

# Identify best configuration
st.subheader("Best Configuration")

# Define scoring logic (you can tweak weights if needed)
summary["score"] = (
    summary["recall_at_k"] * 0.4 +
    summary["mrr"] * 0.4 +
    (1 - summary["hallucination"]) * 0.2
)

best = summary.sort_values(by="score", ascending=False).iloc[0]

st.success(
    f"""
    Best Setup:
    - Retriever: {best['retriever']}
    - LLM: {best['llm']}
    - Prompt: {best['prompt']}

    Metrics:
    - Recall@K: {best['recall_at_k']:.3f}
    - MRR: {best['mrr']:.3f}
    - Hallucination: {best['hallucination']:.3f}
    """
)

# charts
# Recall@K Comparison
st.subheader("Recall@K Comparison")
chart = alt.Chart(summary).mark_bar().encode(
    x=alt.X('retriever:N', title='Retriever'),
    y=alt.Y('recall_at_k:Q', title='Recall@K'),
    color=alt.Color('llm:N', title='LLM'),
    xOffset='llm:N',   # THIS FIXES OVERLAP
    column=alt.Column('prompt:N', title='Prompt'),
    tooltip=['retriever', 'llm', 'prompt', 'recall_at_k']
)

st.altair_chart(chart, use_container_width=True)

# MRR Comparison
st.subheader("MRR Comparison")
chart = alt.Chart(summary).mark_bar().encode(
    x='retriever:N',
    y='mrr:Q',
    color='llm:N',
    xOffset='llm:N',
    column='prompt:N',
    tooltip=['retriever', 'llm', 'prompt', 'mrr']
)
st.altair_chart(chart, use_container_width=True)

# Hallucination Rate Comparison
st.subheader("Hallucination Rate")
chart = alt.Chart(summary).mark_bar().encode(
    x='retriever:N',
    y='hallucination:Q',
    color='llm:N',
    xOffset='llm:N',
    column='prompt:N',
    tooltip=['retriever', 'llm', 'prompt', 'hallucination']
)
st.altair_chart(chart, use_container_width=True)

# st.write(summary)
# cf_recommender_app.py
"""
Collaborative-Filtering Recommender Playground  👥🔀📊
────────────────────────────────────────────────────────────────────────────
Upload a user-item-rating CSV and explore:
1. Memory-based CF: cosine-similarity on the user–item matrix.
2. Top-N recommendations per user.
3. Interactive similarity heatmap for users or items.
4. LLM “Explain why” for any recommendation via OpenRouter.

*Proof-of-concept only.* For production recommender systems, contact → drtomharty.com/bio
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ───────────────────────────────── OpenRouter Chat Helper ─────────────────────────────────
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"

def openrouter_chat(messages, model=DEFAULT_MODEL, temperature=0.7):
    if not API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in Streamlit secrets or env")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://portfolio.example",
        "X-Title": "CF-Recommender-Explain",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ───────────────────────────── Session State Init ─────────────────────────────
st.session_state.setdefault("cf_matrix", None)
st.session_state.setdefault("user_sim", None)
st.session_state.setdefault("item_sim", None)
st.session_state.setdefault("recs_df", None)
st.session_state.setdefault("chat_history", [
    {"role":"system","content":"You are a helpful assistant explaining collaborative-filtering recommendations."}
])

# ───────────────────────────────────────── UI ─────────────────────────────────────
st.set_page_config(page_title="CF Recommender Playground", layout="wide")
st.title("👥🔀 Collaborative-Filtering Recommender Playground")

st.info(
    "🔔 **Demo Notice**  \n"
    "This is a lightweight proof-of-concept. For full production recommender systems, "
    "[contact me](https://drtomharty.com/bio).",
    icon="💡"
)

# 1) Upload CSV
uploaded = st.file_uploader("📂 Upload user–item–rating CSV", type="csv")
if not uploaded:
    st.stop()
df = pd.read_csv(uploaded)
st.subheader("Data preview")
st.dataframe(df.head())

# 2) Select columns
user_col = st.selectbox("User ID column", df.columns, key="uc")
item_col = st.selectbox("Item ID column", df.columns, key="ic")
rating_col = st.selectbox("Rating column", df.select_dtypes(include="number").columns, key="rc")

# 3) Build CF matrices
if st.button("🚀 Build recommender"):
    st.session_state.cf_matrix = df.pivot_table(
        index=user_col, columns=item_col, values=rating_col, fill_value=0
    )
    # similarity matrices
    st.session_state.user_sim = pd.DataFrame(
        cosine_similarity(st.session_state.cf_matrix),
        index=st.session_state.cf_matrix.index,
        columns=st.session_state.cf_matrix.index
    )
    st.session_state.item_sim = pd.DataFrame(
        cosine_similarity(st.session_state.cf_matrix.T),
        index=st.session_state.cf_matrix.columns,
        columns=st.session_state.cf_matrix.columns
    )
    st.success("Recommender built!")

if st.session_state.cf_matrix is None:
    st.stop()

cf = st.session_state.cf_matrix

# 4) Recommend for a selected user
st.subheader("Top-N Recommendations")
selected_user = st.selectbox("Choose user", cf.index)
n_rec = st.slider("Number of recommendations", 1, 20, 5)

# Compute user-based CF predictions
user_vec = cf.loc[selected_user].values
# weighted sum of other users' ratings
weights = st.session_state.user_sim[selected_user].values
pred = weights.dot(cf.values) / np.clip(weights.sum(), 1e-8, None)
pred_series = pd.Series(pred, index=cf.columns)
# remove items already rated
pred_series[cf.loc[selected_user] > 0] = -np.inf
top_items = pred_series.nlargest(n_rec).rename("predicted_score")
recs_df = top_items.reset_index().rename(columns={"index":item_col})
st.session_state.recs_df = recs_df
st.table(recs_df)

# Download recommendations
csv_bytes = recs_df.to_csv(index=False).encode()
st.download_button("⬇️ Download recommendations (CSV)", csv_bytes,
                   f"recs_{selected_user}.csv", "text/csv")

# 5) Similarity heatmap
st.subheader("Similarity Heatmap")
sim_choice = st.radio("Heatmap type", ["User–User", "Item–Item"])
if sim_choice == "User–User":
    fig = px.imshow(st.session_state.user_sim, labels={'x':'User','y':'User','color':'Similarity'},
                    title="User–User Cosine Similarity")
else:
    fig = px.imshow(st.session_state.item_sim, labels={'x':'Item','y':'Item','color':'Similarity'},
                    title="Item–Item Cosine Similarity")
st.plotly_chart(fig, use_container_width=True)

# 6) Chat explanation
st.subheader("💬 Explain a recommendation")
for msg in st.session_state.chat_history[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Ask why an item was recommended…")
if user_input:
    st.session_state.chat_history.append({"role":"user","content":user_input})
    st.chat_message("user").markdown(user_input)

    # inject context: show top recommendations for the user
    context = recs_df.to_markdown(index=False)
    st.session_state.chat_history.append({
        "role":"system",
        "content": f"Here are the top {n_rec} recommendations for user {selected_user}:\n\n{context}"
    })

    with st.spinner("Generating explanation…"):
        reply = openrouter_chat(st.session_state.chat_history)
    st.session_state.chat_history.append({"role":"assistant","content":reply})
    st.chat_message("assistant").markdown(reply)

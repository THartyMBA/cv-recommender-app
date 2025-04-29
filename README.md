# cv-recommender-app

👥🔀 Collaborative-Filtering Recommender Playground
Upload a user–item–rating CSV, explore memory-based collaborative filtering, get top-N recommendations, visualize similarity heatmaps, and even ask an LLM to explain why an item was suggested.

Proof-of-concept only—no production-grade data pipelines or scaling.
For enterprise recommender systems, contact me.

🔍 What it does
Load a CSV with columns: UserID, ItemID, Rating.

Build the user–item matrix and compute cosine similarities:

User–User similarity

Item–Item similarity

Recommend top-N items for a selected user via weighted neighbor ratings.

Visualize similarity matrices as interactive heatmaps.

Chat with an OpenRouter LLM to explain any recommendation, with context injected automatically.

Download recommendations as CSV for further analysis.

✨ Key Features
Memory-based CF: straightforward, interpretable collaborative filtering

Top-N recommendations: filtered to unseen items

Interactive heatmaps: Plotly-powered user–user and item–item views

LLM explanations: OpenRouter chat returns human-readable “why” for any recommendation

Single-file app: all logic in cf_recommender_app.py

Downloadable outputs: CSV export of your top-N recs

🔑 Add your OpenRouter API Key
Streamlit Community Cloud
Deploy the repo → ⋯ → Edit secrets

Add:

toml
Copy
Edit
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
Local development
Create ~/.streamlit/secrets.toml:

toml
Copy
Edit
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
—or—

bash
Copy
Edit
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
🚀 Quick Start (Local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/cf-recommender-playground.git
cd cf-recommender-playground
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run cf_recommender_app.py
Open http://localhost:8501.

Upload your user–item–rating CSV.

Build the recommender → explore heatmaps → ask for explanations.

☁️ Deploy on Streamlit Community Cloud
Push this repo (public or private) under THartyMBA to GitHub.

Visit streamlit.io/cloud → New app → select repo/branch → Deploy.

Add your OPENROUTER_API_KEY in Secrets—no other config needed.

🛠️ Requirements
shell
Copy
Edit
streamlit>=1.32
pandas
numpy
scikit-learn
plotly
requests
🗂️ Repo Structure
vbnet
Copy
Edit
cf-recommender-playground/
├─ cf_recommender_app.py    ← single-file Streamlit app  
├─ requirements.txt  
└─ README.md               ← you’re reading it  
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – rapid interactive Python UIs

scikit-learn – cosine similarity & CF basics

Plotly – interactive heatmaps

OpenRouter – LLM explanations

Recommend, visualize, and explain—your CF playground in a single tab! 🚀

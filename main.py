import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("tourism-data.csv")

def parse_keywords(x):
    if pd.isna(x) or x == "":
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []

df["clean_keywords"] = df["clean_keywords"].apply(parse_keywords)

sentences = df["clean_keywords"].tolist()

# =========================
# TRAIN WORD2VEC
# =========================
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4,
    seed=42
)

def normalize_text(text):
    return re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]', '', str(text).lower()).strip()

def w2v_embedding(keywords, model):
    vectors = []
    for k in keywords:
        k = normalize_text(k)
        if k in model.wv:
            vectors.append(model.wv[k])
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# =========================
# PRE-COMPUTE EMBEDDING
# =========================
df["w2v_embedding"] = df["clean_keywords"].apply(
    lambda x: w2v_embedding(x, w2v_model)
)

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend_w2v(user_keywords, top_k=20, province=None):
    data = df.copy()

    # 🔹 LỌC THEO TỈNH
    if province:
        province = normalize_text(province)
        data = data[
            data["province"].apply(lambda x: normalize_text(x)) == province
        ]

        if data.empty:
            return pd.DataFrame(columns=[
                "name", "province", "activities", "similarity", "clean_keywords"
            ])

    user_emb = w2v_embedding(user_keywords, w2v_model)

    sims = cosine_similarity(
        user_emb.reshape(1, -1),
        np.vstack(data["w2v_embedding"])
    )[0]

    data = data.copy()
    data["similarity"] = sims

    return data.sort_values(
        "similarity", ascending=False
    ).head(top_k)[
        ["name", "province", "activities", "similarity", "clean_keywords"]
    ]

# =========================
# FLASK API
# =========================
app = Flask(__name__)

@app.route("/recommend_w2v", methods=["POST"])
def recommend_w2v_api():
    data = request.json
    if not data or "keywords" not in data:
        return jsonify({"error": "Cần trường 'keywords'"}), 400

    user_keywords = data["keywords"]
    province = data.get("province")  # optional

    result_df = recommend_w2v(
        user_keywords=user_keywords,
        province=province
    )

    return jsonify(result_df.to_dict(orient="records"))

@app.route("/test_recommend_w2v", methods=["GET"])
def recommend_w2v_test():
    keywords = request.args.getlist("kw")
    province = request.args.get("province")

    if not keywords:
        return jsonify({"error": "Cần ít nhất 1 keyword ?kw=..."}), 400

    result_df = recommend_w2v(
        user_keywords=keywords,
        province=province
    )

    return jsonify(result_df.to_dict(orient="records"))

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)

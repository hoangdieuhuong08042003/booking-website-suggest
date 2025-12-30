import os
import numpy as np
import pandas as pd
import ast
import re
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# =========================
# CONFIG & PATHS
# =========================
DATA_PATH = "tourism-data.csv"
MODEL_PATH = "models/word2vec_tourism.model"
EMBEDDING_PATH = "models/place_embeddings.npy"
INDEX_PATH = "models/faiss_index.index"

os.makedirs("models", exist_ok=True)

OPENWEATHER_API_KEY = "cfdf512182b6e4a04dd23de34d184235"

# =========================
# LOAD & CLEAN DATA
# =========================
df = pd.read_csv(DATA_PATH)

def parse_keywords(x):
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return []

def normalize_text(text):
    return re.sub(r"[^a-zA-ZÀ-ỹ0-9\s]", "", str(text).lower()).strip()

df["clean_keywords"] = df["clean_keywords"].apply(parse_keywords)
df["activities"] = df["activities"].str.lower()
df["province_norm"] = df["province"].apply(normalize_text)
df = df.reset_index(drop=True)

# =========================
# WORD2VEC + EMBEDDINGS + FAISS
# =========================
sentences = df["clean_keywords"].tolist()

if os.path.exists(MODEL_PATH) and os.path.exists(EMBEDDING_PATH) and os.path.exists(INDEX_PATH):
    model = Word2Vec.load(MODEL_PATH)
    embeddings_matrix = np.load(EMBEDDING_PATH)
    index = faiss.read_index(INDEX_PATH)
    print("Loaded pre-trained model, embeddings and FAISS index.")
else:
    print("Training Word2Vec and building embeddings + FAISS index...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=5,
        min_count=1,
        workers=4,
        epochs=10
    )
    model.save(MODEL_PATH)

    def get_average_embedding(keywords):
        vectors = []
        for k in keywords:
            k_norm = normalize_text(k)
            if k_norm in model.wv:
                vectors.append(model.wv[k_norm])
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    embeddings_matrix = np.array([get_average_embedding(kws) for kws in df["clean_keywords"]])
    np.save(EMBEDDING_PATH, embeddings_matrix)

    # Normalize cho cosine similarity
    faiss.normalize_L2(embeddings_matrix)
    index = faiss.IndexFlatIP(50)
    index.add(embeddings_matrix)
    faiss.write_index(index, INDEX_PATH)
    print("Saved model, embeddings and FAISS index.")

# =========================
# TIME SLOT DETECTION (Similarity-based)
# =========================
SLOT_REPRESENTATIVES = {
    "sáng": ["lịch sử", "di tích", "bảo tàng", "chùa", "tham quan", "di sản", "tâm linh", "kiến trúc"],
    "trưa": ["ẩm thực", "ăn uống", "chợ", "shopping", "nhà hàng", "cà phê"],
    "chiều": ["thiên nhiên", "cảnh đẹp", "công viên", "hồ", "suối", "nghỉ dưỡng", "ngắm cảnh", "check-in"],
    "tối": ["giải trí", "phố đi bộ", "nightlife", "biểu diễn", "quảng trường", "mua sắm", "náo nhiệt"]
}

slot_vectors = {}
for slot, words in SLOT_REPRESENTATIVES.items():
    vecs = [model.wv[normalize_text(w)] for w in words if normalize_text(w) in model.wv]
    slot_vectors[slot] = np.mean(vecs, axis=0) if vecs else np.zeros(50)

def detect_best_time_slot(place_keywords):
    if not place_keywords:
        return "chiều"
    place_vec = np.mean([model.wv[normalize_text(k)] for k in place_keywords if normalize_text(k) in model.wv], axis=0)
    if len(place_vec) == 0:
        return "chiều"
    place_vec = place_vec.reshape(1, -1)
    faiss.normalize_L2(place_vec)
    scores = {slot: cosine_similarity(place_vec, slot_vec.reshape(1, -1))[0][0]
              for slot, slot_vec in slot_vectors.items()}
    return max(scores, key=scores.get)

df["time_slot"] = df["clean_keywords"].apply(detect_best_time_slot)

# =========================
# WEATHER CACHE
# =========================
weather_cache = {}  # key: province_norm -> (timestamp, forecast)

def get_weather_forecast(province, days=3):
    province_norm = normalize_text(province)
    now = datetime.now()

    if province_norm in weather_cache:
        cached_time, forecast = weather_cache[province_norm]
        if now - cached_time < timedelta(hours=1):
            return forecast[:days]

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": province,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "vi"
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        if res.get("cod") != "200":
            return []

        daily = {}
        for item in res["list"]:
            date = item["dt_txt"].split(" ")[0]
            if date not in daily:
                daily[date] = item

        forecast = []
        for date, item in list(sorted(daily.items()))[:days]:
            forecast.append({
                "date": date,
                "main": item["weather"][0]["main"].lower(),
                "description": item["weather"][0]["description"],
                "temp": round(item["main"]["temp"], 1)
            })

        weather_cache[province_norm] = (now, forecast)
        return forecast[:days]
    except:
        return []

def weather_to_activity(main):
    return "indoor" if main in ["rain", "drizzle", "thunderstorm", "snow"] else "outdoor"

# =========================
# SELECT PLACE (FAISS + Filter)
# =========================
def get_user_embedding(keywords):
    vecs = [model.wv[normalize_text(k)] for k in keywords if normalize_text(k) in model.wv]
    if not vecs:
        return np.zeros(50)
    vec = np.mean(vecs, axis=0).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

def select_place(province, user_keywords, slot, activity_type, used_places):
    province_norm = normalize_text(province)

    mask = (
        (df["province_norm"] == province_norm) &
        (df["time_slot"] == slot) &
        (df["activities"] == activity_type) &
        (~df["name"].isin(used_places))
    )
    candidates_idx = df[mask].index.values

    if len(candidates_idx) == 0:
        # Fallback: bỏ điều kiện activity hoặc time_slot
        mask_fallback = (
            (df["province_norm"] == province_norm) &
            (~df["name"].isin(used_places))
        )
        candidates_idx = df[mask_fallback].index.values
        if len(candidates_idx) == 0:
            return None

    user_vec = get_user_embedding(user_keywords)

    # Tìm top 5 gần nhất trong candidates
    D, I = index.search(user_vec, min(5, len(candidates_idx)))
    valid_I = [i for i in I[0] if i in candidates_idx and i != -1]

    if not valid_I:
        # Chọn rating cao nhất làm fallback cuối
        best_idx = df.loc[candidates_idx].sort_values("rating", ascending=False).index[0]
    else:
        best_idx = valid_I[0]

    row = df.loc[best_idx]

    return {
        "tên": row["name"],
        "tỉnh": row["province"],
        "mô_tả": row["description"],
        "đánh_giá": float(row["rating"]),
        "hình_ảnh": row["image"],
        "hoạt_động": "Trong nhà" if row["activities"] == "indoor" else "Ngoài trời"
    }

# =========================
# BUILD ITINERARY
# =========================
def build_itinerary(province, keywords, days):
    forecast = get_weather_forecast(province, days)
    if not forecast:
        return []  # hoặc trả lỗi

    itinerary = []
    used_places = set()

    normalized_keywords = [normalize_text(k) for k in keywords]

    for i, day in enumerate(forecast[:days]):
        activity_type = weather_to_activity(day["main"])
        schedule = {}

        for slot in ["sáng", "trưa", "chiều", "tối"]:
            place = select_place(
                province=province,
                user_keywords=normalized_keywords,
                slot=slot,
                activity_type=activity_type,
                used_places=used_places
            )
            if place:
                used_places.add(place["tên"])
            schedule[slot] = place

        itinerary.append({
            "ngày": i + 1,
            "date": day["date"],
            "thời_tiết": day,
            "loại_ngày": "Trong nhà" if activity_type == "indoor" else "Ngoài trời",
            "lịch_trình": schedule
        })

    return itinerary

# =========================
# FLASK API
# =========================
app = Flask(__name__)
app.json.ensure_ascii = False
app.json.indent = 4
@app.route("/itinerary", methods=["POST"])
def itinerary_api():
    data = request.json
    required = ["province", "keywords", "days"]
    if not data or not all(k in data for k in required):
        return jsonify({"error": f"Cần các trường: {required}"}), 400

    try:
        days = min(3, max(1, int(data["days"])))
        keywords = data["keywords"]
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]

        result = build_itinerary(data["province"], keywords, days)

        return jsonify({
            "tỉnh_thành": data["province"],
            "số_ngày": days,
            "từ_khóa": keywords,
            "lịch_trình": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/itinerary/test", methods=["GET"])
def itinerary_test():
    province = request.args.get("province")
    days = request.args.get("days", default=2, type=int)
    keywords = request.args.getlist("kw")

    if not province or not keywords:
        return jsonify({"error": "Cần province và ít nhất 1 kw"}), 400

    itinerary = build_itinerary(province, keywords, min(3, days))

    return jsonify({
        "tỉnh_thành": province,
        "số_ngày": days,
        "từ_khóa": keywords,
        "lịch_trình": itinerary
    })
# =========================
# NEW: RECOMMEND TOP PLACES (NO ITINERARY)
# =========================

def recommend_places(province, keywords, num_places=15, consider_weather=True, days_for_weather=2):
    """
    Trả về tối đa 15 địa điểm, ưu tiên cao độ phù hợp từ khóa
    Returns: (list_of_places, weather_summary)
    """
    province_norm = normalize_text(province)
    normalized_keywords = [normalize_text(k) for k in keywords]

    # Lọc địa điểm trong tỉnh
    province_mask = (df["province_norm"] == province_norm)
    candidates = df[province_mask].copy()

    if len(candidates) == 0:
        return [], "Không tìm thấy địa điểm nào cho tỉnh/thành phố này."

    # Tính similarity từ khóa
    user_vec = get_user_embedding(normalized_keywords)

    if np.all(user_vec == 0):
        candidates["similarity"] = 0.0
    else:
        candidate_indices = candidates.index.values
        D, I = index.search(user_vec, len(candidate_indices))
        similarity_scores = dict(zip(I[0], D[0]))
        candidates["similarity"] = candidates.index.map(similarity_scores.get).fillna(0.0)

    # Ưu tiên mạnh similarity: loại bỏ những cái quá thấp (trừ khi tất cả đều thấp)
    min_threshold = 0.1
    if candidates["similarity"].max() < min_threshold:
        min_threshold = 0.0  # fallback nếu từ khóa không match gì cả

    candidates = candidates[candidates["similarity"] >= min_threshold]

    if len(candidates) == 0:
        return [], "Không tìm thấy địa điểm phù hợp với từ khóa."

    # Xử lý thời tiết
    weather_summary = "Không có dữ liệu thời tiết."
    if consider_weather:
        forecast = get_weather_forecast(province, days_for_weather)
        if forecast:
            today = forecast[0]
            temp = today["temp"]
            desc = today["description"].capitalize()
            main = today["main"].capitalize()

            rainy_days = sum(1 for day in forecast if day["main"] in ["rain", "drizzle", "thunderstorm"])
            if rainy_days >= len(forecast) // 2:
                advice = "Có mưa nhiều ngày, nên ưu tiên hoạt động trong nhà."
            else:
                advice = "Thời tiết khô ráo, rất phù hợp cho hoạt động ngoài trời."

            weather_summary = f"Hiện tại: {main} – {desc}, {temp}°C. {advice}"

            # Ưu tiên indoor/outdoor
            if rainy_days >= len(forecast) // 2:
                candidates["weather_match"] = candidates["activities"].apply(lambda x: 2.0 if x == "indoor" else 0.5)
            else:
                candidates["weather_match"] = candidates["activities"].apply(lambda x: 2.0 if x == "outdoor" else 0.8)
        else:
            candidates["weather_match"] = 1.0
            weather_summary = "Không lấy được dự báo thời tiết."
    else:
        candidates["weather_match"] = 1.0
        weather_summary = "Không xem xét thời tiết."

    # Tính điểm: ưu tiên mạnh similarity
    candidates["score"] = (
        candidates["similarity"] * 0.8 +      # 80% từ khóa
        candidates["weather_match"] * 0.1 +  # 10% thời tiết
        (candidates["rating"] / 5.0) * 0.1    # 10% rating
    )

    # Sắp xếp và lấy tối đa 15
    candidates = candidates.sort_values("score", ascending=False)
    num_places = min(15, len(candidates))  # cứng tối đa 15
    top_places = candidates.head(num_places)

    results = []
    for _, row in top_places.iterrows():
        sim_score = float(row["similarity"])
        results.append({
            "tên": row["name"],
            "tỉnh": row["province"],
            "mô_tả": row["description"],
            "đánh_giá": float(row["rating"]),
            "hình_ảnh": row["image"],
            "hoạt_động": "Trong nhà" if row["activities"] == "indoor" else "Ngoài trời",
            "độ_phù_hợp_từ_khóa": round(sim_score, 3)
        })

    return results, weather_summary

# =========================
# NEW API ENDPOINT: /recommend
# =========================

@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.json
    required = ["province", "keywords"]
    if not data or not all(k in data for k in required):
        return jsonify({"error": f"Cần các trường bắt buộc: {required}"}), 400

    try:
        province = data["province"]
        keywords = data["keywords"]
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]

        days = int(data.get("days", 2))
        num_places = int(data.get("num_places", 12))
        consider_weather = data.get("consider_weather", True)

        places, weather_summary = recommend_places(
            province=province,
            keywords=keywords,
            num_places=num_places,
            consider_weather=consider_weather,
            days_for_weather=days
        )

        return jsonify({
            "tỉnh_thành": province,
            "từ_khóa": keywords,
            "số_lượng_gợi_ý": len(places),
            "xem_xét_thời_tiết": consider_weather,
            "thông_tin_thời_tiết": weather_summary,
            "danh_sách_địa_điểm": places
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Optional: Test endpoint GET
@app.route("/recommend/test", methods=["GET"])
def recommend_test():
    province = request.args.get("province")
    keywords = request.args.getlist("kw")
    days = request.args.get("days", default=2, type=int)
    num = request.args.get("num", default=12, type=int)
    consider = request.args.get("weather", default="true").lower() == "true"

    if not province or not keywords:
        return jsonify({"error": "Cần province và ít nhất 1 kw"}), 400

    places, weather_summary = recommend_places(
        province=province,
        keywords=keywords,
        num_places=num,
        consider_weather=consider,
        days_for_weather=days
    )

    return jsonify({
        "tỉnh_thành": province,
        "từ_khóa": keywords,
        "số_lượng_gợi_ý": len(places),
        "xem_xét_thời_tiết": consider,
        "thông_tin_thời_tiết": weather_summary,
        "danh_sách_địa_điểm": places
    })
if __name__ == "__main__":
    app.run(debug=True)
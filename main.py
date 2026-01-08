import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
import random
from datetime import datetime, timedelta
import os
import pickle

# =========================
# INTENT GROUPS
# =========================
THEME_GROUPS = {
    "nature": ["thiên nhiên", "sinh thái", "nguyên sơ"],
    "scenery": ["cảnh đẹp", "ngắm cảnh", "thơ mộng"],
    "culture": ["văn hóa", "truyền thống", "lễ hội"],
    "history": ["lịch sử", "di sản", "di tích"],
    "architecture": ["kiến trúc", "kiến trúc tôn giáo"],
    "spiritual": ["tâm linh", "đi lễ chùa"]
}

ACTIVITY_GROUPS = {
    "food": ["ẩm thực", "ăn uống", "nhà hàng"],
    "photo": ["chụp ảnh", "check-in", "quán cà phê"],
    "visit": ["tham quan", "dạo phố", "tham quan di tích", "tham quan bảo tàng"],
    "trekking": ["trekking", "leo núi", "camping", "picnic"],
    "shopping": ["mua sắm", "shopping", "chợ"]
}
def normalize_province(province):
    """Chuẩn hóa tên tỉnh/thành phố về dạng chữ thường, loại bỏ ký tự đặc biệt, tiền tố và khoảng trắng thừa."""
    if not province:
        return ""
    text = str(province).lower()
    text = re.sub(r'^(thành phố|tp\.?|tỉnh|quận|huyện|thị xã)\s*', '', text)
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# =========================
# INTENT SPLITTING
# =========================
def split_keywords_by_intent(user_keywords):
    intents = {}
    for group in (THEME_GROUPS, ACTIVITY_GROUPS):
        for name, kws in group.items():
            matched = [k for k in user_keywords if k in kws]
            if matched:
                intents[name] = matched
    return intents

# =========================
# SINGLE INTENT RECOMMENDER
# =========================
def recommend_single_intent(data, keywords, top_k=10):
    tokens = tokenize_text(" ".join(keywords))
    emb = w2v_embedding(tokens, w2v_model)
    if np.all(emb == 0):
        return pd.DataFrame()
    sims = cosine_similarity(
        emb.reshape(1, -1),
        np.vstack(data["w2v_embedding"])
    )[0]
    result = data.copy()
    result["similarity"] = sims
    return result.nlargest(top_k, "similarity")[
        ["name", "province", "activities", "similarity", "description", "image", "rating"]
    ]

# =========================
# MULTI-INTENT RECOMMENDER (REPLACEMENT FOR recommend_w2v)
# =========================
def recommend_w2v_multi_list(user_keywords, province=None, top_k=10):
    intents = split_keywords_by_intent(user_keywords)
    province_norm = normalize_province(province) if province else None
    if province_norm:
        data = df[
            df["province"].fillna("").str.lower() == province_norm
        ].copy()
    else:
        data = df.copy()
    if data.empty:
        return {}
    results = {}
    for intent, kws in intents.items():
        rec = recommend_single_intent(data, kws, top_k)
        if not rec.empty:
            results[intent] = rec.to_dict(orient="records")
    return results
# =========================
# WEATHER API (THÊM CACHE ĐƠN GIẢN TRONG 1 REQUEST)
# =========================
OPENWEATHER_API_KEY = "cfdf512182b6e4a04dd23de34d184235"
_weather_cache = {}

def get_weather_status(province):
    return get_weather_status_with_date(province)

def get_weather_status_with_date(province, date_str=None):
    key = (province, date_str)
    if key in _weather_cache:
        return _weather_cache[key]
    province_norm = normalize_province(province)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={province_norm},VN&appid={OPENWEATHER_API_KEY}&lang=vi"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        weather = data.get("weather", [{}])
        if weather and ("rain" in weather[0].get("main", "").lower() or "rain" in data):
            status = "rain"
        elif any(w.get("main", "").lower() == "rain" for w in weather):
            status = "rain"
        else:
            status = "clear"
    except Exception:
        status = "clear"
    _weather_cache[key] = status
    return status

def normalize_text(text):
    return re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]', '', str(text).lower()).strip()

def tokenize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().split()

def w2v_embedding(tokens, model):
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


# =========================
# LOAD DATA & WORD2VEC MODEL (TỐI ƯU KHỞI ĐỘNG)
# =========================
DATA_CSV = "data2.csv"
EMBED_CSV = "data2_with_embed.csv"
W2V_MODEL_PATH = "w2v_model.pkl"

def load_or_train_w2v(sentences):
    if os.path.exists(W2V_MODEL_PATH):
        with open(W2V_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model = Word2Vec(
            sentences=sentences,
            vector_size=50,
            window=5,
            min_count=1,
            workers=1,  # giảm worker để tránh quá tải CPU trên Render
            seed=42
        )
        with open(W2V_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
    return model

def load_data_and_embeddings():
    if os.path.exists(EMBED_CSV):
        df = pd.read_csv(EMBED_CSV)
        df["w2v_embedding"] = df["w2v_embedding"].apply(lambda x: np.array(eval(x)))
        df["w2v_tokens"] = df["w2v_tokens"].apply(eval)
    else:
        df = pd.read_csv(DATA_CSV)
        df["w2v_tokens"] = df["text_for_model"].apply(tokenize_text)
        sentences = df["w2v_tokens"].tolist()
        w2v_model = load_or_train_w2v(sentences)
        df["w2v_embedding"] = df["w2v_tokens"].apply(lambda x: w2v_embedding(x, w2v_model))
        # Lưu lại embedding để lần sau load nhanh
        df_save = df.copy()
        df_save["w2v_embedding"] = df_save["w2v_embedding"].apply(lambda x: x.tolist())
        df_save["w2v_tokens"] = df_save["w2v_tokens"].apply(lambda x: list(x))
        df_save.to_csv(EMBED_CSV, index=False)
    return df

# Khởi tạo dữ liệu và model chỉ 1 lần
df = load_data_and_embeddings()
sentences = df["w2v_tokens"].tolist()
w2v_model = load_or_train_w2v(sentences)

# =========================
# RECOMMEND_W2V2 FUNCTION (TỐI ƯU FILTER)
# =========================
def recommend_w2v2(user_keywords, province=None, top_k=20, days=None):
    user_tokens = tokenize_text(" ".join(user_keywords))
    user_emb = w2v_embedding(user_tokens, w2v_model)

    province_norm = normalize_province(province) if province else None
    if province_norm:
        data = df[df["province"].str.lower() == province_norm]
    else:
        data = df

    if data.empty:
        return pd.DataFrame(columns=["name", "province", "activities", "similarity", "description", "image", "rating"])

    sims = cosine_similarity(
        user_emb.reshape(1, -1),
        np.vstack(data["w2v_embedding"])
    )[0]

    data = data.copy()
    data["similarity"] = sims

    # 3️⃣ Xử lý danh sách ngày (days) truyền vào dạng list các chuỗi ngày 'dd/mm/yyyy'
    weather_statuses = []
    weather_vi = []
    weather_dates = []
    if days is not None and isinstance(days, list) and len(days) > 0:
        date_list = days
    else:
        # Nếu không truyền vào thì mặc định là 3 ngày từ hôm nay
        today = datetime.now()
        date_list = [(today + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(3)]

    for date_str in date_list:
        status = get_weather_status_with_date(province, date_str)
        weather_statuses.append(status)
        weather_vi.append(weather_vi_message(province, status, date_str))
        weather_dates.append(date_str)

    # Hàm này vẫn trả về DataFrame như cũ, nếu muốn trả thêm thông tin thời tiết thì có thể trả về tuple hoặc dict
    return data.sort_values(
        "similarity", ascending=False
    ).head(top_k)[
        ["name", "province", "activities", "similarity", "description", "image", "rating"]
    ]
# =========================
# RECOMMEND_W2V FUNCTION
# =========================
def recommend_w2v(user_keywords, province=None, top_k=20, date_str=None):
    user_tokens = tokenize_text(" ".join(user_keywords))
    user_emb = w2v_embedding(user_tokens, w2v_model)

    province_norm = normalize_province(province) if province else None
    if province_norm:
        data = df[df["province"].str.lower() == province_norm].copy()
    else:
        data = df.copy()

    # 1️⃣ Filter theo province (nếu có)
    if province:
        data = df[df["province"].str.lower() == province.lower()].copy()
    else:
        data = df.copy()

    if data.empty:
        return pd.DataFrame(columns=["name", "province", "activities", "similarity", "description", "image", "rating"])

    # 2️⃣ Similarity
    sims = cosine_similarity(
        user_emb.reshape(1, -1),
        np.vstack(data["w2v_embedding"])
    )[0]

    data["similarity"] = sims

    # 3️⃣ Lọc theo thời tiết với ngày truyền vào
    weather_status = get_weather_status_with_date(province, date_str) if province else "clear"
    if weather_status == "rain":
        # Lấy indoor
        data = data[data["activities"].str.lower() == "indoor"]
    elif weather_status == "clear":
        # Lấy outdoor
        data = data[data["activities"].str.lower() == "outdoor"]
    # Nếu không có kết quả sau khi lọc thì trả về top_k không lọc
    if data.empty:
        data = df.copy()
        if province:
            data = data[data["province"].str.lower() == province.lower()]
        sims = cosine_similarity(
            user_emb.reshape(1, -1),
            np.vstack(data["w2v_embedding"])
        )[0]
        data["similarity"] = sims
        return data.sort_values(
            "similarity", ascending=False
        ).head(top_k)[
            ["name", "province", "activities", "similarity", "description", "image", "rating"]
        ]

    return data.sort_values(
        "similarity", ascending=False
    ).head(top_k)[
        ["name", "province", "activities", "similarity", "description", "image", "rating"]
    ]
# =========================
# PRE-COMPUTE 
# =========================
df["w2v_embedding"] = df["w2v_tokens"].apply(lambda x: w2v_embedding(x, w2v_model))

# =========================
# RECOMMEND FUNCTION
# =========================
def weather_vi_message(province, weather_status, date_str=None):
    province_display = province if province else "(không xác định)"
    date_display = f" vào ngày {date_str}" if date_str else ""
    if weather_status == "rain":
        msg = (
            f"Thời tiết{date_display} tại {province_display} dự báo có mưa. "
            "Bạn nên ưu tiên các địa điểm vui chơi trong nhà (indoor) để chuyến đi luôn thoải mái và an toàn. "
            "Đừng quên mang theo áo mưa hoặc ô nhé!"
        )
    elif weather_status == "clear":
        msg = (
            f"Thời tiết{date_display} tại {province_display} rất đẹp, không mưa. "
            "Đây là thời điểm lý tưởng để khám phá các địa điểm ngoài trời (outdoor), tận hưởng không khí trong lành và lưu lại những khoảnh khắc tuyệt vời!"
        )
    else:
        msg = (
            f"Không xác định được thời tiết{date_display} tại {province_display}. "
            "Bạn nên kiểm tra lại tên tỉnh hoặc thử lại sau để có thông tin dự báo chính xác hơn."
        )
    return msg



def remove_ndarray_fields(obj):
    """Loại bỏ các trường kiểu ndarray (ví dụ: w2v_embedding) khỏi dict và chuẩn hóa output."""
    if isinstance(obj, dict):
        # Chuyển đổi activities sang tiếng Việt
        activities_vi = ""
        if "activities" in obj and pd.notna(obj["activities"]):
            if str(obj["activities"]).lower() == "outdoor":
                activities_vi = "ngoài trời"
            elif str(obj["activities"]).lower() == "indoor":
                activities_vi = "trong nhà"
        # Always include image and description fields
        image = obj.get("image", "")
        if pd.isna(image):
            image = ""
        description = obj.get("description", "")
        if pd.isna(description):
            description = ""
        return {
            "name": obj.get("name", ""),
            "province": obj.get("province", ""),
            "description": description,
            "image": image,
            "rating": obj.get("rating", 4.5),
            "hoạt_động": activities_vi
        }
    return obj

def generate_itinerary(sim_df, user_keywords, province, days=None):
    # days: list các ngày dạng 'dd/mm/yyyy'
    sim_places = sim_df.copy().reset_index(drop=True)
    lunch_keywords = ['ẩm thực', 'ăn uống', 'nhà hàng']
    lunch_places = recommend_w2v2(lunch_keywords, province=province, top_k=20).reset_index(drop=True)

    evening_keywords = ['quán cà phê', 'chụp ảnh']
    evening_places = recommend_w2v2(evening_keywords, province=province, top_k=20).reset_index(drop=True)

    dinner_keywords = ['ẩm thực', 'ăn uống', 'nhà hàng']
    dinner_places = recommend_w2v2(dinner_keywords, province=province, top_k=20).reset_index(drop=True)
    used_dinner = set()

    itinerary = []
    used_idx = set()
    used_lunch = set()
    used_evening = set()

    # Nếu không truyền days thì mặc định là 1 ngày hôm nay
    if days is None or not isinstance(days, list) or len(days) == 0:
        today = datetime.now()
        days = [(today + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(1)]

    for day_idx, date_str in enumerate(days):
        day_plan = {}
        # Lấy thời tiết cho ngày đó
        weather_status = get_weather_status_with_date(province, date_str)
        weather_message = weather_vi_message(province, weather_status, date_str)
        day_plan['thời tiết'] = weather_message
        # Thêm trường weather_forecast cho từng ngày
        weather_forecast = weather_message

        # Sáng
        morning_idx = None
        for i, row in sim_places.iterrows():
            if i not in used_idx:
                morning_idx = i
                used_idx.add(i)
                break
        if morning_idx is not None:
            day_plan['sáng'] = remove_ndarray_fields(sim_places.loc[morning_idx].to_dict())
        else:
            day_plan['sáng'] = {}

        # Trưa: random 1 kết quả phù hợp, không trùng các ngày trước
        lunch_row = {}
        available_lunch_idx = [i for i in lunch_places.index if i not in used_lunch]
        if available_lunch_idx:
            idx = random.choice(available_lunch_idx)
            used_lunch.add(idx)
            lunch_row = lunch_places.loc[idx].to_dict()
            day_plan['trưa'] = remove_ndarray_fields(lunch_row)
        else:
            day_plan['trưa'] = {}

        # Chiều
        afternoon_idx = None
        for i, row in sim_places.iterrows():
            if i not in used_idx:
                afternoon_idx = i
                used_idx.add(i)
                break
        if afternoon_idx is not None:
            day_plan['chiều'] = remove_ndarray_fields(sim_places.loc[afternoon_idx].to_dict())
        else:
            day_plan['chiều'] = {}

        # Tối: thêm địa điểm ăn uống trước khi đi cf
        evening_row = {}
        dinner_row = {}
        available_dinner_idx = [i for i in dinner_places.index if i not in used_dinner]
        if available_dinner_idx:
            idx_dinner = random.choice(available_dinner_idx)
            used_dinner.add(idx_dinner)
            dinner_row = dinner_places.loc[idx_dinner].to_dict()
            day_plan['tối_ăn_uống'] = remove_ndarray_fields(dinner_row)
        else:
            day_plan['tối_ăn_uống'] = {}

        available_evening_idx = [i for i in evening_places.index if i not in used_evening]
        if available_evening_idx:
            idx_evening = random.choice(available_evening_idx)
            used_evening.add(idx_evening)
            evening_row = evening_places.loc[idx_evening].to_dict()
            day_plan['tối'] = remove_ndarray_fields(evening_row)
        else:
            day_plan['tối'] = {}

        # Sắp xếp thứ tự các trường theo yêu cầu
        ordered_day_plan = {
            "thời tiết": day_plan.get("thời tiết", {}),
            "sáng": day_plan.get("sáng", {}),
            "trưa": day_plan.get("trưa", {}),
            "chiều": day_plan.get("chiều", {}),
            "tối_ăn_uống": day_plan.get("tối_ăn_uống", {}),
            "tối": day_plan.get("tối", {})
        }
        itinerary.append({
            "plan": ordered_day_plan,
            "date": date_str,
            "weather_forecast": weather_forecast
        })
    return itinerary

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
    top_k = data.get("top_k", 20)
    days = data.get("days")  # list các ngày dạng 'dd/mm/yyyy', optional

    # Bắt buộc phải truyền ngày
    if not days or not isinstance(days, list) or len(days) == 0:
        return jsonify({"error": "Cần chọn ngày (days)!"}), 400

    # Chuẩn hóa province trước khi truyền vào các hàm xử lý
    province_norm = normalize_province(province) if province else None

    # Use new multi-intent recommender
    results_by_intent = recommend_w2v_multi_list(user_keywords, province=province_norm, top_k=top_k)

    # Lấy thông tin thời tiết cho từng ngày (nếu có province)
    weather_forecast = None
    if province_norm:
        # Xác định danh sách ngày
        date_list = days
        weather_forecast = []
        for date_str in date_list:
            status = get_weather_status_with_date(province_norm, date_str)
            message = weather_vi_message(province_norm, status, date_str)
            weather_forecast.append({
                "date": date_str,
                "status": status,
                "message": message
            })

    # Backward compatibility: if no intent detected, fallback to old recommend_w2v
    if not results_by_intent:
        df_all = recommend_w2v(user_keywords, province=province_norm, top_k=top_k, date_str=None)
        results = df_all.to_dict(orient="records")
        return jsonify({
            "results": results,
            "weather_forecast": weather_forecast,
            "message": "Không phát hiện nhóm ý định, trả về gợi ý tổng hợp."
        })

    return jsonify({
        "results_by_intent": results_by_intent,
        "weather_forecast": weather_forecast,
        "message": "Dưới đây là gợi ý địa điểm du lịch theo từng nhóm ý định (intent group) phù hợp với từ khóa và tỉnh bạn chọn."
    })

@app.route("/test_recommend_w2v", methods=["GET"])
def recommend_w2v_test():
    keywords = request.args.getlist("kw")
    province = request.args.get("province")
    top_k = int(request.args.get("top_k", 20))
    days = request.args.getlist("days")  # ?days=01/06/2024&days=02/06/2024

    if not keywords:
        return jsonify({"error": "Cần ít nhất 1 keyword ?kw=..."}), 400

    results_by_intent = recommend_w2v_multi_list(keywords, province=province, top_k=top_k)

    # Lấy thông tin thời tiết cho từng ngày (nếu có province)
    weather_forecast = None
    if province:
        if days and isinstance(days, list) and len(days) > 0:
            date_list = days
        else:
            today = datetime.now()
            date_list = [(today + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(3)]
        weather_forecast = [
            weather_vi_message(province, get_weather_status_with_date(province, date_str), date_str)
            for date_str in date_list
        ]

    if not results_by_intent:
        df_all = recommend_w2v(keywords, province=province, top_k=top_k, date_str=None)
        results = df_all.to_dict(orient="records")
        return jsonify({
            "results": results,
            "weather_forecast": weather_forecast,
            "message": "Không phát hiện nhóm ý định, trả về gợi ý tổng hợp."
        })

    return jsonify({
        "results_by_intent": results_by_intent,
        "weather_forecast": weather_forecast,
        "message": "Dưới đây là gợi ý địa điểm du lịch theo từng nhóm ý định (intent group) phù hợp với từ khóa và tỉnh bạn chọn."
    })


# Helper function to build daily itinerary results
def build_daily_itinerary(user_keywords, province, days, top_k=20):
    province_norm = normalize_province(province) if province else None
    results_by_intent = recommend_w2v_multi_list(user_keywords, province=province_norm, top_k=top_k)
    # Fallback: if no intent, use old method
    if not results_by_intent:
        date_str = days[0] if days and isinstance(days, list) and len(days) > 0 else None
        result_df = recommend_w2v(user_keywords, province=province_norm, top_k=top_k, date_str=date_str)
        itinerary = generate_itinerary(result_df, user_keywords, province_norm, days)
        days_list = days if days and isinstance(days, list) and len(days) > 0 else [
            (datetime.now() + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(len(itinerary))
        ]
        daily_results = []
        for i, day in enumerate(itinerary):
            # day đã là dict {"plan":..., "date":..., "weather_forecast":...}
            daily_results.append({
                "date": day.get("date", days_list[i] if i < len(days_list) else ""),
                "plan": day.get("plan", {}),
                "weather_forecast": day.get("weather_forecast", "")
            })
        return daily_results

    # Chuẩn bị các intent group
    days_list = days if days and isinstance(days, list) and len(days) > 0 else [
        (datetime.now() + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(1)
    ]
    intent_keys = list(results_by_intent.keys())
    intent_dfs = {k: pd.DataFrame(v) for k, v in results_by_intent.items()}
    used_indices = {k: set() for k in intent_keys}
    n_days = len(days_list)
    n_intents = len(intent_keys)

    # Chuẩn bị group cho trưa/tối
    lunch_keywords = ['ẩm thực', 'ăn uống', 'nhà hàng']
    evening_keywords = ['quán cà phê', 'chụp ảnh']
    dinner_keywords = ['ẩm thực', 'ăn uống', 'nhà hàng']
    lunch_places = recommend_w2v2(lunch_keywords, province=province_norm, top_k=top_k).reset_index(drop=True)
    evening_places = recommend_w2v2(evening_keywords, province=province_norm, top_k=top_k).reset_index(drop=True)
    dinner_places = recommend_w2v2(dinner_keywords, province=province_norm, top_k=top_k).reset_index(drop=True)
    used_lunch = set()
    used_evening = set()
    used_dinner = set()

    # Đảm bảo không trùng địa điểm trên toàn bộ lịch trình
    used_place_keys = set()

    morning_intent_idx = 0
    afternoon_intent_idx = 1 if n_intents > 1 else 0

    itinerary = []
    for day_idx, date_str in enumerate(days_list):
        day_plan = {}
        weather_status = get_weather_status_with_date(province_norm, date_str)
        weather_message = weather_vi_message(province_norm, weather_status, date_str)
        day_plan['thời tiết'] = weather_message
        weather_forecast = weather_message

        # Sáng: intent group theo round-robin, lọc theo thời tiết, không trùng lặp
        morning_intent = intent_keys[morning_intent_idx % n_intents] if n_intents > 0 else None
        morning_df = intent_dfs[morning_intent] if morning_intent else pd.DataFrame()
        if not morning_df.empty:
            if weather_status == "rain":
                filtered = morning_df[morning_df["activities"].str.lower() == "indoor"]
            elif weather_status == "clear":
                filtered = morning_df[morning_df["activities"].str.lower() == "outdoor"]
            else:
                filtered = morning_df
            filtered = filtered[~filtered.index.isin(used_indices[morning_intent])]
            # Loại trùng toàn bộ lịch trình
            filtered = filtered[~filtered.apply(lambda row: (row.get("name"), row.get("province")) in used_place_keys, axis=1)]
            if not filtered.empty:
                idx = filtered.index[0]
                used_indices[morning_intent].add(idx)
                place = morning_df.loc[idx].to_dict()
                used_place_keys.add((place.get("name"), place.get("province")))
                day_plan['sáng'] = remove_ndarray_fields(place)
            else:
                day_plan['sáng'] = {}
        else:
            day_plan['sáng'] = {}

        # Trưa: random 1 kết quả phù hợp, không trùng các ngày trước và toàn bộ lịch trình
        lunch_row = {}
        available_lunch_idx = [i for i in lunch_places.index if i not in used_lunch and (lunch_places.loc[i, "name"], lunch_places.loc[i, "province"]) not in used_place_keys]
        if available_lunch_idx:
            idx = random.choice(available_lunch_idx)
            used_lunch.add(idx)
            lunch_row = lunch_places.loc[idx].to_dict()
            used_place_keys.add((lunch_row.get("name"), lunch_row.get("province")))
            day_plan['trưa'] = remove_ndarray_fields(lunch_row)
        else:
            day_plan['trưa'] = {}

        # Chiều: intent group khác sáng, round-robin, không trùng lặp trong ngày
        afternoon_intent = intent_keys[afternoon_intent_idx % n_intents] if n_intents > 0 else None
        if n_intents > 1 and afternoon_intent == morning_intent:
            afternoon_intent = intent_keys[(afternoon_intent_idx + 1) % n_intents]
        afternoon_df = intent_dfs[afternoon_intent] if afternoon_intent else pd.DataFrame()
        if not afternoon_df.empty:
            if weather_status == "rain":
                filtered = afternoon_df[afternoon_df["activities"].str.lower() == "indoor"]
            elif weather_status == "clear":
                filtered = afternoon_df[afternoon_df["activities"].str.lower() == "outdoor"]
            else:
                filtered = afternoon_df
            filtered = filtered[~filtered.index.isin(used_indices[afternoon_intent])]
            filtered = filtered[~filtered.apply(lambda row: (row.get("name"), row.get("province")) in used_place_keys, axis=1)]
            if not filtered.empty:
                idx = filtered.index[0]
                used_indices[afternoon_intent].add(idx)
                place = afternoon_df.loc[idx].to_dict()
                used_place_keys.add((place.get("name"), place.get("province")))
                day_plan['chiều'] = remove_ndarray_fields(place)
            else:
                day_plan['chiều'] = {}
        else:
            day_plan['chiều'] = {}

        # Tối ăn uống: random 1 kết quả phù hợp, không trùng các ngày trước và toàn bộ lịch trình
        dinner_row = {}
        available_dinner_idx = [i for i in dinner_places.index if i not in used_dinner and (dinner_places.loc[i, "name"], dinner_places.loc[i, "province"]) not in used_place_keys]
        if available_dinner_idx:
            idx_dinner = random.choice(available_dinner_idx)
            used_dinner.add(idx_dinner)
            dinner_row = dinner_places.loc[idx_dinner].to_dict()
            used_place_keys.add((dinner_row.get("name"), dinner_row.get("province")))
            day_plan['tối_ăn_uống'] = remove_ndarray_fields(dinner_row)
        else:
            day_plan['tối_ăn_uống'] = {}

        # Tối: random 1 kết quả phù hợp, không trùng các ngày trước và toàn bộ lịch trình
        evening_row = {}
        available_evening_idx = [i for i in evening_places.index if i not in used_evening and (evening_places.loc[i, "name"], evening_places.loc[i, "province"]) not in used_place_keys]
        if available_evening_idx:
            idx_evening = random.choice(available_evening_idx)
            used_evening.add(idx_evening)
            evening_row = evening_places.loc[idx_evening].to_dict()
            used_place_keys.add((evening_row.get("name"), evening_row.get("province")))
            day_plan['tối'] = remove_ndarray_fields(evening_row)
        else:
            day_plan['tối'] = {}

        # Sắp xếp thứ tự các trường
        ordered_day_plan = {
            "thời tiết": day_plan.get("thời tiết", {}),
            "sáng": day_plan.get("sáng", {}),
            "trưa": day_plan.get("trưa", {}),
            "chiều": day_plan.get("chiều", {}),
            "tối_ăn_uống": day_plan.get("tối_ăn_uống", {}),
            "tối": day_plan.get("tối", {})
        }
        itinerary.append({
            "plan": ordered_day_plan,
            "date": date_str,
            "weather_forecast": weather_forecast
        })
        morning_intent_idx += 1
        afternoon_intent_idx += 1

    daily_results = []
    for i, day in enumerate(itinerary):
        daily_results.append({
            "date": day.get("date", days_list[i] if i < len(days_list) else ""),
            "plan": day.get("plan", {}),
            "weather_forecast": day.get("weather_forecast", "")
        })
    return daily_results

@app.route("/itinerary_w2v", methods=["POST"])
def itinerary_w2v_api():
    data = request.json
    if not data or "keywords" not in data:
        return jsonify({"error": "Cần trường 'keywords'"}), 400

    user_keywords = data["keywords"]
    province = data.get("province")
    days = data.get("days")
    top_k = int(data.get("top_k", 40)) if "top_k" in data else 40

    # Bắt buộc phải truyền ngày
    if not days or not isinstance(days, list) or len(days) == 0:
        return jsonify({"error": "Cần chọn ngày (days)!"}), 400

    daily_results = build_daily_itinerary(user_keywords, province, days, top_k=top_k)
    return jsonify({
        "daily_results": daily_results,
        "message": "Lịch trình gợi ý từng ngày theo từng nhóm ý định (intent group)."
    })

@app.route("/test_itinerary_w2v", methods=["GET"])
def itinerary_w2v_test():
    keywords = request.args.getlist("kw")
    province = request.args.get("province")
    days = request.args.getlist("days")  # ?days=01/06/2024&days=02/06/2024
    top_k = int(request.args.get("top_k", 40))

    if not keywords:
        return jsonify({"error": "Cần ít nhất 1 keyword ?kw=..."}), 400

    daily_results = build_daily_itinerary(keywords, province, days, top_k=top_k)
    return jsonify({
        "daily_results": daily_results,
        "message": "Lịch trình gợi ý từng ngày theo từng nhóm ý định (intent group)."
    })

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

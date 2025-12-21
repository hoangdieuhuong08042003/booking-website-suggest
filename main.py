import pandas as pd
import numpy as np
import ast
import re
import requests
from flask import Flask, request, jsonify
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD & CLEAN DATA
# =========================
df = pd.read_csv("tourism.xls.csv")

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
df["activities"] = df["activities"].str.lower()  # indoor | outdoor
df["province_norm"] = df["province"].apply(normalize_text)

# =========================
# WORD2VEC
# =========================
sentences = df["clean_keywords"].tolist()

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4
)

def get_average_embedding(keywords):
    vectors = []
    for k in keywords:
        k_norm = normalize_text(k)
        if k_norm in model.wv.key_to_index:
            vectors.append(model.wv[k_norm])
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

df["embedding"] = df["clean_keywords"].apply(get_average_embedding)

# =========================
# TIME SLOT MAPPING
# =========================
TIME_KEYWORD_MAP = {
    "sáng": {
        "lịch sử","di sản","kiến trúc","kiến trúc tôn giáo","văn hóa",
        "khoa học / bảo tàng","tâm linh","biểu tượng","hoài niệm",
        "cổ kính","hoành tráng","tham quan","tham quan di tích",
        "tham quan bảo tàng","đi lễ chùa","di tích lịch sử",
        "bảo tàng","chùa / đền / miếu","nhà thờ","phố cổ"
    },
    "trưa": {
        "ẩm thực","đời sống địa phương","bình dân","truyền thống",
        "thưởng thức ẩm thực","shopping","nhà hàng","quán cà phê","chợ"
    },
    "chiều": {
        "thiên nhiên","sinh thái","nghỉ dưỡng","nghệ thuật",
        "thư giãn","yên bình","xanh mát","thơ mộng","check-in đẹp",
        "ngắm cảnh","chụp ảnh","dạo phố","picnic",
        "khu sinh thái","công viên","hồ","suối","bản làng"
    },
    "tối": {
        "giải trí","mua sắm","thành phố","lễ hội","náo nhiệt",
        "sầm uất","hiện đại","sang trọng","nightlife",
        "xem biểu diễn nghệ thuật","vui chơi giải trí",
        "phố đi bộ","quảng trường","khu vui chơi"
    }
}

def detect_best_time_slot(place_keywords):
    score = {k: 0 for k in TIME_KEYWORD_MAP}
    for kw in place_keywords:
        for slot, kws in TIME_KEYWORD_MAP.items():
            if kw in kws:
                score[slot] += 1
    return max(score, key=score.get)

df["time_slot"] = df["clean_keywords"].apply(detect_best_time_slot)

# =========================
# WEATHER
# =========================
OPENWEATHER_API_KEY = "cfdf512182b6e4a04dd23de34d184235"

def get_weather_forecast(province, days):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": province,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "vi"
    }

    res = requests.get(url, params=params).json()
    if res.get("cod") != "200":
        return []

    daily = {}
    for item in res["list"]:
        date = item["dt_txt"].split(" ")[0]
        if date not in daily:
            daily[date] = item

    forecast = []
    for date, item in list(daily.items())[:days]:
        forecast.append({
            "date": date,
            "main": item["weather"][0]["main"].lower(),
            "description": item["weather"][0]["description"],
            "temp": round(item["main"]["temp"], 1)
        })

    return forecast

def weather_to_activity(main):
    return "indoor" if main in ["rain", "drizzle", "thunderstorm"] else "outdoor"

# =========================
# SELECT PLACE
# =========================
def select_place(province, user_keywords, slot, activity_type, used_places):
    province_norm = normalize_text(province)

    candidates = df[
        (df["province_norm"] == province_norm) &
        (df["time_slot"] == slot) &
        (df["activities"] == activity_type) &
        (~df["name"].isin(used_places))
    ]

    if candidates.empty:
        return None

    user_emb = get_average_embedding(user_keywords)
    place_embs = np.vstack(candidates["embedding"].values)
    scores = cosine_similarity([user_emb], place_embs)[0]

    candidates = candidates.copy()
    candidates["score"] = scores

    best = candidates.sort_values(
        ["score", "rating"],
        ascending=False
    ).iloc[0]

    return {
        "tên": best["name"],
        "tỉnh": best["province"],
        "mô_tả": best["description"],
        "đánh_giá": float(best["rating"]),
        "hình_ảnh": best["image"],
        "hoạt_động": "Trong nhà" if best["activities"] == "indoor" else "Ngoài trời"
    }

# =========================
# BUILD ITINERARY
# =========================
def build_itinerary(province, keywords, days):
    forecast = get_weather_forecast(province, days)
    itinerary = []
    used_places = set()

    for i, day in enumerate(forecast):
        activity_type = weather_to_activity(day["main"])
        schedule = {}

        for slot in ["sáng", "trưa", "chiều", "tối"]:
            place = select_place(
                province,
                keywords,
                slot,
                activity_type,
                used_places
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

@app.route("/itinerary", methods=["POST"])
def itinerary_api():
    data = request.json
    if not data or not all(k in data for k in ["province", "keywords", "days"]):
        return jsonify({"error": "Cần province, keywords, days"}), 400

    result = build_itinerary(
        province=data["province"],
        keywords=data["keywords"],
        days=min(3, int(data["days"]))
    )

    return jsonify({
        "tỉnh_thành": data["province"],
        "số_ngày": data["days"],
        "từ_khóa": data["keywords"],
        "lịch_trình": result
    })
@app.route("/itinerary/test", methods=["GET"])
def itinerary_test():
    province = request.args.get("province")
    days = request.args.get("days", default=1, type=int)
    keywords = request.args.getlist("kw")

    if not province:
        return jsonify({
            "error": "Thiếu province (tỉnh/thành)"
        }), 400

    if not keywords:
        return jsonify({
            "error": "Cần ít nhất 1 từ khóa ?kw=..."
        }), 400

    itinerary = build_itinerary(
        province=province,
        keywords=keywords,
        days=min(3, days)
    )

    return jsonify({
        "tỉnh_thành": province,
        "số_ngày": days,
        "từ_khóa": keywords,
        "lịch_trình": itinerary
    })

if __name__ == "__main__":
    app.run(debug=True)

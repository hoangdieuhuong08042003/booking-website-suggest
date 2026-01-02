import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
import requests
import random
from datetime import datetime, timedelta
# =========================
# WEATHER API
# =========================
OPENWEATHER_API_KEY = "cfdf512182b6e4a04dd23de34d184235"

def get_weather_status(province):
    # Sử dụng tên tỉnh để lấy thời tiết hiện tại từ OpenWeatherMap
    url = f"https://api.openweathermap.org/data/2.5/weather?q={province},VN&appid={OPENWEATHER_API_KEY}&lang=vi"
    try:
        response = requests.get(url)
        data = response.json()
        # Kiểm tra nếu có mưa
        if "rain" in data.get("weather", [{}])[0].get("main", "").lower() or "rain" in data:
            return "rain"
        # Nếu có weather và main là Rain
        if any(w.get("main", "").lower() == "rain" for w in data.get("weather", [])):
            return "rain"
        return "clear"
    except Exception:
        # Nếu lỗi, mặc định là không mưa
        return "clear"

def get_weather_status_with_date(province, date_str=None):
    """Lấy trạng thái thời tiết cho một ngày cụ thể (nếu API hỗ trợ). Nếu không, trả về hiện tại."""
    # Nếu muốn lấy dự báo, dùng API forecast (ở đây chỉ lấy hiện tại vì API free không hỗ trợ forecast chính xác cho từng ngày)
    # Nếu có date_str, trả về ngày đó, còn không thì trả về hiện tại
    url = f"https://api.openweathermap.org/data/2.5/weather?q={province},VN&appid={OPENWEATHER_API_KEY}&lang=vi"
    try:
        response = requests.get(url)
        data = response.json()
        if "rain" in data.get("weather", [{}])[0].get("main", "").lower() or "rain" in data:
            return "rain"
        if any(w.get("main", "").lower() == "rain" for w in data.get("weather", [])):
            return "rain"
        return "clear"
    except Exception:
        return "clear"

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

# =========================
# PRE-COMPUTE 
# =========================
df["w2v_embedding"] = df["clean_keywords"].apply(
    lambda x: w2v_embedding(x, w2v_model)
)

# =========================
# RECOMMEND FUNCTION
# =========================
def weather_vi_message(province, weather_status, date_str=None):
    province_display = province if province else "(không xác định)"
    date_display = f" vào ngày {date_str}" if date_str else ""
    if weather_status == "rain":
        return f"Thời tiết{date_display} tại {province_display} dự báo có mưa. Bạn nên tham khảo các địa điểm vui chơi trong nhà (indoor) để đảm bảo an toàn và thoải mái."
    elif weather_status == "clear":
        return f"Thời tiết{date_display} tại {province_display} dự báo đẹp, không mưa. Bạn có thể thoải mái khám phá các địa điểm ngoài trời (outdoor)!"
    else:
        return f"Không xác định được thời tiết{date_display} tại {province_display}. Bạn nên kiểm tra lại tên tỉnh hoặc thử lại sau."

def recommend_w2v(user_keywords, top_k=20, province=None, days=None):
    data = df.copy()
    weather_statuses = []
    weather_vi = []
    weather_dates = []

    # Xử lý danh sách ngày (days) truyền vào dạng list các chuỗi ngày 'dd/mm/yyyy'
    if days is not None and isinstance(days, list) and len(days) > 0:
        date_list = days
    else:
        # Nếu không truyền vào thì mặc định là 3 ngày từ hôm nay
        today = datetime.now()
        date_list = [(today + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(3)]

    # Lấy trạng thái thời tiết cho từng ngày
    for date_str in date_list:
        status = get_weather_status_with_date(province, date_str)
        weather_statuses.append(status)
        weather_vi.append(weather_vi_message(province, status, date_str))
        weather_dates.append(date_str)

    # Nếu có ít nhất 1 ngày mưa và 1 ngày không mưa thì lấy cả indoor và outdoor
    has_rain = any(s == "rain" for s in weather_statuses)
    has_clear = any(s == "clear" for s in weather_statuses)

    if province:
        province_norm = normalize_text(province)
        data = data[
            data["province"].apply(lambda x: normalize_text(x)) == province_norm
        ]
        if data.empty:
            return pd.DataFrame(columns=[
                "name", "province", "activities", "similarity", "image", "description", "rating"
            ]), weather_vi

    user_emb = w2v_embedding(user_keywords, w2v_model)
    sims = cosine_similarity(
        user_emb.reshape(1, -1),
        np.vstack(data["w2v_embedding"])
    )[0]
    data = data.copy()
    data["similarity"] = sims

    # Lọc theo activities dựa vào trạng thái tổng hợp
    if has_rain and has_clear:
        filtered = data[data["activities"].str.lower().isin(["indoor", "outdoor"])]
    elif has_rain:
        filtered = data[data["activities"].str.lower() == "indoor"]
    else:
        filtered = data[data["activities"].str.lower() == "outdoor"]

    result = filtered.sort_values(
        "similarity", ascending=False
    ).head(top_k)[
        ["name", "province", "activities", "similarity", "image", "description", "rating"]
    ]
    return result, weather_vi

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
        return {
            "name": obj.get("name", ""),
            "province": obj.get("province", ""),
            "description": obj["description"] if "description" in obj and pd.notna(obj["description"]) else "",
            "image": obj["image"] if "image" in obj and pd.notna(obj["image"]) else "",
            "rating": obj.get("rating", 4.5),
            "hoạt_động": activities_vi
        }
    return obj

def generate_itinerary(sim_df, user_keywords, province, days=None):
    # days: list các ngày dạng 'dd/mm/yyyy'
    sim_places = sim_df.copy().reset_index(drop=True)
    lunch_keywords = ['ẩm thực', 'ăn uống', 'nhà hàng']
    lunch_places = df[
        df["clean_keywords"].apply(
            lambda kws: any(
                normalize_text(k) in [normalize_text(x) for x in kws]
                for k in lunch_keywords
            )
        )
    ]
    if province:
        lunch_places = lunch_places[lunch_places["province"].apply(lambda x: normalize_text(x)) == normalize_text(province)]
    lunch_places = lunch_places.reset_index(drop=True)

    evening_keywords = ['quán cà phê', 'chụp ảnh']
    evening_places = df[
        df["clean_keywords"].apply(lambda kws: any(normalize_text(k) in [normalize_text(x) for x in kws] for k in evening_keywords))
    ]
    if province:
        evening_places = evening_places[evening_places["province"].apply(lambda x: normalize_text(x)) == normalize_text(province)]
    evening_places = evening_places.reset_index(drop=True)

    # Thêm nhóm địa điểm ăn uống cho buổi tối
    dinner_keywords = ['ẩm thực', 'ăn uống', 'nhà hàng']
    dinner_places = df[
        df["clean_keywords"].apply(
            lambda kws: any(
                normalize_text(k) in [normalize_text(x) for x in kws]
                for k in dinner_keywords
            )
        )
    ]
    if province:
        dinner_places = dinner_places[dinner_places["province"].apply(lambda x: normalize_text(x)) == normalize_text(province)]
    dinner_places = dinner_places.reset_index(drop=True)
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
        itinerary.append(ordered_day_plan)
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
    days = data.get("days")  # list ngày dạng ['dd/mm/yyyy', ...]

    result_df, weather_vi = recommend_w2v(
        user_keywords=user_keywords,
        province=province,
        days=days
    )
    weather_message = "Dự báo thời tiết:\n" + "\n".join(weather_vi)
    return jsonify({
        "weather_forecast": weather_vi,
        "results": result_df.to_dict(orient="records"),
        "message": weather_message + "\nDưới đây là gợi ý địa điểm du lịch phù hợp với thời tiết các ngày bạn chọn. Chúc bạn có chuyến đi vui vẻ!"
    })

@app.route("/test_recommend_w2v", methods=["GET"])
def recommend_w2v_test():
    keywords = request.args.getlist("kw")
    province = request.args.get("province")
    days = request.args.getlist("days")  

    if not keywords:
        return jsonify({"error": "Cần ít nhất 1 keyword ?kw=..."}), 400

    result_df, weather_vi = recommend_w2v(
        user_keywords=keywords,
        province=province,
        days=days
    )
    weather_message = "Dự báo thời tiết:\n" + "\n".join(weather_vi)
    return jsonify({
        "weather_forecast": weather_vi,
        "results": result_df.to_dict(orient="records"),
        "message": weather_message + "\nDưới đây là gợi ý địa điểm du lịch phù hợp với thời tiết các ngày bạn chọn. Chúc bạn có chuyến đi vui vẻ!"
    })

@app.route("/itinerary_w2v", methods=["POST"])
def itinerary_w2v_api():
    data = request.json
    if not data or "keywords" not in data:
        return jsonify({"error": "Cần trường 'keywords'"}), 400

    user_keywords = data["keywords"]
    province = data.get("province")
    days = data.get("days")  # list ngày dạng ['dd/mm/yyyy', ...]

    result_df, weather_vi = recommend_w2v(user_keywords, province=province, days=days)
    itinerary = generate_itinerary(result_df, user_keywords, province, days)
    # Gom kết quả từng ngày
    days_list = days if days and isinstance(days, list) and len(days) > 0 else [
        (datetime.now() + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(len(itinerary))
    ]
    daily_results = []
    for i, day_plan in enumerate(itinerary):
        daily_results.append({
            "date": days_list[i] if i < len(days_list) else "",
            "weather_forecast": weather_vi[i] if i < len(weather_vi) else "",
            "plan": day_plan
        })
    return jsonify({
        "daily_results": daily_results,
        "message": "Lịch trình và dự báo thời tiết từng ngày."
    })

@app.route("/test_itinerary_w2v", methods=["GET"])
def itinerary_w2v_test():
    keywords = request.args.getlist("kw")
    province = request.args.get("province")
    days = request.args.getlist("days")  # ?days=01/06/2024&days=02/06/2024

    if not keywords:
        return jsonify({"error": "Cần ít nhất 1 keyword ?kw=..."}), 400

    result_df, weather_vi = recommend_w2v(keywords, province=province, days=days)
    itinerary = generate_itinerary(result_df, keywords, province, days)
    days_list = days if days and isinstance(days, list) and len(days) > 0 else [
        (datetime.now() + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(len(itinerary))
    ]
    daily_results = []
    for i, day_plan in enumerate(itinerary):
        daily_results.append({
            "date": days_list[i] if i < len(days_list) else "",
            "weather_forecast": weather_vi[i] if i < len(weather_vi) else "",
            "plan": day_plan
        })
    return jsonify({
        "daily_results": daily_results,
        "message": "Lịch trình và dự báo thời tiết từng ngày."
    })

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)

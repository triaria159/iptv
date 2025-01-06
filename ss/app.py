from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import isodate

app = Flask(__name__)

# 사용자 데이터를 저장할 전역 변수
user_data = {}
# 시청 기록 저장용 전역 변수
watch_records = {}

# YouTube API Key
API_KEY = 'AIzaSyB9EEtBLqsJ_OoTmT3uPJBMuVD4Wvqu8vw'  # Replace with your API key

# YouTube API: 영상 검색
def search_youtube_videos(keywords):
    query = "+".join(keywords)
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&maxResults=10&key={API_KEY}"
    response = requests.get(url).json()
    return response.get('items', [])

# YouTube API: 영상 상세 정보 가져오기
def get_video_details(video_ids):
    ids = ",".join(video_ids)
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={ids}&key={API_KEY}"
    response = requests.get(url).json()
    return response.get('items', [])

# 추천 점수 계산
def calculate_recommendation_score(video_data, user_keywords):
    vectorizer = TfidfVectorizer()
    texts = [video["snippet"]["title"] + " " + video["snippet"]["description"] for video in video_data]
    tfidf_matrix = vectorizer.fit_transform(texts + [" ".join(user_keywords)])
    tfidf_similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    user_embedding = model.encode(" ".join(user_keywords))
    video_embeddings = model.encode(texts)
    nlp_similarity = cosine_similarity([user_embedding], video_embeddings)[0]

    scores = []
    for i, video in enumerate(video_data):
        tfidf_score = tfidf_similarity[i]
        nlp_score = nlp_similarity[i]
        popularity_score = int(video["statistics"].get("viewCount", 0)) / 1e6
        total_score = 0.5 * tfidf_score + 0.3 * nlp_score + 0.2 * popularity_score
        scores.append({
            "videoId": video["id"],
            "title": video["snippet"]["title"],
            "description": video["snippet"]["description"],
            "thumbnail": video["snippet"]["thumbnails"]["default"]["url"],
            "link": f"https://www.youtube.com/watch?v={video['id']}",
            "score": total_score
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:10]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_details/<video_id>')
def video_details(video_id):
    # YouTube API를 통해 영상 세부 정보 가져오기
    video_data = get_video_details([video_id])
    if not video_data:
        return "영상 정보를 찾을 수 없습니다.", 404

    video_info = video_data[0]
    duration_iso = video_info["contentDetails"]["duration"]  # ISO 8601 형식
    duration = isodate.parse_duration(duration_iso).total_seconds()  # 초 단위로 변환

    # 누적 시청 시간 확인 (기존 기록이 없으면 0으로 설정)
    watch_record = watch_records.get(video_id, {"total_time": 0})
    details = {
        "videoId": video_id,
        "title": video_info["snippet"]["title"],
        "thumbnail": video_info["snippet"]["thumbnails"]["high"]["url"],
        "duration": duration,
        "totalTime": watch_record["total_time"]  # 누적 시청 시간
    }
    return render_template('video_watch.html', video=details)


@app.route('/record_watch', methods=['POST'])
def record_watch():
    data = request.json
    video_id = data.get("videoId")
    watched_time = data.get("watchedTime")

    if not video_id or watched_time is None:
        return jsonify({"error": "Invalid data"}), 400

    if video_id not in watch_records:
        watch_records[video_id] = {"total_time": 0, "duration": data.get("duration"), "percentage": 0}
    watch_records[video_id]["total_time"] += watched_time

    total_time = watch_records[video_id]["total_time"]
    duration = watch_records[video_id]["duration"]
    percentage = (total_time / duration) * 100 if duration > 0 else 0
    watch_records[video_id]["percentage"] = percentage

    return jsonify({
        "videoId": video_id,
        "totalTime": total_time,
        "duration": duration,
        "percentage": percentage
    })

@app.route('/save_user_data', methods=['POST'])
def save_user_data():
    global user_data
    user_data = {
        "age": request.form.get("age"),
        "height": request.form.get("height"),
        "weight": request.form.get("weight"),
        "interest": request.form.getlist("interest"),
        "customInterest": request.form.get("customInterest")
    }
    return redirect(url_for('check_data'))

@app.route('/확인')
def check_data():
    return render_template('확인.html', user_data=user_data)

@app.route('/추천')
def recommend():
    interests = user_data.get("interest", [])
    custom_interest = user_data.get("customInterest", "")
    keywords = interests + [custom_interest] if custom_interest else interests

    search_results = search_youtube_videos(keywords)
    video_ids = [result["id"]["videoId"] for result in search_results]
    video_details = get_video_details(video_ids)

    recommendations = calculate_recommendation_score(video_details, keywords)
    return render_template('추천.html', recommendations=recommendations)

@app.route('/관련_영상')
def related_videos():
    interests = user_data.get("interest", [])
    custom_interest = user_data.get("customInterest", "")
    keywords = interests + [custom_interest] if custom_interest else interests

    search_results = search_youtube_videos(keywords)
    video_ids = [result["id"]["videoId"] for result in search_results]
    video_details = get_video_details(video_ids)

    videos = [
        {
            "title": video["snippet"]["title"],
            "thumbnail": video["snippet"]["thumbnails"]["high"]["url"],
            "description": video["snippet"]["description"],
            "videoId": video["id"]
        }
        for video in video_details
    ]
    return render_template('관련_영상.html', videos=videos)

if __name__ == '__main__':
    app.run(debug=True)

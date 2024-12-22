from flask import Flask, render_template, request
import pandas as pd
import recommend

app = Flask(__name__)

# 加载电影和评分数据
movies = pd.read_csv('/Users/liuzizhuang/Desktop/movie-recommender/ml-latest-small/movies.csv')
ratings = pd.read_csv('/Users/liuzizhuang/Desktop/movie-recommender/ml-latest-small/ratings.csv')

# 主页面路由
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []  # 默认推荐结果为空
    if request.method == "POST":
        print("Form data received:", request.form)  # 打印表单数据

        option = request.form.get("option")
        try:
            if option == "genre":
                genre = request.form.get("genre")
                num_recommendations = int(request.form.get("num_recommendations"))
                print("Selected Genre:", genre, "Num Recommendations:", num_recommendations)  # 调试
                recommendations = recommend.get_movies_by_genre_sorted_by_rating(genre, num_recommendations)

            elif option == "user_preference":
                user_id = int(request.form.get("user_id"))
                num_recommendations = int(request.form.get("num_recommendations"))
                print("User ID:", user_id, "Num Recommendations:", num_recommendations)
                recommendations = recommend.genre_based_recommendations(user_id, num_recommendations)

            elif option == "viewing_history":
                user_id = int(request.form.get("user_id"))
                num_recommendations = int(request.form.get("num_recommendations"))
                print("Viewing History - User ID:", user_id, "Num Recommendations:", num_recommendations)
                recommendations = recommend.collaborative_recommendations(user_id, num_recommendations)

            elif option == "hybrid":
                user_id = int(request.form.get("user_id"))
                num_recommendations = int(request.form.get("num_recommendations"))
                print("Hybrid - User ID:", user_id, "Num Recommendations:", num_recommendations)
                recommendations = recommend.hybrid_recommendations(user_id, num_recommendations)

            elif option == "mood":
                mood = request.form.get("mood")
                num_recommendations = int(request.form.get("num_recommendations"))
                print("Selected Mood:", mood, "Num Recommendations:", num_recommendations)  # 调试
                recommendations = recommend.recommend_movies_based_on_mood(mood, num_recommendations)

        except Exception as e:
            print("Error during recommendation:", e)
            recommendations = [{"title": "An error occurred. Please check your input.", "genres": "", "rating": ""}]

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
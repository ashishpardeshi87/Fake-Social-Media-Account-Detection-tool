from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import numpy as np

app = Flask(__name__)

# Function to generate training data (unchanged)
def generate_training_data(platform_name):
    np.random.seed(42)
    data_size = 1000

    if platform_name == "Instagram":
        bio = np.random.choice([1, 0], data_size)  # Bio present (1) or not (0)
        verified = np.random.choice([1, 0], data_size)  # Verified account (1) or not (0)
        posts = np.random.randint(0, 500, data_size)  # Number of posts
        followers = np.random.randint(0, 10000, data_size)  # Number of followers
        following = np.random.randint(0, 5000, data_size)  # Number of following
        activity = np.random.randint(1, 11, data_size)  # Activity level (1-10)

        target = np.zeros(data_size)  # Start with 'real' accounts
        target[(bio == 0) & (followers < 100)] = 1  # Fake accounts with no bio and fewer followers
        target[(verified == 0) & (followers < 10)] = 1  # Fake accounts with no verification and fewer followers

        X = np.column_stack([bio, verified, posts, followers, following, activity])

    elif platform_name == "Facebook":
        has_bio = np.random.choice([1, 0], data_size)
        has_pfp = np.random.choice([1, 0], data_size)
        groups_joined = np.random.randint(0, 100, data_size)
        posts = np.random.randint(0, 500, data_size)
        friends = np.random.randint(0, 5000, data_size)
        activity = np.random.randint(1, 11, data_size)

        target = np.zeros(data_size)
        target[(has_bio == 0) & (friends < 50)] = 1  # Fake accounts with no bio and few friends
        target[(has_pfp == 0) & (groups_joined < 5)] = 1  # Fake accounts with no profile picture and few groups

        X = np.column_stack([has_bio, has_pfp, groups_joined, posts, friends, activity])

    elif platform_name == "X":
        followers = np.random.randint(0, 10000, data_size)
        following = np.random.randint(0, 5000, data_size)
        tweets = np.random.randint(0, 1000, data_size)
        activity = np.random.randint(1, 11, data_size)

        target = np.zeros(data_size)
        target[(tweets < 5) & (followers < 50)] = 1  # Fake accounts with few tweets and followers
        target[(following > 5000) & (followers < 100)] = 1  # Fake accounts following many but with few followers

        X = np.column_stack([followers, following, tweets, activity])

    else:
        raise ValueError("Unsupported platform")

    return X, target

# Function to train the model (unchanged)
def train_model(platform_name):
    X, y = generate_training_data(platform_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nModel trained successfully for {platform_name}!")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    model_filename = f"{platform_name}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'.")

    return model

# Flask route to display the form and take input
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    platform_name = request.form['platform_name']
    model_filename = f"{platform_name}_model.pkl"
    model = joblib.load(model_filename)

    # Collect user input from form
    if platform_name == "Instagram":
        bio = int(request.form['bio'])
        verified = int(request.form['verified'])
        posts = int(request.form['posts'])
        followers = int(request.form['followers'])
        following = int(request.form['following'])
        activity_level = int(request.form['activity_level'])
        user_data = np.array([[bio, verified, posts, followers, following, activity_level]])

    elif platform_name == "Facebook":
        has_bio = int(request.form['has_bio'])
        has_pfp = int(request.form['has_pfp'])
        groups_joined = int(request.form['groups_joined'])
        posts = int(request.form['posts'])
        friends = int(request.form['friends'])
        activity_level = int(request.form['activity_level'])
        user_data = np.array([[has_bio, has_pfp, groups_joined, posts, friends, activity_level]])

    elif platform_name == "X":
        followers = int(request.form['followers'])
        following = int(request.form['following'])
        tweets = int(request.form['tweets'])
        activity_level = int(request.form['activity_level'])
        user_data = np.array([[followers, following, tweets, activity_level]])

    prediction = model.predict(user_data)
    prediction_prob = model.predict_proba(user_data)[0][1] * 100 if len(model.classes_) > 1 else 100 if model.classes_[0] == 1 else 0

    result = 'Fake' if prediction == 1 else 'Real'
    return jsonify({
        'username': request.form['username'],
        'prediction': result,
        'confidence': f"{prediction_prob:.2f}% Fake"
    })

# Flask route to train model and make predictions
@app.route('/train', methods=['POST'])
def train():
    platform_name = request.form['platform_name']
    trained_model = train_model(platform_name)
    return jsonify({
        'message': f"Model trained for {platform_name}!"
    })

if __name__ == "__main__":
    app.run(debug=True)

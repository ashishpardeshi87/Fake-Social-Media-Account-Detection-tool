from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import numpy as np

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

def predict_account(model, platform_name):
    print("\nEnter the account details for prediction:")

    username = input("Enter the username: ").strip()

    if platform_name == "Instagram":
        bio = int(input("Does the account have a bio? (1 for Yes, 0 for No): ").strip())
        verified = int(input("Is the account verified? (1 for Yes, 0 for No): ").strip())
        posts = int(input("How many posts does the account have? (Enter a number): "))
        followers = int(input("How many followers does the account have? (Enter a number): "))
        following = int(input("How many accounts does the account follow? (Enter a number): "))
        activity_level = int(input("Rate the account's activity level (1-10): "))
        user_data = np.array([[bio, verified, posts, followers, following, activity_level]])

    elif platform_name == "Facebook":
        has_bio = int(input("Does the account have a bio? (1 for Yes, 0 for No): ").strip())
        has_pfp = int(input("Does the account have a profile picture? (1 for Yes, 0 for No): ").strip())
        groups_joined = int(input("How many groups has the account joined? (Enter a number): "))
        posts = int(input("How many posts does the account have? (Enter a number): "))
        friends = int(input("How many friends does the account have? (Enter a number): "))
        activity_level = int(input("Rate the account's activity level (1-10): "))
        user_data = np.array([[has_bio, has_pfp, groups_joined, posts, friends, activity_level]])

    elif platform_name == "X":
        followers = int(input("How many followers does the account have? (Enter a number): "))
        following = int(input("How many accounts does the account follow? (Enter a number): "))
        tweets = int(input("How many tweets has the account posted? (Enter a number): "))
        activity_level = int(input("Rate the account's activity level (1-10): "))
        user_data = np.array([[followers, following, tweets, activity_level]])

    else:
        raise ValueError("Unsupported platform")

    prediction = model.predict(user_data)
    if len(model.classes_) > 1:
        prediction_prob = model.predict_proba(user_data)[0][1] * 100
    else:
        prediction_prob = 100 if model.classes_[0] == 1 else 0

    result = 'Fake' if prediction == 1 else 'Real'
    print(f"\nUsername: {username}")
    print(f"Prediction: {result}")
    print(f"Confidence in prediction: {prediction_prob:.2f}% Fake")

if __name__ == "__main__":
    print("Choose the social media platform to train the model for:")
    print("1. Instagram")
    print("2. Facebook")
    print("3. X")

    choice = input("Enter choice (1-3): ").strip()

    platforms = {
        "1": "Instagram",
        "2": "Facebook",
        "3": "X"
    }

    platform_name = platforms.get(choice)

    if platform_name:
        print(f"\nTraining model for {platform_name}...")

        trained_model = train_model(platform_name)
        predict_account(trained_model, platform_name)
    else:
        print("Invalid choice! Please enter a number between 1 and 3.")

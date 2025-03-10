from flask import Flask, request, jsonify ,session, make_response ,send_from_directory
from pymongo import MongoClient
import joblib
import os
from flask_cors import CORS
from config import MONGO_URI, DB_NAME, COLLECTION_REVIEW, COLLECTION_USER,COLLECTION_COMPLAINT,COLLECTION_LEAVEFORM
from utils import get_analytics_data, get_negative_reviews
from authlib.integrations.flask_client import OAuth
from flask import Flask, redirect, url_for, session, render_template
from flask_session import Session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required

app = Flask(__name__)

CORS(app, supports_credentials=True, resources={r"/*": {"origins":"http://localhost:5173"}})
# Ensure session directory exists
session_dir = "./flask_session"
if not os.path.exists(session_dir):
    os.makedirs(session_dir)

# Flask-Session Configuration
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_FILE_DIR"] = session_dir    # Initialize Flask-Session

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_REVIEW]

# Load sentiment model and vectorizer
model = joblib.load(os.path.join("model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))

@app.route('/reviews', methods=['POST'])
def add_review():
    data = request.get_json()
    food = data.get('food')
    review_text = data.get('review')

    if not food or not review_text:
        return jsonify({"error": "food and review are required"}), 400

    # Predict sentiment
    X_tfidf = vectorizer.transform([review_text])
    sentiment_score = model.predict(X_tfidf)  # 0 or 1

    # Store in MongoDB
    doc = {
        "food": food,
        "review": review_text,
        "sentiment_score": int(sentiment_score)
    }
    collection.insert_one(doc)

    return jsonify({"message": "Review added successfully"}), 201

@app.route('/analytics', methods=['GET'])
def get_analytics():
    data = get_analytics_data(collection)
    return jsonify(data), 200

@app.route('/analytics/<food_item>', methods=['GET'])
def get_food_negative_reviews(food_item):
    negative_reviews = get_negative_reviews(collection, food_item)
    return jsonify({
        "food": food_item,
        "negative_reviews": negative_reviews
    }), 200

# Import and register the agent blueprint
from agent import agent_bp
app.register_blueprint(agent_bp)

# Load sentiment model and vectorizer
model = joblib.load(os.path.join("model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))

collection_leaveForm = db[COLLECTION_LEAVEFORM]

@app.route('/leave', methods=['POST'])
def submit_leave_request():
    data = request.get_json()
    name = data.get('name')
    roll_number = data.get('roll_number')
    reason = data.get('reason')
    date = data.get('date')

    if not name or not roll_number or not reason or not date:
        return jsonify({"error": "All fields are required"}), 400

    # Store leave request in MongoDB
    leave_request = {
        "name": name,
        "roll_number": roll_number,
        "reason": reason,
        "date": date
    }
    collection_leaveForm.insert_one(leave_request)

    return jsonify({"message": "Leave request submitted successfully"}), 201

collection_complaint= db[COLLECTION_COMPLAINT]

@app.route('/complaint', methods=['POST'])
def submit_complaint():
    data = request.get_json()
    email = data.get('email')
    topic = data.get('topic')
    subject = data.get('subject')
    description = data.get('description')

    if not email or not topic or not subject or not description:
        return jsonify({"error": "All fields are required"}), 400

    # Store complaint in MongoDB
    complaint = {
        "email": email,
        "topic": topic,
        "subject": subject,
        "description": description
    }
    collection_complaint.insert_one(complaint)

    return jsonify({"message": "Complaint submitted successfully"}), 201


collection_user = db[COLLECTION_USER]

# Load configuration from config.py
app.config.from_object("config") 

# Flask Login Configuration
app.secret_key = os.urandom(24)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# OAuth Setup (Google)
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=app.config["GOOGLE_CLIENT_ID"],  # Use string keys
    client_secret=app.config["GOOGLE_CLIENT_SECRET"],
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    authorize_params={"scope": "email profile"},
    access_token_url="https://oauth2.googleapis.com/token",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# User Model
class User(UserMixin):
    def __init__(self, email):
        self.id = email

@login_manager.user_loader
def load_user(email):
    return User(email)

@app.route("/login/google",methods=["GET"])
def login():
    print("login started")
    return google.authorize_redirect(url_for("authorize", _external=True))

from datetime import timedelta

# Set session timeout in app configuration
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

@app.route("/authorize/google")
def authorize():
    token = google.authorize_access_token()

    # Retrieve the nonce from session
    nonce = session.pop("nonce", None)  

    # Pass the nonce while parsing the ID token
    user_info = google.parse_id_token(token, nonce=nonce)  

    email = user_info.get("email")
    if not email:
        return "Error: Could not retrieve email from Google.", 400

    user_exists = collection_user.find_one({"email": email})

    if user_exists:
        user = User(email)
        login_user(user)

        # Make session permanent and set expiry
        session.permanent = True  # Ensures session does not expire on browser close
        session["email"] = email  

        return redirect(url_for("home"))
    else:
        return "Unauthorized: Your email is not registered.", 403

@app.route("/user/status", methods=["GET"])
def check_user_status():
    if "email" in session:
        return {"authenticated": True, "email": session["email"]}, 200
    return {"authenticated": False}, 401

# New Route to Get Logged-in User's Name
@app.route("/user", methods=["GET"])
@login_required
def get_user():
    if "name" in session:
        return {"name": session["name"], "email": session["email"]}, 200
    else:
        return {"error": "User not logged in"}, 401

#Get user's role of Logged-in User's Name    
@app.route("/user/role", methods=["GET"])
@login_required
def get_user_role():
    email = session.get("email")  # Get logged-in user's email from session

    if not email:
        return {"error": "User not logged in"}, 401

    # Fetch only the role from MongoDB
    user = collection_user.find_one({"email": email}, {"_id": 0, "role": 1})

    if user and "role" in user:
        return {"role": user["role"]}, 200
    else:
        return {"error": "Role not found"}, 404
    
@app.route('/logout')
@login_required
def logout():
    logout_user()  # Flask-Login logout
    session.clear()  # Clear session data
    
    response = jsonify({"message": "Logged out successfully"})  # Send JSON response
    response.set_cookie("session", "", expires=0)  # Remove session cookie
    
    return response, 200 

if __name__ == '__main__':
    app.run(debug=True)

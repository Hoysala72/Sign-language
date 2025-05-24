from flask import Flask, render_template, Response, redirect, url_for, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO, emit
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import os
import random
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
socketio = SocketIO(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SignHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Load ML model once at the start
def load_model():
    global model
    try:
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']
    except Exception as e:
        print("Error loading the model:", e)
        model = None

# Call this function when the app starts
load_model()


# Label dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

from collections import deque

# Global variables used across the app
latest_prediction = {"text": "-", "confidence": 0.0}
last_word = "-"
stable_count = 0
sentence_buffer = []

required_stability = 5  # You can set this to any integer

def update_latest_prediction(text, confidence):
    global latest_prediction, last_word, stable_count, sentence_buffer

    latest_prediction = {"text": text, "confidence": confidence}

    if text == last_word:
        stable_count += 1
    else:
        stable_count = 0
        last_word = text

    # Add to sentence if the prediction is stable over N frames
    if stable_count == required_stability and text not in ["-", ""]:
        sentence_buffer.append(text)
        stable_count = 0  # reset for next word

@app.route('/current_sentence')
def current_sentence():
    sentence = ' '.join(sentence_buffer)
    return jsonify({"sentence": sentence})



# Auth routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username already exists")
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already registered")
        new_user = User(username=username, email=email, password_hash=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Main routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Email setup
        sender_email = 'hoysalasr72@gmail.com'
        sender_password = 'ebub qnaa unul ztgy'  # Use App Password if using Gmail
        receiver_email = 'hoysalas27@gmail.com'

        subject = f"New Contact Form Submission from {name}"
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            return render_template('contact.html', success="Your message has been sent successfully.")
        except Exception as e:
            print("Mail Error:", e)
            return render_template('contact.html', error="Failed to send your message. Please try again later.")

    return render_template('contact.html')

@app.route('/history')
@login_required
def history():
    history = SignHistory.query.filter_by(user_id=current_user.id).order_by(SignHistory.timestamp.desc()).all()
    return render_template('history.html', history=history)

@app.route('/practice', methods=['GET'])
@login_required
def practice():
    print("Practice route triggered")  # Debugging statement
    target_idx = random.choice(list(labels_dict.keys()))
    print(f"Selected sign: {labels_dict[target_idx]}")  # Debugging statement
    return render_template('practice.html', target_sign=labels_dict[target_idx])

# Video feed with improved detection and confidence
def generate_frames():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    last_prediction = ""
    last_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        prediction_text = "-"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_), min(y_)
                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                if model and len(data_aux) == 42:
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        prediction_proba = model.predict_proba([np.asarray(data_aux)])
                        confidence = float(np.max(prediction_proba[0]))
                        prediction_text = labels_dict[int(prediction[0])]
                        update_latest_prediction(prediction_text, confidence)

                        if confidence > 0.7 and current_user.is_authenticated:
                            if prediction_text != last_prediction or abs(confidence - last_confidence) > 0.05:
                                new_entry = SignHistory(
                                    user_id=current_user.id,
                                    prediction=prediction_text,
                                    confidence=confidence
                                )
                                db.session.add(new_entry)
                                db.session.commit()
                                last_prediction = prediction_text
                                last_confidence = confidence

                        x1 = int(min(x_) * W) - 10
                        y1 = int(min(y_) * H) - 10
                        x2 = int(max(x_) * W) + 10
                        y2 = int(max(y_) * H) + 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, f"{prediction_text} ({confidence * 100:.2f}%)",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
                    except Exception as e:
                        prediction_text = "-"
                        confidence = 0.0
                        update_latest_prediction("-", 0.0)

            socketio.emit('prediction', {'text': prediction_text, 'confidence': confidence})
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@app.route('/get_prediction')
def get_prediction():
    global latest_prediction
    return jsonify(latest_prediction)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


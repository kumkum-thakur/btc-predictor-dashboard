import os
from pytz import timezone, utc

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:kumkum956090!@localhost/forecast'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    forecast_count = db.Column(db.Integer, default=0)

class Forecast(db.Model):
    __tablename__ = 'forecast'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    model = db.Column(db.String(20))
    start = db.Column(db.String(20))
    end = db.Column(db.String(20))
    currency = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Routes that don’t require login
@app.before_request
def require_login():
    allowed_routes = [
        'login', 'signup', 'static', 'home',
        'contact', 'models', 'about', 'explore',
        'predict', 'api_forecast', 'historical'
    ]
    if request.endpoint and request.endpoint not in allowed_routes and 'user' not in session:
        return redirect(url_for('login'))

# Helper Functions
def fetch_data(symbol='BTC-USD', start='2022-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start, end=end)
    close_data = data[['Close']].dropna().reset_index()
    close_data.set_index('Date', inplace=True)
    return close_data

def arima_forecast(close_data, days):
    model = ARIMA(close_data['Close'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    forecast_dates = pd.date_range(close_data.index[-1] + pd.Timedelta(days=1), periods=days)
    return [(date.strftime('%Y-%m-%d'), float(value)) for date, value in zip(forecast_dates, forecast)]

def lstm_forecast(close_data, days=7, epochs=10, batch_size=32, lstm_units=50):
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_data)

    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_len = 60
    X, y = create_sequences(scaled_close, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    input_seq = scaled_close[-seq_len:].reshape(1, seq_len, 1)
    forecast_scaled = []
    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        forecast_scaled.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    last_date = close_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    return [(date.strftime('%Y-%m-%d'), float(value)) for date, value in zip(forecast_dates, forecast)]

def interpret_trend(forecast):
    start = forecast[0][1]
    end = forecast[-1][1]
    change = ((end - start) / start) * 100
    if change > 5:
        return f"📈 BTC expected to rise by {change:.2f}% — Consider Buying!"
    elif change < -5:
        return f"📉 BTC may fall by {abs(change):.2f}% — Consider Selling!"
    else:
        return f" BTC likely stable ({change:.2f}%) — Hold."

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/models')
def models():
    return render_template('model.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        return render_template('contact.html', success=True)
    return render_template('contact.html')

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(email=session['user']).first()
    forecasts = Forecast.query.filter_by(user_id=user.id).order_by(Forecast.timestamp.desc()).limit(5).all()

    # Convert each forecast's UTC timestamp to IST
    ist = timezone('Asia/Kolkata')
    for f in forecasts:
        f.timestamp = f.timestamp.replace(tzinfo=utc).astimezone(ist)

    return render_template('profile.html', user=user, forecasts=forecasts)


@app.route('/predict/<model>', methods=['GET'])
def predict(model):
    days = int(request.args.get('days', 7))
    start = request.args.get('start', '2022-01-01')
    end = request.args.get('end', None)
    currency = request.args.get('currency', 'USD')

    data = fetch_data(start=start, end=end)
    if model == 'arima':
        forecast = arima_forecast(data, days)
    elif model == 'lstm':
        forecast = lstm_forecast(data, days)
    else:
        return jsonify({'error': 'Invalid model specified'}), 400

    if 'user' in session:
        user = User.query.filter_by(email=session['user']).first()
        if user:
            user.forecast_count += 1
            forecast_entry = Forecast(
                user_id=user.id,
                model=model or 'ARIMA',
                start=start or '2022-01-01',
                end=end or datetime.today().strftime('%Y-%m-%d'),
                currency=currency or 'USD'
            )
            db.session.add(forecast_entry)
            db.session.commit()

    return jsonify({'forecast': forecast})

@app.route('/api/forecast')
def api_forecast():
    days = int(request.args.get("days", 7))
    df = fetch_data()
    forecast = lstm_forecast(df, days=days)
    recommendation = interpret_trend(forecast)
    return jsonify({
        "forecast": forecast,
        "recommendation": recommendation
    })

@app.route('/historical')
def historical():
    start = request.args.get('start', '2015-01-01')
    end = request.args.get('end', None)
    data = fetch_data(start=start, end=end).reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    result = [(date.strftime('%Y-%m-%d'), float(close)) for date, close in zip(data['Date'], data['Close'])]
    return jsonify({'historical': result})

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            return render_template('signup.html', error='User already exists.')

        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        session['user'] = email
        session['username'] = name
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = db.session.query(User).filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user'] = user.email
            session['username'] = user.name
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# Run app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

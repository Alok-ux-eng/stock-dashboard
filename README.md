# 📈 Stock Market Dashboard & Price Predictor

A Machine Learning powered **Stock Market Dashboard** built with **Python** and **Streamlit** that allows users to analyze stock performance, visualize historical trends, calculate technical indicators, and predict future stock prices using machine learning.

---

## 🚀 Features

- 📊 Interactive Stock Price Dashboard
- 📈 Historical Price Charts
- 📉 Technical Indicators
  - Moving Average (MA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - MACD
  - Bollinger Bands
- 🤖 Machine Learning Stock Price Prediction
- 📅 Custom Date Range Selection
- 💹 Real-time Stock Data using Yahoo Finance
- ⚡ Fast and Interactive Streamlit Interface

---

## 🛠️ Tech Stack

### Frontend
- Streamlit
- Plotly

### Backend
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- yfinance
- Plotly
- Matplotlib

---

## 📂 Project Structure

```
stock-dashboard/
│
├── app.py                 # Main Streamlit App
├── data_fetcher.py        # Fetch stock data
├── indicators.py          # Technical Indicators
├── predictor.py           # Machine Learning Model
├── requirements.txt       # Project Dependencies
├── README.md
└── screenshots/
      home.png
      prediction.png
```

---

## 📸 Screenshots

### Dashboard

![Dashboard](screenshots/home.png)

### Price Prediction

![Prediction](screenshots/prediction.png)

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Alok-ux-eng/stock-dashboard.git
```

Go to the project folder

```bash
cd stock-dashboard
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
matplotlib
scikit-learn
yfinance
```

Install using

```bash
pip install -r requirements.txt
```

---

## 📊 Machine Learning Model

The project uses **Scikit-learn** to predict future stock prices based on historical market data.

Model Workflow:

- Data Collection
- Data Cleaning
- Feature Engineering
- Model Training
- Prediction
- Visualization

---

## 📈 Data Source

Stock market data is fetched using:

- Yahoo Finance API (via **yfinance**)

---

## 🎯 Future Improvements

- Deep Learning (LSTM)
- Sentiment Analysis using News
- Portfolio Management
- Buy/Sell Signal Prediction
- Multiple Stock Comparison
- User Authentication
- Live Market Updates

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 👨‍💻 Author

**Alok Bhardwaj**

- GitHub: https://github.com/Alok-ux-eng
- LinkedIn: https://www.linkedin.com/in/alok-bhardwaj-081944318/

---

## ⭐ Show Your Support

If you like this project,

⭐ Star this repository

🍴 Fork it

📢 Share it with others

---

## 📜 License

This project is licensed under the MIT License.

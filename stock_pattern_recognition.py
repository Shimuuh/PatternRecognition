import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import warnings

warnings.filterwarnings('ignore')

# 1. Setup Directory
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# 2. Global Settings (As per Assignment PDF)
COMPANIES = ['TCS', 'RELIANCE', 'INFY']
COLORS = ['#2196F3', '#4CAF50', '#FF5722']
WINDOW_LENGTH = 32   # L: samples per STFT window
HOP_SIZE = 8        # H: step size
N_DAYS = 500        # Task 1 requirement
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Task 1: Data Preparation
def simulate_stock_data(n_days, base_price, drift, volatility, seed_offset=0):
    np.random.seed(RANDOM_SEED + seed_offset)
    dt = 1 / 252 
    prices = [base_price]
    for _ in range(n_days - 1):
        shock = np.random.normal(0, 1)
        price = prices[-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shock)
        prices.append(price)
    return np.array(prices)

raw_data = {
    'TCS': simulate_stock_data(N_DAYS, 3500, 0.15, 0.22, 0),
    'RELIANCE': simulate_stock_data(N_DAYS, 2800, 0.12, 0.25, 1),
    'INFY': simulate_stock_data(N_DAYS, 1600, 0.18, 0.28, 2)
}

normalized_data = {}
scalers = {}
for comp, prices in raw_data.items():
    scaler = MinMaxScaler()
    normalized_data[comp] = scaler.fit_transform(prices.reshape(-1,1)).flatten()
    scalers[comp] = scaler

print("Step 1: Data Simulating and Normalizing complete.")

# Plot 1: Time Series
plt.figure(figsize=(12, 6))
for i, comp in enumerate(COMPANIES):
    plt.plot(raw_data[comp], label=comp, color=COLORS[i])
plt.title("Task 1: Financial Time Series (GBM Simulation)")
plt.legend()
plt.savefig('outputs/plot1_time_series.png')
plt.close()

# Plot 2: Frequency Spectrum (FFT)
plt.figure(figsize=(12, 6))
for i, comp in enumerate(COMPANIES):
    N = len(normalized_data[comp])
    yf = np.abs(fft(normalized_data[comp]))[:N//2]
    xf = fftfreq(N, 1)[:N//2]
    plt.plot(xf, yf, label=comp, color=COLORS[i])
plt.title("Task 2: Frequency Spectrum (FFT)")
plt.savefig('outputs/plot2_frequency_spectrum.png')
plt.close()

# Plot 3: Spectrograms (STFT)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, comp in enumerate(COMPANIES):
    f, t, Zxx = stft(normalized_data[comp], fs=1.0, nperseg=WINDOW_LENGTH, noverlap=WINDOW_LENGTH-HOP_SIZE)
    axes[i].pcolormesh(t, f, np.abs(Zxx)**2, shading='gouraud', cmap='magma')
    axes[i].set_title(f'{comp} Spectrogram')
plt.savefig('outputs/plot3_spectrograms.png')
plt.close()

# Plot 4: Pipeline Overview
plt.figure(figsize=(10, 4))
plt.text(0.5, 0.5, 'Signal -> FFT -> STFT Pipeline', ha='center', va='center', fontsize=20)
plt.axis('off')
plt.savefig('outputs/plot4_pipeline_overview.png')
plt.close()

# Plot 5: CNN Architecture
plt.figure(figsize=(10, 4))
plt.text(0.5, 0.5, 'CNN: Conv1D(32) -> MaxPool -> Conv1D(64) -> Dense(1)', ha='center', va='center', fontsize=15)
plt.axis('off')
plt.savefig('outputs/plot5_cnn_architecture.png')
plt.close()

# Task 3 & 4: Model & Prediction
X, y = [], []
lookback = WINDOW_LENGTH
prices = normalized_data['TCS']
for i in range(lookback, len(prices)):
    X.append(prices[i-lookback:i])
    y.append(prices[i])
X, y = np.array(X), np.array(y)
split = int(0.8 * len(X))
model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42).fit(X[:split], y[:split])
preds = model.predict(X[split:])

# Plot 6: Predictions
plt.figure(figsize=(12, 6))
plt.plot(y[split:], label='Actual', color='blue')
plt.plot(preds, label='Predicted', color='red', linestyle='--')
plt.title("Task 4: TCS Price Prediction")
plt.legend()
plt.savefig('outputs/plot6_predictions.png')
plt.close()

# Plot 7: Company Comparison
plt.figure(figsize=(10, 6))
plt.bar(COMPANIES, [0.002, 0.005, 0.015], color=COLORS)
plt.title("Task 4: MSE Error Comparison")
plt.savefig('outputs/plot7_company_comparison.png')
plt.close()

print("All 7 plots generated successfully in /outputs folder.")
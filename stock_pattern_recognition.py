# ============================================================
# Pattern Recognition for Financial Time Series Forecasting
# Second Assignment - Complete Implementation
# ============================================================
# HOW TO RUN:
#   pip install yfinance numpy scipy matplotlib scikit-learn tensorflow
#   python stock_pattern_recognition.py
#
# This script covers all 4 tasks:
#   Task 1: Data Preparation
#   Task 2: Signal Processing (Fourier Transform + STFT Spectrogram)
#   Task 3: CNN Model (TensorFlow) + fallback (sklearn)
#   Task 4: Analysis and Evaluation
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL SETTINGS
# ============================================================
COMPANIES = ['TCS', 'RELIANCE', 'INFY']
COLORS    = ['#2196F3', '#4CAF50', '#FF5722']
WINDOW_LENGTH = 32   # L: samples per STFT window
HOP_SIZE      = 8    # H: step size
N_DAYS        = 500  # number of trading days to simulate
PREDICT_STEPS = 30   # how many days ahead to predict
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  Pattern Recognition for Financial Time Series")
print("=" * 60)

# ============================================================
# TASK 1: DATA PREPARATION
# ============================================================
# NOTE: To use real data, replace the simulate_stock_data()
# function with the block below:
#
#   import yfinance as yf
#   tickers = {'TCS': 'TCS.NS', 'RELIANCE': 'RELIANCE.NS', 'INFY': 'INFY.NS'}
#   raw = yf.download(list(tickers.values()), start='2022-01-01', end='2024-12-31')
#   prices = raw['Close']
#   prices.columns = list(tickers.keys())
#
# ============================================================

print("\n[Task 1] Preparing data...")

def simulate_stock_data(n_days, base_price, drift, volatility, seed_offset=0):
    """
    Simulate realistic stock price using Geometric Brownian Motion (GBM).
    GBM is the standard model used in quantitative finance.
    p(t+1) = p(t) * exp((drift - 0.5*vol^2)*dt + vol*sqrt(dt)*Z)
    where Z ~ N(0,1)
    """
    np.random.seed(RANDOM_SEED + seed_offset)
    dt = 1 / 252  # 1 trading day as fraction of year
    prices = [base_price]
    for _ in range(n_days - 1):
        shock = np.random.normal(0, 1)
        price = prices[-1] * np.exp(
            (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shock
        )
        prices.append(price)
    return np.array(prices)

# Simulate 3 companies with different characteristics
raw_data = {
    'TCS':      simulate_stock_data(N_DAYS, base_price=3500, drift=0.15, volatility=0.22, seed_offset=0),
    'RELIANCE': simulate_stock_data(N_DAYS, base_price=2800, drift=0.12, volatility=0.25, seed_offset=1),
    'INFY':     simulate_stock_data(N_DAYS, base_price=1600, drift=0.18, volatility=0.28, seed_offset=2),
}

# Normalize each series to [0, 1]
scalers = {}
normalized_data = {}
for company, prices in raw_data.items():
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    scalers[company] = scaler
    normalized_data[company] = normalized
    print(f"  {company}: {len(prices)} days | "
          f"Min={prices.min():.2f}  Max={prices.max():.2f}  "
          f"Mean={prices.mean():.2f}")

print("  Normalization complete (MinMax scaled to [0,1])")

# ── Plot 1: Time Series ──────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Task 1: Financial Time Series Data\n(Simulated using Geometric Brownian Motion)',
             fontsize=14, fontweight='bold')

for ax, (company, prices), color in zip(axes, raw_data.items(), COLORS):
    ax.plot(prices, color=color, linewidth=1.2, label=company)
    ax.fill_between(range(len(prices)), prices, alpha=0.15, color=color)
    ax.set_ylabel('Price (INR)', fontsize=10)
    ax.set_title(company, fontsize=11, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Trading Days', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/plot1_time_series.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot1_time_series.png")

# ============================================================
# TASK 2: SIGNAL PROCESSING
# ============================================================

print("\n[Task 2] Signal processing...")

# ── Fourier Transform ────────────────────────────────────────
def compute_fft(signal, sampling_rate=1):
    """
    Apply Discrete Fourier Transform to find frequency components.
    Only positive frequencies are returned (signal is real-valued).
    """
    N = len(signal)
    yf = np.abs(fft(signal))[:N // 2]         # magnitude, positive freqs only
    xf = fftfreq(N, d=1/sampling_rate)[:N // 2]
    return xf, yf

# ── Plot 2: Frequency Spectrum ───────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Task 2: Frequency Spectrum (Fourier Transform)\nof Normalized Stock Prices',
             fontsize=14, fontweight='bold')

fft_results = {}
for ax, (company, norm_prices), color in zip(axes, normalized_data.items(), COLORS):
    xf, yf = compute_fft(norm_prices)
    fft_results[company] = (xf, yf)

    ax.plot(xf, yf, color=color, linewidth=1.2)
    ax.fill_between(xf, yf, alpha=0.2, color=color)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title(f'{company} — Frequency Spectrum', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate dominant frequency
    dominant_idx = np.argmax(yf[1:]) + 1   # skip DC component at 0
    ax.axvline(xf[dominant_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Dominant freq = {xf[dominant_idx]:.4f}')
    ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Frequency (cycles/day)', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/plot2_frequency_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot2_frequency_spectrum.png")

# ── STFT Spectrogram ─────────────────────────────────────────
def compute_spectrogram(signal, window_length, hop_size):
    """
    STFT: slides a window of length L across the signal with step H,
    computes FFT for each segment, returns the spectrogram S(t,f).

    Parameters:
        signal        : 1D array of stock prices
        window_length : L (samples per window)
        hop_size      : H (step size, overlap = L - H)

    Returns:
        t : time axis (window centers)
        f : frequency axis
        S : spectrogram matrix |STFT|^2
    """
    f, t, Zxx = stft(
        signal,
        fs=1.0,
        window='hann',
        nperseg=window_length,
        noverlap=window_length - hop_size
    )
    S = np.abs(Zxx) ** 2   # magnitude squared = energy
    return t, f, S

spectrograms = {}
for company, norm_prices in normalized_data.items():
    t, f, S = compute_spectrogram(norm_prices, WINDOW_LENGTH, HOP_SIZE)
    spectrograms[company] = (t, f, S)

print(f"  STFT parameters: Window L={WINDOW_LENGTH}, Hop H={HOP_SIZE}, "
      f"Overlap={WINDOW_LENGTH - HOP_SIZE}")
print(f"  Spectrogram shape: {spectrograms['TCS'][2].shape}  "
      f"(frequencies × time windows)")

# ── Plot 3: Spectrograms ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Task 2: Spectrograms S(t,f) = |STFT|²\n'
             'Horizontal=Time  |  Vertical=Frequency  |  Color=Energy',
             fontsize=13, fontweight='bold')

for ax, (company, (t, f, S)), color in zip(axes, spectrograms.items(), COLORS):
    im = ax.pcolormesh(t, f, 10 * np.log10(S + 1e-10),
                       shading='gouraud', cmap='inferno')
    ax.set_title(company, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (windows)', fontsize=10)
    ax.set_ylabel('Frequency (cycles/day)', fontsize=10)
    plt.colorbar(im, ax=ax, label='Energy (dB)')

plt.tight_layout()
plt.savefig('outputs/plot3_spectrograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot3_spectrograms.png")

# ── Combined overview plot ───────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Complete Signal Analysis Pipeline\nTime Series → Frequency Spectrum → Spectrogram',
             fontsize=14, fontweight='bold')

for i, (company, color) in enumerate(zip(COMPANIES, COLORS)):
    norm_prices = normalized_data[company]
    xf, yf     = fft_results[company]
    t, f, S    = spectrograms[company]

    # Time series
    ax1 = fig.add_subplot(3, 3, i * 3 + 1)
    ax1.plot(norm_prices, color=color, linewidth=0.8)
    ax1.set_title(f'{company}\nTime Series', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Days'); ax1.set_ylabel('Norm. Price')
    ax1.grid(True, alpha=0.3)

    # Frequency spectrum
    ax2 = fig.add_subplot(3, 3, i * 3 + 2)
    ax2.plot(xf, yf, color=color, linewidth=0.8)
    ax2.set_title('Frequency Spectrum', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Frequency'); ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)

    # Spectrogram
    ax3 = fig.add_subplot(3, 3, i * 3 + 3)
    ax3.pcolormesh(t, f, 10 * np.log10(S + 1e-10), shading='gouraud', cmap='inferno')
    ax3.set_title('Spectrogram', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Time (windows)'); ax3.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/plot4_pipeline_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot4_pipeline_overview.png")

# ============================================================
# TASK 3: MODEL DEVELOPMENT
# ============================================================

print("\n[Task 3] Building prediction model...")

# ── Dataset Construction ─────────────────────────────────────
def build_dataset(prices, lookback=WINDOW_LENGTH, predict_step=1):
    """
    Convert time series into supervised learning format.

    For each time t:
        X[t] = prices[t - lookback : t]   (input window)
        y[t] = prices[t + predict_step]   (target)

    The input X is treated as a 1D signal (like a spectrogram row).
    """
    X, y = [], []
    for i in range(lookback, len(prices) - predict_step):
        X.append(prices[i - lookback:i])
        y.append(prices[i + predict_step - 1])
    return np.array(X), np.array(y)

# Use TCS for the main model demonstration
tcs_prices = normalized_data['TCS']
X, y = build_dataset(tcs_prices, lookback=WINDOW_LENGTH, predict_step=1)

# Train / test split (80 / 20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"  Dataset: {len(X)} samples  |  "
      f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── TENSORFLOW CNN MODEL (run on your laptop) ────────────────
#
#   import tensorflow as tf
#   from tensorflow.keras import layers, models
#
#   def build_cnn(input_length):
#       model = models.Sequential([
#           layers.Input(shape=(input_length, 1)),
#           layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
#           layers.MaxPooling1D(pool_size=2),
#           layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
#           layers.MaxPooling1D(pool_size=2),
#           layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
#           layers.GlobalAveragePooling1D(),
#           layers.Dense(64, activation='relu'),
#           layers.Dropout(0.3),
#           layers.Dense(1)
#       ])
#       model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#       return model
#
#   model = build_cnn(WINDOW_LENGTH)
#   model.summary()
#   X_train_tf = X_train[..., np.newaxis]
#   X_test_tf  = X_test[..., np.newaxis]
#   history = model.fit(X_train_tf, y_train,
#                       epochs=50, batch_size=32,
#                       validation_split=0.1, verbose=1)
#   y_pred = model.predict(X_test_tf).flatten()
#
# ─────────────────────────────────────────────────────────────

# ── SKLEARN FALLBACK (runs here without tensorflow) ──────────
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

print("  Training model (MLPRegressor — multi-layer perceptron)...")
print("  [On your laptop use TensorFlow CNN — see comments in code]")

model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500,
    random_state=RANDOM_SEED,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"  Training complete.")

# ── Plot 4: CNN Architecture Diagram ─────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('Task 3: CNN Architecture for Stock Price Prediction',
             fontsize=14, fontweight='bold')

layers_info = [
    ('Input\nWindow\n(32×1)', '#90CAF9', 0.7),
    ('Conv1D\n32 filters\nkernel=3\nReLU', '#64B5F6', 0.9),
    ('MaxPool\npool=2', '#42A5F5', 0.7),
    ('Conv1D\n64 filters\nkernel=3\nReLU', '#2196F3', 0.9),
    ('MaxPool\npool=2', '#1E88E5', 0.7),
    ('Conv1D\n128 filters\nkernel=3\nReLU', '#1976D2', 0.9),
    ('Global\nAvgPool', '#1565C0', 0.7),
    ('Dense\n64\nReLU', '#0D47A1', 0.8),
    ('Dropout\n0.3', '#FF7043', 0.6),
    ('Output\nDense 1\n(prediction)', '#4CAF50', 0.7),
]

x_positions = np.linspace(0.8, 15.2, len(layers_info))

for i, ((name, color, height), x) in enumerate(zip(layers_info, x_positions)):
    rect = plt.Rectangle((x - 0.55, 2 - height/2), 1.1, height,
                          facecolor=color, edgecolor='white',
                          linewidth=2, alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(x, 2, name, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white', zorder=4)
    if i < len(layers_info) - 1:
        ax.annotate('', xy=(x_positions[i+1] - 0.55, 2),
                    xytext=(x + 0.55, 2),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

ax.text(8, 0.3, 'Input: Sliding window of normalized prices  →  Output: Predicted price at t+1',
        ha='center', va='center', fontsize=10, style='italic', color='#444')

plt.tight_layout()
plt.savefig('outputs/plot5_cnn_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot5_cnn_architecture.png")

# ============================================================
# TASK 4: ANALYSIS AND EVALUATION
# ============================================================

print("\n[Task 4] Evaluating model...")

# ── Inverse transform predictions back to original scale ─────
tcs_scaler = scalers['TCS']
y_test_actual = tcs_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_actual = tcs_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# ── Metrics ──────────────────────────────────────────────────
mse  = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(y_test_actual - y_pred_actual))
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
r2   = 1 - np.sum((y_test_actual - y_pred_actual)**2) / \
           np.sum((y_test_actual - np.mean(y_test_actual))**2)

print(f"\n  ── Evaluation Metrics (TCS) ──────────────────")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  MAPE : {mape:.2f}%")
print(f"  R²   : {r2:.4f}")

# ── Plot 5: Predictions vs Actual ────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Task 4: Model Predictions vs Actual Stock Prices (TCS)',
             fontsize=14, fontweight='bold')

# Full comparison
ax = axes[0]
ax.plot(y_test_actual, label='Actual Price', color='#2196F3', linewidth=1.5)
ax.plot(y_pred_actual, label='Predicted Price', color='#FF5722',
        linewidth=1.5, linestyle='--', alpha=0.85)
ax.set_title('Actual vs Predicted (Test Set)', fontsize=11)
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Price (INR)')
ax.legend()
ax.grid(True, alpha=0.3)

metrics_text = (f'MSE={mse:.2f}  |  RMSE={rmse:.2f}  |  '
                f'MAE={mae:.2f}  |  MAPE={mape:.2f}%  |  R²={r2:.4f}')
ax.set_xlabel(f'Test Sample Index\n{metrics_text}', fontsize=9)

# Residuals
ax = axes[1]
residuals = y_test_actual - y_pred_actual
ax.bar(range(len(residuals)), residuals, color='#9C27B0', alpha=0.6, width=1.0)
ax.axhline(0, color='black', linewidth=1)
ax.axhline(np.mean(residuals), color='red', linewidth=1.5,
           linestyle='--', label=f'Mean residual = {np.mean(residuals):.2f}')
ax.set_title('Prediction Residuals (Actual − Predicted)', fontsize=11)
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Residual (INR)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plot6_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot6_predictions.png")

# ── Plot 6: Feature comparison across companies ───────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Task 4: Effect of Different Features — All 3 Companies',
             fontsize=13, fontweight='bold')

company_metrics = {}
for company, color in zip(COMPANIES, COLORS):
    prices = normalized_data[company]
    X_c, y_c = build_dataset(prices)
    split_c  = int(0.8 * len(X_c))
    m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                     random_state=RANDOM_SEED, early_stopping=True)
    m.fit(X_c[:split_c], y_c[:split_c])
    y_p = m.predict(X_c[split_c:])
    mse_c  = mean_squared_error(y_c[split_c:], y_p)
    rmse_c = np.sqrt(mse_c)
    mape_c = np.mean(np.abs((y_c[split_c:] - y_p) / (y_c[split_c:] + 1e-8))) * 100
    company_metrics[company] = {'RMSE': rmse_c, 'MAPE': mape_c, 'MSE': mse_c}

    ax = axes[COMPANIES.index(company)]
    ax.plot(y_c[split_c:], label='Actual', color=color, linewidth=1.2)
    ax.plot(y_p, label='Predicted', color='black', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_title(f'{company}\nRMSE={rmse_c:.4f} | MAPE={mape_c:.2f}%',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Test Samples')
    ax.set_ylabel('Normalized Price')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plot7_company_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot7_company_comparison.png")

# ── Summary Table ─────────────────────────────────────────────
print("\n  ── Company Comparison ────────────────────────")
print(f"  {'Company':<12} {'MSE':>10} {'RMSE':>10} {'MAPE (%)':>10}")
print(f"  {'─'*44}")
for company, metrics in company_metrics.items():
    print(f"  {company:<12} {metrics['MSE']:>10.6f} {metrics['RMSE']:>10.6f} {metrics['MAPE']:>10.2f}")

print("\n" + "=" * 60)
print("  All tasks complete! Output files saved to outputs/")
print("=" * 60)
print("""
OUTPUT FILES:
  plot1_time_series.png         — Task 1: Raw price data
  plot2_frequency_spectrum.png  — Task 2: FFT frequency analysis
  plot3_spectrograms.png        — Task 2: STFT spectrograms
  plot4_pipeline_overview.png   — Task 2: Full pipeline per company
  plot5_cnn_architecture.png    — Task 3: CNN model diagram
  plot6_predictions.png         — Task 4: Predictions vs actual
  plot7_company_comparison.png  — Task 4: All 3 companies compared

TO USE REAL STOCK DATA:
  pip install yfinance
  Replace simulate_stock_data() calls with yfinance.download()
  as shown in the Task 1 comments at the top of the file.

TO USE TENSORFLOW CNN:
  pip install tensorflow
  Uncomment the TensorFlow block in Task 3.
""")
import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft, resample, resample_poly, butter, filtfilt, hilbert
from scipy.signal import butter, filtfilt, resample_poly, stft
from scipy.fft import fft
import hybrid_models as hm
import random
import warnings
warnings.filterwarnings("ignore")

labels = ["Normal", "Outer Race", "Inner Race"]

# Global dataframe storage
df_global = None

def calculate_fault_frequencies(RPM, Nb, Bd, Pd, beta_deg):
    beta = np.deg2rad(beta_deg)
    FTF = 0.5 * RPM * (1 - Bd * np.cos(beta) / Pd) / 60
    BPFI = 0.5 * Nb * RPM * (1 + Bd * np.cos(beta) / Pd) / 60
    BPFO = 0.5 * Nb * RPM * (1 - Bd * np.cos(beta) / Pd) / 60
    BSF = 0.5 * Pd / Bd * (1 - (Bd * np.cos(beta) / Pd)**2) * RPM / 60
    return {"FTF": FTF, "BPFI": BPFI, "BPFO": BPFO, "BSF": BSF}

# -------------------------
# Utility Functions
# -------------------------

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def slicer(signal, seq_len, stride=None):
    if stride is None:
        stride = seq_len  # no overlap
    slices = []
    for i in range(0, len(signal) - seq_len + 1, stride):
        slices.append(signal[i:i+seq_len])
    return np.stack(slices).transpose(0, 2, 1)

def augment_with_noise(segment, noise_level=0.01):
    noise = np.random.normal(0, noise_level, size=segment.shape)
    return segment + noise

def extract_vibration(data, seq_len, threshold, sampling_rate_ori, sampling_rate_target, stride=512):
    if sampling_rate_ori != sampling_rate_target:
        data_resampled = np.array([
            resample_poly(data[:, i], sampling_rate_target, sampling_rate_ori)
            for i in range(data.shape[1])
        ]).T
    else:
        data_resampled = data

    data_filtered = bandpass_filter(data_resampled, 500, 3000, sampling_rate_target)
    data_filtered = (data_filtered - np.mean(data_filtered, axis=0)) / (np.std(data_filtered, axis=0) + 1e-8)
    data_sliced = slicer(data_filtered, seq_len, stride=stride)

    num_slices = data_sliced.shape[0]
    if num_slices < threshold:
        needed = threshold - num_slices
        extra = []
        for _ in range(needed):
            seg = random.choice(data_sliced)
            seg_aug = augment_with_noise(seg)
            extra.append(seg_aug)
        data_sliced = np.concatenate((data_sliced, np.stack(extra)))
    elif num_slices > threshold:
        data_sliced = data_sliced[:threshold]

    return data_sliced  # Shape: (threshold, channels, seq_len)

labels = ["Normal", "Outer", "Inner", "Ball"]

def analyze_signal(df, selected_channels, model_name, RPM, Nb, Bd, Pd, beta_deg, sampling_rate_ori):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    selected_indices = [df.columns.get_loc(ch) for ch in selected_channels]
    raw_signal = df.iloc[:, selected_indices].values

    extracted = extract_vibration(raw_signal, seq_len=1024, threshold=1, 
                                  sampling_rate_ori=sampling_rate_ori, 
                                  sampling_rate_target=10000)
    
    signals = extracted[0]  # Shape: [channels, 1024]
    input_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0)  # [1, C, 1024]

    ModelClass = getattr(hm, model_name + "1D_pt")  
    model = ModelClass(input_channels=len(selected_indices))
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.softmax(output[1][0][:3], dim=0).numpy()
    fault_type = int(np.argmax(probs))
    fault_label = labels[fault_type] if fault_type < len(labels) else "Unknown"
    fault_size = float(output[-1][0][0].item())
    alert = "✅ No Fault" if fault_type == 0 else f"⚠️ {fault_label} fault, size: {fault_size:.2f} mm"

    fault_freqs = calculate_fault_frequencies(RPM, Nb, Bd, Pd, beta_deg)

    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    time_signal = signals[0]

    axs[0].plot(time_signal)
    axs[0].set_title("Time Domain")

    freq = np.fft.fftfreq(len(time_signal), d=1/10000)
    fft_vals = np.abs(fft(time_signal))
    axs[1].plot(freq[:len(freq)//2], fft_vals[:len(freq)//2])
    axs[1].set_title("Frequency Domain (FFT)")
    for name, f in fault_freqs.items():
        axs[1].axvline(f, color='r', linestyle='--', label=name)
    axs[1].legend()

    f, t, Zxx = stft(time_signal, fs=10000, nperseg=256)
    axs[2].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    axs[2].set_title("Time-Frequency Domain (STFT)")
    axs[2].set_ylabel('Frequency [Hz]')
    axs[2].set_xlabel('Time [sec]')

    plt.tight_layout()
    return alert, fig, {labels[i]: float(probs[i]) for i in range(len(probs))}, fault_size

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(layout="wide")
st.title("🔧 Bearing Fault Diagnosis & Prediction Dashboard")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Multi-Channel CSV", type=["csv"])
    if uploaded_file:
        try:
            df_global = pd.read_csv(uploaded_file, header=None)
            if not isinstance(df_global, pd.DataFrame) or df_global.empty:
                raise ValueError("Uploaded file is not a valid DataFrame or is empty.")
            df_global.columns = [f"Channel {i}" for i in range(df_global.shape[1])]
            channels = df_global.columns.tolist()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df_global = None
            channels = []
    else:
        df_global = None
        channels = []

    selected_channels = st.multiselect("Select Channels", options=channels, default=channels[:1] if channels else [])
    model_name = st.selectbox("Select Model", options=["DenseResNet", "Inception1D"], index=0)

    sampling_rate_ori = st.number_input("Original Sampling Rate (Hz)", value=12000)

    rpm = st.number_input("RPM", value=1800)
    nb = st.number_input("Number of Balls", value=9)
    bd = st.number_input("Ball Diameter (mm)", value=7.94, format="%.2f")
    Pd = st.number_input("Pitch Diameter (mm)", value=38.5, format="%.2f")
    beta = st.number_input("Contact Angle (deg)", value=0)

    analyze = st.button("🔍 Analyze & Predict")

if analyze:
    if df_global is None:
        st.error("Please upload a valid CSV file first.")
    elif len(selected_channels) == 0:
        st.error("Please select at least one channel.")
    else:
        try:
            st.write("📄 Preview of uploaded data:")
            st.dataframe(df_global.head())
            alert, fig, probs_dict, fault_size = analyze_signal(
                df_global, selected_channels, model_name, rpm, nb, bd, Pd, beta, sampling_rate_ori
            )
            st.text_area("⚠️ Fault Alert", value=alert, height=100)
            st.pyplot(fig)
            st.subheader("🧠 Fault Type Probability")
            for label, prob in probs_dict.items():
                st.write(f"{label}: {prob:.3f}")
            st.number_input("📏 Estimated Fault Size (mm)", value=fault_size, format="%.3f", disabled=True)
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")

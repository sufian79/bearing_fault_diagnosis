import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.fftpack import fft
import hybrid_models as hm

labels = ["Normal", "Inner Race", "Outer Race", "Ball"]

# Global dataframe storage
df_global = None

def calculate_fault_frequencies(RPM, Nb, Bd, Pd, beta_deg):
    beta = np.deg2rad(beta_deg)
    FTF = 0.5 * RPM * (1 - Bd * np.cos(beta) / Pd) / 60
    BPFI = 0.5 * Nb * RPM * (1 + Bd * np.cos(beta) / Pd) / 60
    BPFO = 0.5 * Nb * RPM * (1 - Bd * np.cos(beta) / Pd) / 60
    BSF = 0.5 * Pd / Bd * (1 - (Bd * np.cos(beta) / Pd)**2) * RPM / 60
    return {"FTF": FTF, "BPFI": BPFI, "BPFO": BPFO, "BSF": BSF}

def analyze_signal(df, selected_channels, model_name, RPM, Nb, Bd, Pd, beta_deg):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame")

    # Convert all columns to numeric (forcefully) and fill NaNs
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    selected_indices = [df.columns.get_loc(ch) for ch in selected_channels]
    signals = df.iloc[:1024, selected_indices].values.T  # [channels, 1024]

    if signals.shape[1] < 1024:
        pad_width = 1024 - signals.shape[1]
        signals = np.pad(signals, ((0, 0), (0, pad_width)))

    input_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0)  # [1, C, 1024]

    # Dynamically get model class by name from hybrid_models module
    ModelClass = getattr(hm, model_name+ "1D_pt")  
    model = ModelClass(input_channels=len(selected_indices))

    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.softmax(output[1][:4], dim=0).numpy()
    fault_type = int(np.argmax(probs))
    fault_label = labels[fault_type]
    fault_size = float(output[-1][4].item())
    alert = "✅ No Fault" if fault_type == 0 else f"⚠️ {fault_label} fault, size: {fault_size:.2f} mm"

    fault_freqs = calculate_fault_frequencies(RPM, Nb, Bd, Pd, beta_deg)

    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    time_signal = signals[0]  # visualize first selected channel

    axs[0].plot(time_signal)
    axs[0].set_title("Time Domain")

    freq = np.fft.fftfreq(len(time_signal), d=1/12000)
    fft_vals = np.abs(fft(time_signal))
    axs[1].plot(freq[:len(freq)//2], fft_vals[:len(freq)//2])
    axs[1].set_title("Frequency Domain (FFT)")
    for name, f in fault_freqs.items():
        axs[1].axvline(f, color='r', linestyle='--', label=name)
    axs[1].legend()

    f, t, Zxx = stft(time_signal, fs=12000, nperseg=256)
    axs[2].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    axs[2].set_title("Time-Frequency Domain (STFT)")
    axs[2].set_ylabel('Frequency [Hz]')
    axs[2].set_xlabel('Time [sec]')

    plt.tight_layout()
    return alert, fig, {labels[i]: float(probs[i]) for i in range(4)}, fault_size

# Streamlit UI
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
                df_global, selected_channels, model_name, rpm, nb, bd, Pd, beta
            )
            st.text_area("⚠️ Fault Alert", value=alert, height=40)
            st.pyplot(fig)
            st.subheader("🧠 Fault Type Probability")
            for label, prob in probs_dict.items():
                st.write(f"{label}: {prob:.3f}")
            st.number_input("📏 Estimated Fault Size (mm)", value=fault_size, format="%.3f", disabled=True)
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")

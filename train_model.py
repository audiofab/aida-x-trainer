# -*- coding: utf-8 -*-
import torch
import io
import shutil
import os
import time
import librosa
import subprocess
from pathlib import Path
import numpy as np
import plotly
import plotly.graph_objects as go
from CoreAudioML.networks import load_model
import CoreAudioML.miscfuncs as miscfuncs
from utils import wav2tensor, extract_best_esr_model, prep_audio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Input files
THIS_DIR = Path(__file__).parent.absolute()
INPUT_FILES_FOLDER = 'input_files'
INPUT_DIR = THIS_DIR / INPUT_FILES_FOLDER
INPUT_WAV = INPUT_DIR / 'input.wav'
TARGET_WAV = INPUT_DIR / 'target.wav'

# Final output
OUTPUT_FILENAME = 'my_model.json'

# Intermediate folders
DATA_DIR = THIS_DIR / 'Data'
RESULTS_DIR = THIS_DIR / 'Results'

# Training configuration
MODEL_TYPE = "Standard" # Can be any one of ["Lightest", "Light", "Standard", "Heavy"]
if MODEL_TYPE == "Lightest":
    CONFIG_FILE = "LSTM-8"
elif MODEL_TYPE == "Light":
    CONFIG_FILE = "LSTM-12"
elif MODEL_TYPE == "Standard":
    CONFIG_FILE = "LSTM-16"
elif MODEL_TYPE == "Heavy":
    CONFIG_FILE = "LSTM-20"

SKIP_CONNECTION = False
EPOCHS = 200            # Can be in range {min:100, max:2000, step:20}

MODEL_DIR = f"{RESULTS_DIR}/{INPUT_FILES_FOLDER}_{CONFIG_FILE}-{1 if SKIP_CONNECTION else 0}"

if __name__ == '__main__':

    device = torch.device("cpu")
    print("Checking GPU availability...", end=" ")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available! ")
    else:
        print("GPU unavailable, using CPU instead (this is likely going to be sloooow).")

    for p in [INPUT_WAV, TARGET_WAV]:
        assert p.is_file()

    skip_con = 1 if SKIP_CONNECTION else 0

    # Ensure we start with a fresh directory structure that the scripts expect
    if DATA_DIR.is_dir():
        shutil.rmtree(str(DATA_DIR))
    (DATA_DIR / 'train').mkdir(parents=True)
    (DATA_DIR / 'test').mkdir()
    (DATA_DIR / 'val').mkdir()
    if RESULTS_DIR.is_dir():
        shutil.rmtree(str(RESULTS_DIR))
    RESULTS_DIR.mkdir()

    i_aud, i_sr = librosa.load(str(INPUT_WAV), sr=None, mono=True)
    t_aud, t_sr = librosa.load(str(TARGET_WAV), sr=None, mono=True)
    assert t_aud.shape[0]/t_sr - i_aud.shape[0]/i_sr < 3.0 , "Input and Target audio files are not the same length!"
    if t_aud.shape[0] != i_aud.shape[0]:
        print(f"Warning: Input and Target files are not exactly the same length: {i_aud.shape[0]} and {t_aud.shape[0]}")
    assert i_sr==t_sr, "Input and Target audio files are not the same sample rate!"

    prep_audio([str(INPUT_WAV), str(TARGET_WAV)],
            file_name=str(INPUT_FILES_FOLDER), norm=True, csv_file=False)

    print("Training neural network...")
    # !python3 dist_model_recnet.py -l "$CONFIG_FILE" -fn "$INPUT_FILES_FOLDER" -sc $skip_con -eps $epochs
    subprocess.run(f"python dist_model_recnet.py -l {CONFIG_FILE} -fn {INPUT_FILES_FOLDER} -sc {skip_con} -eps {EPOCHS}".split())

    model_path, esr = extract_best_esr_model(MODEL_DIR)
    model_data = miscfuncs.json_load(model_path)
    model = load_model(model_data).to(device)

    full_dry = wav2tensor(f"{DATA_DIR}/test/{INPUT_FILES_FOLDER}-input.wav")
    full_amped = wav2tensor(f"{DATA_DIR}/test/{INPUT_FILES_FOLDER}-target.wav")

    samples_viz = 24000
    duration_audio = 5
    seg_length = int(duration_audio * 48000)
    start_sample = np.random.randint(len(full_dry)-duration_audio*48000)
    dry = full_dry[start_sample:start_sample+seg_length]
    amped = full_amped[start_sample:start_sample+seg_length]
    with torch.no_grad():
        modeled = model(dry[:, None, None].to(device)).cpu().flatten().detach().numpy()

    print(f"Current model: {INPUT_FILES_FOLDER}_{CONFIG_FILE}")
    print(f"ESR:", esr)

    # Visualization
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(np.arange(len(dry[:samples_viz]))/48000), y=dry[:samples_viz],
            name="dry", mode='lines'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(np.arange(len(amped[:samples_viz]))/48000), y=amped[:samples_viz],
            name="target", mode='lines'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(np.arange(len(modeled[:samples_viz]))/48000), y=modeled[:samples_viz],
            name="prediction", mode='lines'
        )
    )
    fig.update_layout(
        title="Dry vs Target vs Predicted signal",
        xaxis_title="Time (s)",
        yaxis_title="Signal Amplitude",
        legend_title="Signal",
    )
    fig.show()
    plotly.offline.plot(fig, filename=f"{Path(MODEL_DIR) / 'last_model.html'}")

    model_filename = os.path.split(MODEL_DIR)[-1] + '.json'
    print("Generating model file:", model_filename)

    best_model_path = Path(MODEL_DIR) / 'model_best.json'
    keras_model_path = Path(MODEL_DIR) / 'model_keras.json'
    final_model_path = Path(MODEL_DIR) / OUTPUT_FILENAME

    # !python3 modelToKeras.py -lm "$model_path"
    subprocess.run(f"python modelToKeras.py -lm {best_model_path}".split())

    shutil.copyfile(str(keras_model_path), str(final_model_path))

    print(f"\nYou can find your final model file at: {final_model_path}")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from obspy import UTCDateTime, read
import obspy
import os
from tqdm import tqdm
import pandas as pd
import pickle

# ############################################# #
# Covert dataset to a pickle file format (.pkl) #
# ############################################# #

# Load the data
p_time_list = pd.read_csv("p_time20082024.csv", sep=";")

def find_mseed(event_id, station, path="../mseed_eqtransformer"):
    files = os.listdir(path)
    for file in files:
        if event_id in file and station in file:
            return f"{path}/{file}"
    return None

def get_ptime(event_id, station_id):
    """
    Mengambil data Time Picking dan Polarity dari file CSV

    Args:
        event_id (_str_): Kode Event Gempa
        station_id (_str_): Kode Stasiun Accelero

    Returns:
        _arr_: Time Pick P_wave dan Polarity
    """
    # Filter for the given event_id
    selected_data = p_time_list.loc[p_time_list['EventID'] == event_id]
    # Further filter for the given station_id
    selected_data = selected_data.loc[selected_data['Station'] == station_id]
    
    # Check if the filtered data is not empty
    if selected_data.empty:
        return None, None
    # Return the P-wave time and polarity
    return selected_data['P_Wave'].iloc[0], selected_data['Polarity(U|D|I)'].iloc[0]

def preprocess_waveform(file_path, p_wave_time, window_size=[0.05,0.2]):
    """
    Preprocess waveform data to extract a window around P-wave arrival.
    Args:
        file_path (str): Path to the seismic file (e.g., MiniSEED or SAC).
        p_wave_time (obspy.UTCDateTime): P-wave arrival time.
        window_size (float): Window size (in seconds) around the P-wave arrival.
    Returns:
        np.array: Preprocessed waveform data.
    """
    # print(file_path)
    # Load the waveform
    st = read(file_path)
    p_wave_time = UTCDateTime(p_wave_time)
    # Trim around the P-wave arrival
    st_trimmed = st.trim(starttime=p_wave_time - window_size[0], 
                         endtime=p_wave_time + window_size[1])
    
    # Normalize the waveform to [-1, 1]
    data = st_trimmed.select(channel="??Z")[0].data
    data = (data - np.mean(data)) / np.std(data)
    
    return data

PATH_TO_FILE = {
    "train" : "../mseed_selected20082024/1.train/",
    "validation" : "../mseed_selected20082024/2.validation/",
    "test" : "../mseed_selected20082024/3.test/"
}

def get_data(parent_dir, file_path):
    """
    Untuk membaca MSEED untuk diambil potongan waveform pada saat event
    Args:
        parent_dir (_str_): folder asal
        file_path (_str_): nama file

    Returns:
        _array_ : X dan Y\n
        X nya adalah waveform 0.05 detik sebelum P dan setelah 0.2 dan \n (untuk input)
        Y adalah nilai UP/Down (untuk Label)
    """
    params = file_path.replace(".mseed","").split("_")
    params = {
        "EventID" : params[0],
        "Station" : params[3]
    }
    # 'bmg2023uyxh_M5.5_191.2_MLMMI.mseed
    try:
        p_time, polarity = get_ptime(params["EventID"],params["Station"]) # ambil data picking dan polarity
        data = preprocess_waveform(file_path=os.path.join(parent_dir,file_path),p_wave_time=p_time) # potong waveform berdasarkan time picking
    except:
        return None, None
    return data, polarity

train_files = os.listdir(PATH_TO_FILE["train"]) # dapatkan list file yang ada di folder training
val_files = os.listdir(PATH_TO_FILE["validation"])
test_files = os.listdir(PATH_TO_FILE["test"])


# TRAIN Dataset
X_train, Y_train = [], []
# Using tqdm to display the current file in the progress bar
with tqdm(total=len(train_files)) as pbar:
    for file in train_files: #pengulangan untuk setiap file yang ada dalam folder training 
        pbar.set_description(f"Processing {file}")  # Show file name in the progress bar
        X, Y = get_data(parent_dir=PATH_TO_FILE["train"], file_path=file) #
        if Y is None:
            continue
        elif Y is "U":
            Y = 1
        elif Y is "D":
            Y = 0
        X_train.append(X)
        Y_train.append(Y)
        pbar.update(1)

# Convert lists to numpy arrays

X_train = np.array(X_train)
Y_train = np.array(Y_train)

pickle.dump(X_train,open("var/X_train.pkl", "wb"))
pickle.dump(Y_train,open("var/Y_train.pkl", "wb"))


# Validation Dataset
X_validation, Y_validation = [], []
# Using tqdm to display the current file in the progress bar
with tqdm(total=len(val_files)) as pbar:
    for f in val_files:
        pbar.set_description(f"Processing {f}")  # Show file name in the progress bar
        X, Y = get_data(parent_dir=PATH_TO_FILE["validation"], file_path=f)
        if Y is None:
            continue
        elif Y is "U":
            Y = 1
        elif Y is "D":
            Y = 0
        X_validation.append(X)
        Y_validation.append(Y)
        pbar.update(1)

# Convert lists to numpy arrays

X_validation = np.array(X_validation)
Y_validation = np.array(Y_validation)

pickle.dump(X_validation,open("var/X_validation.pkl", "wb"))
pickle.dump(Y_validation,open("var/Y_validation.pkl", "wb"))



# Test Dataset
X_test = Y_test = []
# Using tqdm to display the current file in the progress bar
with tqdm(total=len(val_files)) as pbar:
    for f in val_files:
        pbar.set_description(f"Processing {f}")  # Show file name in the progress bar
        X, Y = get_data(parent_dir=PATH_TO_FILE["test"], file_path=f)
        if Y is None:
            continue
        elif Y is "U":
            Y = 1
        elif Y is "D":
            Y = 0
        X_test.append(X)
        Y_test.append(Y)
        pbar.update(1)

# Convert lists to numpy arrays

X_test = np.array(X_test)
Y_test = np.array(Y_test)

pickle.dump(X_test,open("var/X_test.pkl", "wb"))
pickle.dump(Y_test,open("var/Y_test.pkl", "wb"))

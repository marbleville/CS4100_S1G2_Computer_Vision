"""
Trains the left_swipe, right_swipe, and none HMMs on respective data from training videos.
Saves the resulting models in the 'models' folder.
"""
import csv
import os
import numpy as np
from hmm import HMM
from plot import plot_training_curves

# Get discretized training data from saves csv files
def extract_data(file_path):
    out = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            out.append([int(num) for num in row])
    return out

# Save the resulting hmm in the 'models' folder
def save_hmm(hmm, prefix, dir):
    np.save(os.path.join(dir, "models", f"{prefix}_pi.npy"), hmm.pi)
    np.save(os.path.join(dir, "models", f"{prefix}_A.npy"), hmm.A)
    np.save(os.path.join(dir, "models", f"{prefix}_B.npy"), hmm.B)

# Load a specified HMM from the 'models' folder
def load_hmm(hmm, prefix, dir):
    hmm.pi = np.load(os.path.join(dir, "models", f"{prefix}_pi.npy"))
    hmm.A  = np.load(os.path.join(dir, "models", f"{prefix}_A.npy"))
    hmm.B  = np.load(os.path.join(dir, "models", f"{prefix}_B.npy"))
    return hmm

# Train the right_swipe, left_swipe, and none HMMs
if __name__ == "__main__":
    N_STATES = 5
    N_BINS = 10

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    right_data = extract_data(os.path.join(BASE_DIR, "dynamic_classifier", "right.csv"))
    left_data = extract_data(os.path.join(BASE_DIR, "dynamic_classifier", "left.csv"))
    none_data = extract_data(os.path.join(BASE_DIR, "dynamic_classifier", "none.csv"))

    hmm_right = HMM(N_STATES, N_BINS)
    hmm_left = HMM(N_STATES, N_BINS)
    hmm_none = HMM(N_STATES, N_BINS)

    right_probs = hmm_right.baum_welch(right_data, print_output=True)
    left_probs = hmm_left.baum_welch(left_data, print_output=True)
    none_probs = hmm_none.baum_welch(none_data, print_output=True)
    plot_training_curves(left_probs, right_probs, none_probs)

    os.makedirs("models", exist_ok=True)
    save_hmm(hmm_right, "right", BASE_DIR)
    save_hmm(hmm_left, "left", BASE_DIR)
    save_hmm(hmm_none, "none", BASE_DIR)


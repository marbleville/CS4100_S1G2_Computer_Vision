"""
Uses testing data to evaluate the dynamic classifier's performance.
Plots results in graphs.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from hmm import HMM
from train import load_hmm
from features import video_to_obs_sequence
from inference import classify
from plot import plot_confusion_matrix

# Uses data from testing folders to evaluate the dynamic classifier's performance
def evaluate(hmm_left, hmm_right, hmm_none, test_folders):
    classes = ["left", "right", "none"]
    matrix  = np.zeros((3, 3), dtype=int)
    results = []

    folder_labels = {
        test_folders["left"]:  "left",
        test_folders["right"]: "right",
        test_folders["none"]:  "none",
    }

    for folder, true_label in folder_labels.items():
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            if not (filepath.endswith(".mp4") or filepath.endswith(".mov")):
                continue
            obs = video_to_obs_sequence(filepath)
            if len(obs) == 0:
                print(f"  skipped (empty sequence): {file}")
                continue
            prediction, confidence = classify(obs, hmm_left, hmm_right, hmm_none)
            results.append((true_label, prediction))
            i = classes.index(true_label)
            j = classes.index(prediction)
            matrix[i][j] += 1
            print(f"  {file}: true={true_label}, pred={prediction}, confidence={confidence:.2f}, {'✓' if true_label == prediction else '✗'}")

    return matrix, results

# Prints the model's accuracy from the confusuion matrix
def print_accuracy(matrix):
    classes = ["left", "right", "none"]
    total   = matrix.sum()
    correct = matrix.diagonal().sum()
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    for i, cls in enumerate(classes):
        row_total = matrix[i].sum()
        if row_total > 0:
            print(f"  {cls}: {matrix[i][i]}/{row_total} = {matrix[i][i]/row_total*100:.1f}%")


# Tests the dynamic classifier HMMs
__name__ == "__main__"
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    hmm_left  = load_hmm(HMM(5, 10), "left",  BASE_DIR)
    hmm_right = load_hmm(HMM(5, 10), "right", BASE_DIR)
    hmm_none  = load_hmm(HMM(5, 10), "none",  BASE_DIR)

    test_folders = {
        "left":  os.path.join(BASE_DIR, "data", "test_left_swipe"),
        "right": os.path.join(BASE_DIR, "data", "test_right_swipe"),
        "none":  os.path.join(BASE_DIR, "data", "test_no_swipe"),
    }

    matrix, results = evaluate(hmm_left, hmm_right, hmm_none, test_folders)
    print_accuracy(matrix)
    plot_confusion_matrix(matrix)
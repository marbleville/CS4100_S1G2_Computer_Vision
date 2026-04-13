import os
import cv2
#import threading
#from pynput import keyboard
from scipy.special import softmax
import numpy as np
from hmm import HMM
from train import load_hmm
from features import process_frame, get_centroid, discretize, verify_hand

N_STATES = 5
N_BINS = 10
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_MAP = {"right" : "right_swipe", "left" : "left_swipe", "none" : None}

def get_hmms():
    hmm_right = HMM(N_STATES, N_BINS)
    hmm_left  = HMM(N_STATES, N_BINS)
    hmm_none  = HMM(N_STATES, N_BINS)

    hmm_right = load_hmm(hmm_right, "right", BASE_DIR)
    hmm_left  = load_hmm(hmm_left,  "left",  BASE_DIR)
    hmm_none  = load_hmm(hmm_none,  "none",  BASE_DIR)
    return hmm_right, hmm_left, hmm_none

def classify(obs_window, hmm_right, hmm_left, hmm_none):
    if len(obs_window) < 3:
        return None, 0.0  
    log_probs = np.array([
        hmm_right.forward(obs_window)[0],
        hmm_left.forward(obs_window)[0],
        hmm_none.forward(obs_window)[0]
    ])

    probs = softmax(log_probs)
    i = np.argmax(probs)
    labels = ["right", "left", "none"]
    return labels[i], float(probs[i])

'''
def run_webcam(hmm_right, hmm_left, hmm_none, n_bins=10, window_size=20):
    cap = cv2.VideoCapture(0)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=25, detectShadows=False)
    kernel = np.ones((10, 10), np.uint8)
    obs_buffer = []
    no_hand_count = 0

    stop = threading.Event()

    def on_press(key):
        if key == keyboard.Key.space:
            stop.set()
            return False
        
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = process_frame(frame, subtractor, kernel)
        if not verify_hand(frame, mask):
            no_hand_count += 1
            if no_hand_count > 10:
                obs_buffer = []
            continue

        no_hand_count = 0
        centroid = get_centroid(mask)
        if centroid is None:
            continue

        width = frame.shape[1]
        obs_buffer.append(discretize(centroid[0], width, n_bins))

        if len(obs_buffer) > window_size:
            obs_buffer.pop(0)

        gesture, confidence = classify(obs_buffer, hmm_right, hmm_left, hmm_none)

        cv2.putText(frame, f"Gesture: {gesture}, {confidence:.2f}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Gesture Classifier", frame)

        if cv2.waitKey(5) & 0xFF == ord(' ') or stop.is_set():
            break

    cap.release()
    cv2.destroyAllWindows()
    listener.stop()
'''

class GestureClassifier:
    def __init__(self, window_size=20):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        hmm_left  = load_hmm(HMM(5, 10), "left",  BASE_DIR)
        hmm_right = load_hmm(HMM(5, 10), "right", BASE_DIR)
        hmm_none  = load_hmm(HMM(5, 10), "none",  BASE_DIR)

        self.hmm_left    = hmm_left
        self.hmm_right   = hmm_right
        self.hmm_none    = hmm_none
        self.n_bins      = 10
        self.window_size = window_size
        self.subtractor  = cv2.createBackgroundSubtractorMOG2(
                               history=10, varThreshold=25, detectShadows=False)
        self.kernel      = np.ones((10, 10), np.uint8)
        self.obs_buffer  = []
        self.no_hand_count = 0

    def predict(self, frame):
        mask = process_frame(frame, self.subtractor, self.kernel)

        if not verify_hand(frame, mask):
            self.no_hand_count += 1
            if self.no_hand_count > 10:
                self.obs_buffer = []
            return None, None

        self.no_hand_count = 0
        centroid = get_centroid(mask)
        if centroid is None:
            return None, None

        width = frame.shape[1]
        self.obs_buffer.append(discretize(centroid[0], width, self.n_bins))

        if len(self.obs_buffer) > self.window_size:
            self.obs_buffer.pop(0)

        result, confidence = classify(self.obs_buffer, self.hmm_left, self.hmm_right, self.hmm_none)
        if result is None:
            return None, 0.0
        return RESULT_MAP[result], confidence


if __name__ == "__main__":
    WINDOW_SIZE = 20

    hmm_left  = load_hmm(HMM(N_STATES, N_BINS), "left",  BASE_DIR)
    hmm_right = load_hmm(HMM(N_STATES, N_BINS), "right", BASE_DIR)
    hmm_none  = load_hmm(HMM(N_STATES, N_BINS), "none",  BASE_DIR)

    #run_webcam(hmm_left, hmm_right, hmm_none, N_BINS, WINDOW_SIZE)
        
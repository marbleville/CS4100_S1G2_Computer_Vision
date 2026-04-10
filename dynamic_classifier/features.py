import cv2
import numpy as np
import os
import csv

def load_video_frames(path, frame_skip=2):
    cap = cv2.VideoCapture(path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def extract_hand_mask(frames):
    subtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=25, detectShadows=False)
    kernel = np.ones((10, 10), np.uint8)
    masks = []

    for frame in frames[:5]:
        subtractor.apply(frame)
    for frame in frames:
        mask = subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        masks.append(mask)

    return masks

def process_frame(frame, subtractor, kernel):
    mask = subtractor.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def verify_hand(frame, mask):
    # TODO: Use preprocessor to check if detected hand area overlaps with moving frames
    return True

def get_centroid(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def discretize(cx, frame_width, n_bins=10):
    return min(int((cx / frame_width) * n_bins), n_bins - 1)

def video_to_obs_sequence(path, n_bins=10):
    frames = load_video_frames(path)
    masks = extract_hand_mask(frames)
    width = frames[0].shape[1]
    obs = []

    for frame, mask in zip(frames, masks):
        if not verify_hand(frame, mask):
            continue
        centroid = get_centroid(mask)
        if centroid is None:
            continue
        obs.append(discretize(centroid[0], width, n_bins))

    if len(obs) < 5:
        return []
    return obs

def process_folder(folder_path, output_csv, n_bins=10):
    with open(output_csv, mode='w', newline='') as out_file:
        obs_list = []
        for file in os.listdir(folder_path):
            filename = os.path.join(folder_path, file)
            if (filename.endswith(".mp4") or filename.endswith(".mov")):
                obs = video_to_obs_sequence(filename, n_bins)
                if (len(obs) > 0):
                    obs_list.append(obs)
        writer = csv.writer(out_file)
        writer.writerows(obs_list)


if __name__ == "main":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    process_folder(os.path.join(BASE_DIR, "data", "right_swipe"), 
                os.path.join(BASE_DIR, "dynamic_classifier", "right.csv"))
    process_folder(os.path.join(BASE_DIR, "data", "left_swipe"),  
                os.path.join(BASE_DIR, "dynamic_classifier", "left.csv"))
    process_folder(os.path.join(BASE_DIR, "data", "no_swipe"),  
                os.path.join(BASE_DIR, "dynamic_classifier", "none.csv"))
    


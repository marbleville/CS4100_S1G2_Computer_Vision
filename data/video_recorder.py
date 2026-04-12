import cv2
from pynput import keyboard
import os
import threading

# find an unused file name and join it to the path
def get_save_path(folder):
    i = 1
    while True:
        path = os.path.join(folder, f'video_{i:03d}.mp4')
        if not os.path.exists(path):
            return path
        i += 1

# record a new video
def record_video(recording_dir):
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(get_save_path(recording_dir), 
                            cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    
    stop = threading.Event()

    def on_press(key):
        if key == keyboard.Key.space:
            stop.set()
            return False
        
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while True:
        ret, frame = cap.read()
        writer.write(frame)
        cv2.imshow('Recording', frame)
        cv2.setWindowProperty('Recording', cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(5) & 0xFF == ord(' ') or stop.is_set():
            break

    cap.release()
    writer.release()
    listener.stop()
    cv2.destroyAllWindows()

# prompt user for which type of video they're recording
def get_folder():
    print("Enter video type selection\n 0 - swipe right\n 1 - swipe left\n 2 - no swipe")
    while True:
        entry = input()
        match entry:
            case "0":
                return 'test_right_swipe'
            case "1":
                return 'test_left_swipe'
            case "2":
                return 'test_no_swipe'
        print("Command not recognized, try again")


folder_name = get_folder()
recording_dir = os.path.join('data', folder_name)
if not os.path.exists(recording_dir):
    os.makedirs(recording_dir)

while True:
    print("Press \'Enter\' to start recording and \'Spacebar\' to stop")
    print("Enter q to quit")
    cmd = input().strip()
    if cmd == "":
        record_video(recording_dir)
    elif cmd == "q":
        break
    else:
        print("Command not recognized")


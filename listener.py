"""Keyboard listener script — runs on your laptop.

Receives action names from the Pi over a local network socket
and presses the corresponding YouTube keyboard shortcuts.

Usage:
    python listener.py

Leave this running on your laptop while the Pi runs main.py.
Find your laptop's local IP with:
    Windows:  ipconfig
    Mac/Linux: ifconfig or ip a
Then set LAPTOP_IP in main.py on the Pi to that address.
"""

import socket
from command_engine.engine import ACTION_TO_KEY

HOST = "0.0.0.0"  # listen on all interfaces
PORT = 5005


def handle_action(action: str) -> None:
    import pyautogui
    pyautogui.PAUSE = 0
    key = ACTION_TO_KEY.get(action)
    if key is None:
        print(f"Unknown action: {action}")
        return
    if isinstance(key, tuple):
        pyautogui.hotkey(*key)
    else:
        pyautogui.press(key)
    print(f"Pressed: {action} -> {key}")


def main() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Listening for Pi on port {PORT}. Press Ctrl+C to stop.")

    try:
        while True:
            conn, addr = server.accept()
            print(f"Pi connected from {addr}")
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("Pi disconnected.")
                        break
                    action = data.decode().strip()
                    handle_action(action)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
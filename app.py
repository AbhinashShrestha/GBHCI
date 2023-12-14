from pynput import keyboard
import pyautogui
import os
from datetime import datetime

# Define your functions here
def Brightness_Decrease():
    pass

def Brightness_Increase():
    pass

def Chrome_Open():
    pass

def Cursor_Movement():
    pass

def Double_Click():
    pass

def Initiation():
    pass

def Left_Click():
    pass

def Neutral():
    pass

def Nothing():
    pass

def Right_Click():
    pass

def Screenshot():
    # Get the path to the desktop
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the filename
    filename = f"screenshot_{now_str}.png"

    # Take a screenshot
    im1 = pyautogui.screenshot()

    # Save the screenshot on the desktop
    im1.save(os.path.join(desktop, filename))

def Scroll():
    pass

def Shutdown():
    pass

def Volume_Decrease():
    pass

def Volume_Increase():
    pass

# Map keys to functions
key_to_function = {
    'a': Brightness_Decrease,
    'b': Brightness_Increase,
    'c': Chrome_Open,
    'd': Cursor_Movement,
    'e': Double_Click,
    'f': Initiation,
    'g': Left_Click,
    'h': Neutral,
    'i': Nothing,
    'j': Right_Click,
    'k': Screenshot,
    'l': Scroll,
    'm': Shutdown,
    'n': Volume_Decrease,
    'o': Volume_Increase
}

# Define the key listener
def on_press(key):
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if k in key_to_function:
        # If the key is in the dictionary, call the corresponding function
        key_to_functionk

# Start the key listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

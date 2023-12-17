import pyautogui
import os
from datetime import datetime
import screen_brightness_control as sbc
# Define your functions here
import monitorcontrol


def Brightness_Increase(increase_value):
    monitors = monitorcontrol.get_monitors()
    for monitor in monitors:
        with monitor:
            current_luminance = monitor.get_luminance()
            monitor.set_luminance(current_luminance + increase_value)
    print("Brightness Increased")

def Brightness_Decrease(decrease_value):
    monitors = monitorcontrol.get_monitors()
    for monitor in monitors:
        with monitor:
            current_luminance = monitor.get_luminance()
            monitor.set_luminance(current_luminance - decrease_value)
    print("Brightness Decreased")


def Chrome_Open():
    print("Chrome Opened")

def Cursor_Movement():
    print("Cursor Moved")

def Double_Click():
    pyautogui.click(clicks=3)
    print("Double Clicked")

def Initiation():
    print("Initiated")

def Left_Click():
    pyautogui.click(button='left')  
    print("Left Clicked")
    
def Neutral():
    print("Neutral")

def Nothing():
    print("Nothing")

def Right_Click():
    pyautogui.click(button='right') 
    print("Right Clicked")

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

    # Print a message
    print("A screenshot was taken and saved on the desktop.")

def Scroll():
    pyautogui.scroll(1) 
    print("Scrolled")

def Shutdown():
    print("Shutdown")

def Volume_Decrease():
    print("Volume Decreased")

def Volume_Increase():
    print("Volume Increased")

while True:
    # Get the user input for the key
    user_key = input("Please enter a key (or 'q' to quit):  ")

    if user_key == 'q':
        break
    elif user_key == 'a':
        Brightness_Decrease(10)
    elif user_key == 'b':
        Brightness_Increase(10)
    elif user_key == 'c':
        Chrome_Open()
    elif user_key == 'd':
        Cursor_Movement()
    elif user_key == 'e':
        Double_Click()
    elif user_key == 'f':
        Initiation()
    elif user_key == 'g':
        Left_Click()
    elif user_key == 'h':
        Neutral()
    elif user_key == 'i':
        Nothing()
    elif user_key == 'j':
        Right_Click()
    elif user_key == 'k':
        Screenshot()
    elif user_key == 'l':
        Scroll()
    elif user_key == 'm':
        Shutdown()
    elif user_key == 'n':
        Volume_Decrease()
    elif user_key == 'o':
        Volume_Increase()
    
    # Purge the user_key
    user_key = None
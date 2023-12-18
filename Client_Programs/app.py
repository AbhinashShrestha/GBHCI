import pyautogui
import os
from datetime import datetime
# import screen_brightness_control as sbc
# Define your functions here
import subprocess
def Brightness_Increase():
    #for mac
    command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 144', '-e', 'end tell']
    subprocess.run(command)
    #for windows
    # monitors = monitorcontrol.get_monitors()
    # for monitor in monitors:
    #     with monitor:
    #         current_luminance = monitor.get_luminance()
    #         monitor.set_luminance(current_luminance + increase_value)
    print("Brightness Increased")

def Brightness_Decrease(decrease_value):
    command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 145', '-e', 'end tell']
    subprocess.run(command)
    # monitors = monitorcontrol.get_monitors()
    # for monitor in monitors:
    #     with monitor:
    #         current_luminance = monitor.get_luminance()
    #         monitor.set_luminance(current_luminance - decrease_value)
    print("Brightness Decreased")


def Chrome_Open():
    print("Chrome Opened")

def Cursor_Movement():
    print("Cursor Moved")

def Double_Click():
    pyautogui.click(clicks=2, interval=0)
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
    #for mac os
  # Get the current date and time
    now = datetime.now()
    # Format the date and time string to be used in the filename
    dt_string = now.strftime("Screen Shot %Y-%m-%d at %I.%M.%S %p")
    # Get the path to the desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    # Define the command
    command = ['screencapture', os.path.join(desktop_path, f'{dt_string}.png')]
    # Run the command
    subprocess.run(command)
    
    
    #for windows
    # # Get the path to the desktop
    # desktop = os.path.join(os.path.expanduser("~"), "Desktop")

    # # Get the current date and time
    # now = datetime.now()

    # # Format the date and time as a string
    # now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # # Create the filename
    # filename = f"screenshot_{now_str}.png"

    # # Take a screenshot
    # im1 = pyautogui.screenshot()

    # # Save the screenshot on the desktop
    # im1.save(os.path.join(desktop, filename))

    # Print a message
    print("A screenshot was taken and saved on the desktop.")

def Scroll():
    pyautogui.scroll(1) 
    print("Scrolled")

def Shutdown():
    print("Shutdown")

def Volume_Increase():
    # Increase volume
    subprocess.call(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])
    print("Volume Increased")

def Volume_Decrease():
    # Decrease volume
    subprocess.call(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])
    print("Volume Decreased")
    
    
while True:
    # Get the user input for the key
    user_key = input("Please enter a key (or 'q' to quit):  ")

    if user_key == 'q':
        break
    elif user_key == 'a':
        Brightness_Decrease()
    elif user_key == 'b':
        Brightness_Increase()
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


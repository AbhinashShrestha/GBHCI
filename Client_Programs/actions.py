import pyautogui
import os
from datetime import datetime
import subprocess
import keyboard
class ActionHandler:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class

    def execute_action(self):
        if self.predicted_class == "Brightness_Increase":
            self.Brightness_Increase()
        elif self.predicted_class == "Brightness_Decrease":
            self.Brightness_Decrease()
        elif self.predicted_class == "Chrome_Open":
            self.Chrome_Open()
        elif self.predicted_class == "Cursor_Movement":
            self.Cursor_Movement()
        elif self.predicted_class == "Double_Click":
            self.Double_Click()
        elif self.predicted_class == "Initiation":
            self.Initiation()
        elif self.predicted_class == "Left_Click":
            self.Left_Click()
        elif self.predicted_class == "Neutral":
            self.Neutral()
        elif self.predicted_class == "Nothing":
            self.Nothing()
        elif self.predicted_class == "Right_Click":
            self.Right_Click()
        elif self.predicted_class == "Screenshot":
            self.Screenshot()
        elif self.predicted_class == "Scroll":
            self.Scroll()
        elif self.predicted_class == "Shutdown":
            self.Shutdown()
        elif self.predicted_class == "Volume_Increase":
            self.Volume_Increase()
        elif self.predicted_class == "Volume_Decrease":
            self.Volume_Decrease()
        else:
            print("Invalid predicted class")

    def Brightness_Increase(self):
        #for mac
        command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 144', '-e', 'end tell']
        subprocess.run(command)
        print("Brightness Increased")

    def Brightness_Decrease(self):
        command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 145', '-e', 'end tell']
        subprocess.run(command)
        print("Brightness Decreased")


    def Chrome_Open(self):
        print("Chrome Opened")

    def Cursor_Movement(self):
        print("Cursor Moved")

    def Double_Click(self):
        pyautogui.click(clicks=2, interval=0)
        print("Double Clicked")

    def Initiation(self):
        print("Initiated")

    def Left_Click(self):
        pyautogui.click(button='left')  
        print("Left Clicked")
        
    def Neutral(self):
        print("Neutral")

    def Nothing(self):
        print("Nothing")

    def Right_Click(self):
        pyautogui.click(button='right') 
        print("Right Clicked")

    def Screenshot(self):
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

    def Scroll(self):
        pyautogui.scroll(1) 
        print("Scrolled")

    def Shutdown(self):
        print("Shutdown")

    def Volume_Increase(self):
        #require cliclick to be installed
        subprocess.call(["cliclick", "kp:volume-up", "kp:volume-up"]) #volume increase by two clicks
        # Increase volume
        #native function for mac only
        # subprocess.call(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])
        # print("Volume Increased")

    def Volume_Decrease(self):
        # Decrease volume
        #require cliclick to be installed
        subprocess.call(["cliclick", "kp:volume-down", "kp:volume-down"]) #volume decrease by two clicks
        
        #native function for mac only
        # subprocess.call(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])
        # print("Volume Decreased")
        
        
# Example usage:
predicted_class = "Screenshot"  # Replace this with the actual predicted class
handler = ActionHandler(predicted_class)
handler.execute_action()
import pyautogui
import os
from datetime import datetime
import subprocess
import logging
import platform
class ActionHandler:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.logger = self.setup_logger()
    
    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Create a file handler and set the formatter
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "action_handler.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def execute_action(self):
        try:
            print(f"Predicted class: {self.predicted_class}")
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
            elif self.predicted_class == "Play":
                self.Play()
            elif self.predicted_class == "Pause":
                self.Pause()
            elif self.predicted_class == "PowerPoint_Open":
                self.PowerPoint_Open()
            else:
                self.logger.warning("Invalid predicted class: %s", self.predicted_class)
        except Exception as e:
            self.logger.error("Error executing action: %s", e)
            print("An error occurred while executing the action.")


    def Brightness_Increase(self):
        try:
            #for mac
            command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 144', '-e', 'end tell']
            subprocess.run(command)
            self.logger.info("Brightness Increased")
        except Exception as e:
            self.logger.error("Error increasing brightness: %s", e)

    def Brightness_Decrease(self):
        try:
            command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 145', '-e', 'end tell']
            subprocess.run(command)
            self.logger.info("Brightness Decreased")
        except Exception as e:
            self.logger.error("Error decreasing brightness: %s", e)

    # def Chrome_Open(self):
    #     subprocess.run(["open", "-a", "Google Chrome"])
    #     self.logger.info("Chrome Opened")
    def Chrome_Open(self):
        if platform.system() == "Windows":
            subprocess.run(["powershell", "-Command", "Start-Process chrome"])
        elif platform.system() == "Darwin":  # Darwin is the OS name for Mac
            subprocess.run(["open", "-a", "Google Chrome"])
        else:
            self.logger.error("Unsupported platform: %s", platform.system())
            return
        self.logger.info("Chrome Opened")

    def Cursor_Movement(self):
        self.logger.info("Cursor Moved")

    def Double_Click(self):
        try:
            # pyautogui.click(clicks=2, interval=0)
            command = 'cliclick tc:.'
            process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            self.logger.info("Double Clicked")
        except Exception as e:
            self.logger.error("Error performing double click: %s", e)

    def Initiation(self):
        self.logger.info("Initiated")

    def Left_Click(self):
        try:
            pyautogui.click(button='left')
            self.logger.info("Left Clicked")
        except Exception as e:
            self.logger.error("Error performing left click: %s", e)
        
    def Neutral(self):
        self.logger.info("Neutral")

    def Nothing(self):
       self.logger.info("Nothing")

    def Right_Click(self):
        try:
            pyautogui.click(button='right')
            self.logger.info("Right Clicked")
        except Exception as e:
            self.logger.error("Error performing right click: %s", e)

    def Screenshot(self):
        try:
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
            self.logger.info("Screenshot taken and saved on the desktop.")
        except Exception as e:
            self.logger.error("Error taking screenshot: %s", e)
            print("An error occurred while taking a screenshot.")

    def Scroll(self):
        try:
            pyautogui.scroll(1)
            self.logger.info("Scrolled")
        except Exception as e:
            self.logger.error("Error scrolling: %s", e)

    def Shutdown(self):
        os.system("sudo -S shutdown -h now")
        self.logger.info("Shutdown")

        
    def Volume_Increase(self):
        try:
            # Check if the system is Windows
            if platform.system() == 'Windows':
                # Define the PowerShell script
                ps_script = "$obj = new-object -com wscript.shell; $obj.SendKeys([char]175)"
                
                # Run the PowerShell script
                os.system('powershell.exe -Command "' + ps_script + '"')
            else:
                # Increase volume for non-Windows systems
                # require cliclick to be installed
                os.system("cliclick kp:volume-up kp:volume-up") # volume increase by two clicks
                
                # native function for mac only
                # os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")
            self.logger.info("Volume Increased")
        except Exception as e:
            self.logger.error("Error increasing volume: %s", e)


    def Volume_Decrease(self):
        try:
            # Check if the system is Windows
            if platform.system() == 'Windows':
                # Define the PowerShell script
                ps_script = "$obj = new-object -com wscript.shell; $obj.SendKeys([char]174)"
                
                # Run the PowerShell script
                os.system('powershell.exe -Command "' + ps_script + '"')
            else:
                # Decrease volume for non-Windows systems
                # require cliclick to be installed
                os.system("cliclick kp:volume-down kp:volume-down") # volume decrease by two clicks
                
                # native function for mac only it wont show the volume change indicator
                # os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")
                # print("Volume Decreased")
        except Exception as e:
            self.logger.error("Error decreasing volume: %s", e)
        
    
    def Play(self):
        pass
    
    
    def Pause(self):
        pass

    
    def PowerPoint_Open(self):
        try:
            # Check if the system is Windows
            if platform.system() == "Windows":
                # Define the PowerShell command
                ps_command = 'Start-Process "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\PowerPoint.lnk"'
                
                # Run the PowerShell command
                subprocess.Popen(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
            else:
                self.logger.error("Unsupported platform: %s", platform.system())
                return
            self.logger.info("PowerPoint Opened")
        except Exception as e:
            self.logger.error("Error opening PowerPoint: %s", e)

            
        
# Example usage:
predicted_class = "PowerPoint_Open"  # Replace this with the actual predicted class
handler = ActionHandler(predicted_class)
handler.execute_action()
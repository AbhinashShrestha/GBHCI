import pyautogui
import os
from datetime import datetime
import subprocess
import logging
import screen_brightness_control as sbc
from PIL import ImageGrab
import platform
import tkinter as tk
from tkinter import messagebox

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
            elif self.predicted_class == "VSCode_Open":
                self.VSCode_Open()
            elif self.predicted_class == "Left_Click":
                self.Left_Click()
            elif self.predicted_class == "Anomaly":
                self.Anomaly()
            elif self.predicted_class == "Right_Click":
                self.Right_Click()
            elif self.predicted_class == "Screenshot":
                self.Screenshot()
            elif self.predicted_class == "Scroll_Up":
                self.Scroll_Up()
            elif self.predicted_class == "Scroll_Down":
                self.Scroll_Down()
            elif self.predicted_class == "Shutdown":
                self.Shutdown()
            elif self.predicted_class == "Restart":
                self.Restart()
            elif self.predicted_class == "Volume_Increase":
                self.Volume_Increase()
            elif self.predicted_class == "Volume_Decrease":
                self.Volume_Decrease()
            elif self.predicted_class == "PowerPoint_Open":
                self.PowerPoint_Open()
            else:
                self.logger.warning("Invalid predicted class: %s", self.predicted_class)
        except Exception as e:
            self.logger.error("Error executing action: %s", e)
            print("An error occurred while executing the action.")

    def Brightness_Increase(self):
        try:
            if platform.system() == 'Darwin':  # for mac
                command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 144', '-e', 'end tell']
                subprocess.run(command)
                self.logger.info("Brightness Increased")
            elif platform.system() == 'Windows':  # for windows
                # get the brightness
                brightness = sbc.get_brightness()

                # increase the brightness for all displays
                new_brightness = [min(b + 5, 100) for b in brightness]

                # calculate the average brightness
                avg_brightness = int(sum(new_brightness) / len(new_brightness))

                # set the new brightness
                sbc.set_brightness(avg_brightness)
                print(avg_brightness)

                # show the current brightness for each detected monitor
                for monitor in sbc.list_monitors():
                    self.logger.info(f"{monitor} : {sbc.get_brightness(display=monitor)} %")
                
        except Exception as e:
            self.logger.error("Error increasing brightness: %s", e)

    # def Brightness_Increase(self):
    #     try:
    #         if platform.system() == 'Darwin':  # for mac
    #             command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 144', '-e', 'end tell']
    #             subprocess.run(command)
    #             self.logger.info("Brightness Increased")
    #         elif platform.system() == 'Windows':  # for windows
    #             # Define the PowerShell script
    #             ps_script = "Get-CimInstance -Namespace root/WMI -Classname WmiMonitorBrightnessMethods | Invoke-CimMethod -Methodname WmiSetBrightness -Argument @{ Timeout = 0; Brightness = 50}"

                
    #             # Run the PowerShell script
    #             os.system('powershell.exe -Command "' + ps_script + '"')
    #             self.logger.info("Brightness Increased")
    #     except Exception as e:
    #         self.logger.error("Error increasing brightness: %s", e)
            
    def Brightness_Decrease(self):
        try:
            if platform.system() == 'Darwin':  # for mac
                command = ['osascript', '-e', 'tell application "System Events"', '-e', 'key code 145', '-e', 'end tell']
                subprocess.run(command)
                self.logger.info("Brightness Decreased")
            elif platform.system() == 'Windows':  # for windows
                # get the brightness
                brightness = sbc.get_brightness()

                # decrease the brightness for all displays
                new_brightness = [max(b - 5, 5) for b in brightness]

                # calculate the average brightness
                avg_brightness = int(sum(new_brightness) / len(new_brightness))

                # set the new brightness
                sbc.set_brightness(avg_brightness)
                print(avg_brightness)

                # show the current brightness for each detected monitor
                for monitor in sbc.list_monitors():
                    self.logger.info(f"{monitor} : {sbc.get_brightness(display=monitor)} %")
        except Exception as e:
            self.logger.error("Error decreasing brightness: %s", e)

    # def Chrome_Open(self):
    #     if platform.system() == "Windows":
    #         subprocess.run(["powershell", "-Command", "Start-Process chrome"])
    #     elif platform.system() == "Darwin":  # Darwin is the OS name for Mac
    #         subprocess.run(["open", "-a", "Google Chrome"])
    #     else:
    #         self.logger.error("Unsupported platform: %s", platform.system())
    #         return
    #     self.logger.info("Chrome Opened")
    def Chrome_Open(self):
        try:
            if platform.system() == "Windows":
                # # Define the PowerShell command
                # ps_command = 'Start-Process "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"'
                # # Run the PowerShell command
                # subprocess.Popen(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
                subprocess.run(["powershell", "-Command", "Start-Process chrome"])
            elif platform.system() == "Darwin":  # Darwin is the OS name for Mac
                subprocess.run(["open", "-a", "Google Chrome"])
                self.logger.info("Google Chrome Opened")
            else:
                self.logger.error("Unsupported platform: %s", platform.system())
                return
        except (subprocess.CalledProcessError, OSError):
            # If Google Chrome can't be opened, show a message box
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showinfo("Error", "Google Chrome could not be opened.")
            root.destroy()  # Destroy the main window
            self.logger.error("Google Chrome could not be opened.")


    def Cursor_Movement(self):
        self.logger.info("Cursor Moved")
        
    def Double_Click(self):
        try:
            # Check if the system is Windows
            if platform.system() == 'Windows':
                # Use pyautogui to perform double click
                pyautogui.doubleClick()
            else:
                # Perform double click for MacOS
                # require cliclick to be installed
                command = 'cliclick tc:.'
                process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            self.logger.info("Double Clicked")
        except Exception as e:
            self.logger.error("Error performing double click: %s", e)

    def VSCode_Open(self):
        try:
            if platform.system() == "Windows":
            # Define the PowerShell command
                # ps_command = 'Start-Process "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Visual Studio Code\\Visual Studio Code.lnk"'
                ps_command = 'code'                
                # Run the PowerShell command
                subprocess.Popen(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
            elif platform.system() == "Darwin":  # Darwin is the OS name for Mac
                subprocess.run(["open", "-a", "Zed"])
                self.logger.info("VSCode Opened")
            else:
                self.logger.error("Unsupported platform: %s", platform.system())
                return
        except (subprocess.CalledProcessError, OSError):
            # If VSCode can't be opened, show a message box
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showinfo("Error", "VSCode could not be opened.")
            root.destroy()  # Destroy the main window
            self.logger.error("VSCode could not be opened.")

    def Left_Click(self):
        try:
            pyautogui.click(button='left')
            self.logger.info("Left Clicked")
        except Exception as e:
            self.logger.error("Error performing left click: %s", e)
        
    def Anomaly(self):
        self.logger.info("Anomaly")

    def Right_Click(self):
        try:
            pyautogui.rightClick()
            self.logger.info("Right Clicked")
        except Exception as e:
            self.logger.error("Error performing right click: %s", e)

    def Screenshot(self):
        try:
            # Get the current date and time
            now = datetime.now()

            # Check the platform
            if platform.system() == 'Darwin':
                # macOS
                # Format the date and time string to be used in the filename
                dt_string = now.strftime("Screen Shot %Y-%m-%d at %I.%M.%S %p")
                # Define the path to save screenshots
                screenshot_path = os.path.join(os.path.expanduser("~"), "Desktop", "Screenshots")
                # Create the directory if it doesn't exist
                os.makedirs(screenshot_path, exist_ok=True)
                # Define the command
                command = ['screencapture', os.path.join(screenshot_path, f'{dt_string}.png')]
                # Run the command
                subprocess.run(command)
                # Open the folder containing the screenshot
                subprocess.run(['open', screenshot_path])
            elif platform.system() == 'Windows':
                # Windows
                # Format the date and time string to be used in the filename
                dt_string = now.strftime("%Y-%m-%d_%I.%M.%S_%p")
                # Define the path to save screenshots
                screenshot_path = r"C:\Users\Anon\Desktop\Screenshots"
                # Create the directory if it doesn't exist
                os.makedirs(screenshot_path, exist_ok=True)
                # Define the filename
                filename = os.path.join(screenshot_path, f"{dt_string}.png")
                # Capture the entire screen
                screenshot = ImageGrab.grab()
                # Save the screenshot to a file
                screenshot.save(filename)
                # Close the screenshot
                screenshot.close()
                # Open the folder containing the screenshot
                os.startfile(screenshot_path)
            else:
                print("Unsupported platform.")
                return

            self.logger.info("Screenshot taken and saved.")
        except Exception as e:
            self.logger.error("Error taking screenshot: %s", e)
            print("An error occurred while taking a screenshot.")

    def Scroll_Up(self):
        try:
            pyautogui.scroll(1000)
            self.logger.info("Scrolled Up")
        except Exception as e:
            self.logger.error("Error scrolling: %s", e)

    def Scroll_Down(self):
        try:
            pyautogui.scroll(-1000)
            self.logger.info("Scrolled Down")
        except Exception as e:
            self.logger.error("Error scrolling: %s", e)

    def Shutdown(self):
        # try:
        #     if platform.system() == "Darwin":
        #         os.system("sudo -S shutdown -h now")
        #     elif platform.system() == "Windows":
        #         os.system("shutdown /s")
        #     else:
        #         self.logger.error("Unsupported operating system.")
        #         return
        #     self.logger.info("Shutdown")
        # except Exception as e:
        #     self.logger.error(f"An error occurred: {e}")
        pass

    def Restart(self):
        # try:
        #     if platform.system() == "Darwin":
        #         os.system("sudo -S shutdown -r now")
        #     elif platform.system() == "Windows":
        #         os.system("shutdown /r /t 1")
        #     else:
        #         self.logger.error("Unsupported operating system.")
        #         return
        #     self.logger.info("Restart")
        # except Exception as e:
        #     self.logger.error(f"An error occurred: {e}")
        pass
            
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

    def PowerPoint_Open(self):
        try:
            # Check if the system is Windows
            if platform.system() == "Windows":
                # # Define the PowerShell command
                ps_command = 'Start-Process "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\PowerPoint.lnk"'
                # Run the PowerShell command
                subprocess.Popen(["powershell", "-Command", ps_command], stdout=subprocess.PIPE)
            elif platform.system() == "Darwin":  # Darwin is the OS name for Mac
                subprocess.run(["open", "-a", "Microsoft PowerPoint"])
                self.logger.info("VSCode Opened")
            else:
                self.logger.error("Unsupported platform: %s", platform.system())
                return
            self.logger.info("PowerPoint Opened")
        except Exception as e:
            self.logger.error("Error opening PowerPoint: %s", e)


# Example usage:
# predicted_class = "Chrome_Open"  # Replace this with the actual predicted class
# handler = ActionHandler(predicted_class)
# handler.execute_action()
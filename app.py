import pyautogui

# Define your mapping
activity_mapping = {
    'a': 'Brightness_Decrease',
    'b': 'Brightness_Increase',
    'c': 'Chrome_Open',
    'd': 'Cursor_Movement',
    'e': 'Double_Click',
    'f': 'Initiation',
    'g': 'Left_Click',
    'h': 'Neutral',
    'i': 'Nothing',
    'j': 'Right_Click',
    'k': 'Screenshot',
    'l': 'Scroll',
    'm': 'Shutdown',
    'n': 'Volume_Decrease',
    'o': 'Volume_Increase'
}

# Get user input
user_input = input("Enter an alphabet: ")

# Perform the corresponding action
if user_input in activity_mapping:
    action = activity_mapping[user_input]
    
    if action == 'Cursor_Movement':
        pyautogui.move(100, 100, duration=1)  # Move the cursor
    elif action == 'Left_Click':
        pyautogui.click(button='left')  # Left click
    elif action == 'Right_Click':
        pyautogui.click(button='right')  # Right click
    elif action == 'Double_Click':
        pyautogui.doubleClick()  # Double click
    elif action == 'Screenshot':
        screenshot = pyautogui.screenshot()  # Take a screenshot
    elif action == 'Scroll':
        pyautogui.scroll(10)  # Scroll up 10 "clicks"
    # Add more elif conditions here for other actions
    else:
        print(f"{action} is not implemented in this example.")
else:
    print("Invalid input. Please enter a valid alphabet.")

import pyautogui
from pynput import mouse
import winsound

# Initialize a list to store the coordinates and colors
coordinates = []

# Ask how many coordinates should be saved
coordinate_length = int(input("How many coordinates should be saved?\n"))
print('Click on the coordinates you''d like to save\n')
# Adjust for scaled screen size
# scaling = 2.5

def on_click(x, y, button, pressed):
    # When a button is pressed, add the coordinates and color to the list
    if pressed:
        # Get the color at the clicked location
        color = pyautogui.screenshot().getpixel((x, y))
        location = pyautogui.position()
        coordinates.append((location, color))
        print(f'Position {len(coordinates)}: ({location}), Color: {color}')
        # If two positions have been recorded, stop the listener
        if len(coordinates) == coordinate_length:
            return False

# Start the mouse listener
with mouse.Listener(on_click=on_click) as listener:
    listener.join()

# Notify program has ended
winsound.Beep(660,100)
winsound.Beep(440,60)
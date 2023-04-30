'''
This is a Python code that captures live video feed from a webcam using OpenCV and performs OCR (optical character recognition) using the Tesseract library to recognize characters from the video frames. Specifically, the code recognizes capital letters (A-Z) and filters out all other characters using regular expressions.

The video feed is assumed to show a scrabble board consisting of a 10x10 grid with 9 squares per row and 7 squares per column. The program calculates the center coordinates of each recognized letter and determines whether the letter is inside or outside the scrabble board based on its position within the grid. If the letter is inside the grid, its information is appended to the list "inside_grid", otherwise it is appended to the list "outside_grid".

The code also draws bounding boxes around each recognized letter and draws the grid lines on the video frame. The resulting image is displayed in a window using the OpenCV function cv2.imshow(). The program terminates when the user presses the 'q' key on the keyboard.
'''
import cv2
import pytesseract
import re                    #The import re statement in Python is used to import the built-in module re, which stands for regular expressions. Regular expressions are a powerful tool for searching and manipulating text strings in Python. The re module provides functions that allow you to search for patterns within strings using regular expressions. It includes functions for matching specific patterns of characters, finding and replacing text, and 
#splitting strings based on certain patterns. The re module is commonly used in data processing, text mining, and web scraping applications.
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Paritosh\tesseract.exe'      # Is used to specify the path to the Tesseract executable file


cap = cv2.VideoCapture(0)

# Define the grid parameters
n_rows = 10
n_cols = 10
m = 9
n = 7

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold to convert to binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Use Tesseract to recognize characters
    text = pytesseract.image_to_string(thresh, config='--psm 11 --psm 10 --oem 3') #psm = pages segmentation mode; oem = OCR engine mode

    # Filter out non-capital letters using regular expressions.
    # not necessecary since using only capital blocks here
    #text = re.sub('[^A-Z]', '', text)

    # Print recognized text
    print(text)

    hImg, wImg, channels = frame.shape
    boxes = pytesseract.image_to_boxes(frame)

    # Defining lists that contain letters which are present inside and outside the scrabble board
    inside_grid = []
    outside_grid = []

    # The function below gives the necessary letter and the its respective x, y coordinates, coordinate of the diagonal elements
    for b in boxes.splitlines():
        b = b.split()  # This converts the above values to a list so each letter has its respective list filled with the coordinates
        if len(b) > 4:  #To check if the list contains more than just the letter and its four corresponding coordinates (x, y, w, h).
 #This could happen if there are additional whitespace characters or other characters in the string that are not part of the coordinates.
            letter, x, y, w, h = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])

            # Calculating the coordinates of the center of each letter
            center_x = x + w / 2
            center_y = hImg - (y + h / 2)

            # Check if the letter is inside the grid
            if x < n_cols * wImg / n and (x + w) > 0 and y < n_rows * hImg / m and (y + h) > 0:
                inside_grid.append((letter, center_x, center_y))
            else:
                outside_grid.append((letter, center_x, center_y))

            # Draw bounding box around each letter
            cv2.rectangle(frame, (x, hImg - y), (w, hImg - h), (0, 0, 255), 3)

    # Draw the grid on the image
    for i in range(1, n_cols):
        cv2.line(frame, (int(i * wImg / n), 0), (int(i * wImg / n), hImg), (0, 255, 0), 2)

    for i in range(1, n_rows):
        cv2.line(frame, (0, int(i * hImg / m)), (wImg, int(i * hImg / m)), (0, 255, 0), 2)

    # Show the resulting image
    cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:15:30 2023

@author: Paritosh
"""

'''
This is a Python code that captures live video feed from a webcam using OpenCV and performs OCR (optical character recognition) using the Tesseract library to recognize characters from the video frames. Specifically, the code recognizes capital letters (A-Z) and filters out all other characters using regular expressions.

The video feed is assumed to show a scrabble board consisting of a 5x5 grid with 9 squares per row and 7 squares per column. The program calculates the center coordinates of each recognized letter and determines whether the letter is inside or outside the scrabble board based on its position within the grid. If the letter is inside the grid, its information is appended to the list "inside_grid", otherwise it is appended to the list "outside_grid".

The code also draws bounding boxes around each recognized letter and draws the grid lines on the video frame. The resulting image is displayed in a window using the OpenCV function cv2.imshow(). The program terminates when the user presses the 'q' key on the keyboard.
'''
import cv2
import pytesseract
import re
import itertools
import numpy as np
import random
import time
import os,sys
import math
import serial
import serial.tools.list_ports
from pymycobot.mycobot import MyCobot
from pymycobot.genre import Angle
from pymycobot.genre import Coord
from pymycobot.mybuddy import MyBuddy
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Paritosh\tesseract.exe'

# mc is the variable that represents the cobot

mc = MyCobot("COM3",115200)

letter_map = { 0: 'A',1: 'B',2: 'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'0',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}


class Rack:
    """
    Creates each player's 'dock', or 'hand'. Allows players to add, remove and replenish the number of tiles in their hand.
    """
    def __init__(self, bag):
        #Initializes the player's rack/hand. Takes the bag from which the racks tiles will come as an argument.
        self.rack = []
        self.bag = bag
        self.initialize()
    
    def add_to_rack(self):
        #Takes a tile from the bag and adds it to the player's rack.
        self.rack.append(self.bag.take_from_bag())
    
    def initialize(self):
        #Adds the initial tiles to the player's hand.
        for i in inList:
            self.add_to_rack()
    
    def get_rack_str(self):
        #Displays the user's rack in string form.
        return ", ".join(str(item.get_letter()) for item in self.rack)
    
    def get_rack_arr(self):
        #Returns the rack as an array of tile instances
        return self.rack
    
    def remove_from_rack(self, tile):
        #Removes a tile from the rack (when a tile is being played).
        self.rack.remove(tile)
    
    def get_rack_length(self):
        #Returns the number of tiles left in the rack.
        return len(self.rack)
    
    def replenish_rack(self):
        #Adds tiles to the rack after a turn such that the rack will have 7 tiles (assuming a proper number of tiles in the bag).
        while self.get_rack_length() < 7 and self.bag.get_remaining_tiles() > 0:
            self.add_to_rack()
            
class Player:
    """
    Creates an instance of a player. Initializes the player's rack.
    """
    def __init__(self, bag):
        #Intializes a player instance. Creates the player's rack by creating an instance of that class.
        #Takes the bag as an argument, in order to create the rack.
        self.rack = Rack(bag)
        self.score = 0
    
    def get_rack_str(self):
        #Returns the player's rack.
        return self.rack.get_rack_str()
    
    def get_rack_arr(self):
        #Returns the player's rack in the form of an array.
        return self.rack.get_rack_arr()
    
    def increase_score(self, increase):
        #Increases the player's score by a certain amount. Takes the increase (int) as an argument and adds it to the score.
        self.score += increase
    
    def get_score(self):
        #Returns the player's score
        return self.score
    

def find_words(board, words):
    # Find all possible words that can be formed on the board.
    rows = len(board)
    cols = len(board[0])
    words_found = []
    
    # Find words horizontally.
    for row in range(rows):
        for col in range(cols):
            word = ""
            for i in range(col, cols):
                if board[row][i] == "":
                    break
                word += board[row][i]
                if word in words and word not in words_found:
                    words_found.append(word)
    
    # Find words vertically.
    for col in range(cols):
        for row in range(rows):
            word = ""
            for i in range(row, rows):
                if board[i][col] == "":
                    break
                word += board[i][col]
                if word in words and word not in words_found:
                    words_found.append(word)
    
    # Find words diagonally.
    for size in range(2, min(rows, cols) + 1):
        for indices in itertools.product(range(rows - size + 1), range(cols - size + 1)):
            word = ""
            for i in range(size):
                row, col = indices[i]
                if board[row][col] == "":
                    break
                word += board[row][col]
                if i == size - 1 and word in words and word not in words_found:
                    words_found.append(word)
    
    return words_found
    

class Word:
    def __init__(self, word, location, player, direction):
        self.word = word.upper()
        self.location = location
        self.player = player
        self.direction = direction.lower()
    
    def check_word(self, board, words):
        # Check the word to make sure that it is in the dictionary, and that the location falls within bounds.
        # Also controls the overlapping of words.
        global round_number, players
        word_score = 0
        global dictionary
        if "dictionary" not in globals():
            dictionary = open("dic.txt").read().splitlines()
    
        current_board_ltr = ""
        needed_tiles = ""
        blank_tile_val = ""
        word = self.word
    
        # Assuming that the player is not skipping the turn:
        if self.word != "":
    
            # Allows for players to declare the value of a blank tile.
            if "#" in self.word:
                while len(blank_tile_val) != 1:
                    blank_tile_val = input("Please enter the letter value of the blank tile: ")
                self.word = self.word[:word.index("#")] + blank_tile_val.upper() + self.word[(word.index("#") + 1):]
    
            # Raises an error if the word being played is not in the official scrabble dictionary (dic.txt).
            if self.word not in dictionary:
                return "Please enter a valid dictionary word.\n"
    
            # Find all valid words that can be formed on the board.
            words_found = find_words(board, words)
    
            # Ensures that the word being played is a valid word on the board.
            if self.word not in words_found:
                return "The word is not a valid word on the board"
    
    
    def calculate_word_score(self):
        #Calculates the score of a word, allowing for the impact by premium squares.
        global LETTER_VALUES, premium_spots
        word_score = 0
        for letter in self.word:
            for spot in premium_spots:
                if letter == spot[0]:
                    if spot[1] == "TLS":
                        word_score += LETTER_VALUES[letter] * 2
                    elif spot[1] == "DLS":
                        word_score += LETTER_VALUES[letter]
            word_score += LETTER_VALUES[letter]
        for spot in premium_spots:
            if spot[1] == "TWS":
                word_score *= 3
            elif spot[1] == "DWS":
                word_score *= 2
        self.player.increase_score(word_score)
    
    def set_word(self, word):
        self.word = word.upper()
    
    def set_location(self, location):
        self.location = location
    
    def set_direction(self, direction):
        self.direction = direction
    
    def get_word(self):
        return self.word

# Turn on the suction pump
def pump_on():
    # make position 2 work
    mc.set_basic_output(2, 0)
    # make position 5 work
    mc.set_basic_output(5, 0)


# stop the suction pump
def pump_off():
    # Stop position 2 from working
    mc.set_basic_output(2, 1)
    # Stop position 5 from working
    mc.set_basic_output(5, 1)


# function to make the cobot move

def move(list1):
    
    for i in range(len(list1)):
        mc.send_coords(list1[i],10,1)
        time.sleep(10)
        pump_on()
    pump_off()
    

#function to convert pixel coordinate system to cobots coordinate system

def convert(x,y):
    angle1 = -90
    angle2 = 180
    
    t1 = math.radians(angle1)
    t2 = math.radians(angle2)
   
    cost1 = math.cos(t1)
    sint1 = math.cos(t1)
    
    cost2 = math.cos(t2)
    sint2 = math.sin(t2)
    
    A = np.array([[1, 0 ,0 ][cost2, -sint2 ,0],[sint2, cost2 ,0]])
    B = np.array([[cost1, -sint1,0][sint2, cost2,0],[1,0,0]])
    
    C = np.dot(A, B)
    
    C = np.array([[0,-1,0],[1,0,0],[0,0,-1]])
    
    D = np.array([[0.26*x] ,[0.26*y],[1]])
    
    D = np.dot(C, D)
    
    return D[0]-112.8, D[1]+76
    

cap = cv2.VideoCapture(0)

# Define the grid parameters
n_rows = 8
n_cols = 8
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
            centers = [center_x, center_y]

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
        
        

    # Seperation of outside and inside world
    coord = (center_x, center_y)
    if coord in outside_grid:       #create a list of all the alphabets outside
        outList = []
        outList.append(text)
        #now we need to select any letter present outside the chessboard and place it into the matrix
        t = len(outside_grid)
        kalyan = True
        
        if t>0:
            r = random.randint(0,t-1)
            outletter = outside_grid[r]
            outside_grid.remove(outside_grid[r])
            letter = outletter[0]
            x_letter = outletter[1]
            y_letter = outletter[2]
            x1, y1 = convert(x_letter, y_letter)
            #now we got the coordinates from cobots frame of reference
            # now we will pick coordinate to place the letter
            
            while True:
                rows =  random.randint(0, n)
                cols = random.randint(0, n)
                if(rows,cols) not in outside_grid:
                    outside_grid.append((rows,cols))
                    break
            
            for center in centers:
                if rows == center[2] and cols == center[3] :
                    outside_grid[rows][cols] = letter
                    x_letter = center[0]
                    y_letter = center[1]
                    x2 , y2 = convert(x_letter , y_letter)
                
            #Now we got initial and final coordinates the cobot should move 
            list1 = [[x1,y1, 57.9, -177.8, -0.99, -131.16],[x2,y2, 57.9, -177.8, -0.99, -131.16]]
            
            #below function makes the cobot pick and place the letter piece
            
            move(list1)
            
                
        cv2.imshow("Frame", gray)
    # Show the resulting image
    cv2.imshow('Result', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

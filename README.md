# AI Apple Catcher

AI Apple Catcher is an interactive Python game controlled through real-time hand tracking.

The player moves a basket by moving their hand horizontally in front of a webcam. The objective is to catch falling apples, avoid bombs, collect points and progress through increasingly difficult levels.

## Project Overview

This project combines a traditional 2D game with computer vision and hand landmark detection.

Instead of using only a keyboard or mouse to control the basket, the game detects the position of the player's hand through the webcam. The detected horizontal hand position is mapped to the horizontal position of the basket.

The project demonstrates how artificial intelligence and computer vision can be integrated into an interactive application.

## Main Features

- Real-time hand tracking through a webcam
- Basket movement controlled by hand position
- Falling apples and bombs
- Score tracking
- Lives system
- Increasing game difficulty
- Start and restart buttons controlled with the mouse
- Animated game objects and visual effects

## Technologies Used

- Python
- Pygame
- OpenCV
- MediaPipe
- NumPy

## AI and Computer Vision Component

The game uses MediaPipe Hand Landmarker to detect hand landmarks from webcam frames.

MediaPipe identifies 21 landmarks on the detected hand. Several landmarks located around the palm are used to estimate the horizontal position of the hand:

## Game Rules
Catching an apple increases the score.
Catching a bomb removes one life.
The game ends when the player loses all lives.
The falling speed increases as the player progresses.
The player can restart the game by clicking the restart button

## Installation
pip install pygame opencv-python mediapipe numpy

The dependencies can also be installed using the included requirements.txt file:

pip install -r requirements.txt
```python
palm_x = sum(
    landmarks[i].x for i in (0, 5, 9, 13, 17)
) / 5

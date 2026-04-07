"""
Configuration for gesture classification and dataset handling.
"""

# List of static gesture class names 
STATIC_GESTURE_CLASSES = [
	"palm",         # play/pause (open hand)
	"thumb",        # increase volume (thumbs up) 
	"down",    		# decrease volume (point down)
	"fist",  		# mute/unmute (closed fist)
]

DYNAMIC_GESTURE_CLASSES = [
    "right_swipe" ,   # next video (swipe right)
	"left_swipe",  	  # previous video (swipe left)
]

ALL_GESTURE_CLASSES = STATIC_GESTURE_CLASSES + DYNAMIC_GESTURE_CLASSES

# Image crop size for hand region (used by classifier and dataset)
HAND_CROP_SIZE = 128

# Normalization constants 
# Replace with computed values once ran separately on actual dataset 
NORMALIZATION_MEAN = [0.485, 0.456, 0.406] # examples
NORMALIZATION_STD = [0.229, 0.224, 0.225]  # examples

# Confidence threshold for predictions 
CONFIDENCE_THRESHOLD = 0.75

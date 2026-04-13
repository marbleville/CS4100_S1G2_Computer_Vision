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
NORMALIZATION_MEAN = [0.5725, 0.5252, 0.4866]
NORMALIZATION_STD  = [0.2495, 0.2408, 0.2307]

# Confidence threshold for predictions 
CONFIDENCE_THRESHOLD = 0.75

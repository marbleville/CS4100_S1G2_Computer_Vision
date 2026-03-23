import numpy as np
import unittest
from classifier.static_classifier import StaticClassifier
from preprocessor.types.results import HandDetectionResult

class TestStaticClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = StaticClassifier()

    def test_classify_returns_hand_detection_result(self):
        # Create a dummy image (e.g., 256x256 RGB)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        result = self.classifier.classify(image)
        self.assertIsInstance(result, HandDetectionResult)
        self.assertFalse(result.hand_detected)
        self.assertEqual(result.confidence_level, 0.0)
        self.assertIsInstance(result.crop_rgb, np.ndarray)
        self.assertEqual(result.crop_rgb.shape, (128, 128, 3))
        self.assertIsNone(result.bbox)

if __name__ == "__main__":
    unittest.main()

from src.adviser import Adviser
from src.detection.detection import EmotionDetection

if __name__ == '__main__':
    emotion_detector = EmotionDetection('detection')
    emotion_detector.start_detection()

from src.adviser import Adviser
from src.detection.detection import EmotionDetection

if __name__ == '__main__':
    #emotion_detector = EmotionDetection('detection')
    #emotion_detector.start_detection()
    ad = Adviser()
    print(ad.getPrompt('sad'))
    topic = ad.predictTopic('My boss doesnt give me money')
    advice = ad.getAdvice(emotion='sad', topic=topic)
    print(advice)


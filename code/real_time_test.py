import cv2
import numpy as np
from matplotlib import pyplot as plt

from code.collect_imgs import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints, image
from code.create_dataset import custom_actions
from tensorflow.keras.models import load_model

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
res = np.array([0.3, 0.5, 0.2])

model = load_model('action.h5')

while len(colors) < len(custom_actions):
    colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, custom_actions, image, colors))

sequence = []
sentence = []
predictions = []
threshold = 0.5

# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(custom_actions[np.argmax(res)])

            # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if custom_actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(custom_actions[np.argmax(res)])
                else:
                    sentence.append(custom_actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, custom_actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
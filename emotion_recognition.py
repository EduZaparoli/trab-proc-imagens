import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time
import os
import datetime

model = load_model('modelo_emocoes_cnn.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

camera = cv2.VideoCapture(0)

emotion_counter = {emotion: 0 for emotion in emotion_labels}

emotion_over_time = []
time_stamps = []

face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

frame_counter = 0
frame_interval = 5
start_time = time.time()

if not os.path.exists('capturas'):
    os.makedirs('capturas')

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")

                if frame_counter % frame_interval == 0:
                    roi_gray = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_gray = roi_gray.astype("float32") / 255.0
                    roi_gray = np.expand_dims(img_to_array(roi_gray), axis=0)

                    emotion_prediction = model.predict(roi_gray, verbose=0)
                    max_index = np.argmax(emotion_prediction[0])
                    emotion = emotion_labels[max_index]

                    emotion_counter[emotion] += 1

                    emotion_over_time.append(emotion)
                    elapsed_time = time.time() - start_time
                    time_stamps.append(elapsed_time)

                    if emotion in ['Angry', 'Sad', 'Fear', 'Happy', 'Surprise', 'Disgust']:
                        current_time = datetime.datetime.now().strftime('%H-%M-%S')
                        filename = f"capturas/{emotion}_{current_time}.png"
                        face_img = frame[y:y2, x:x2]
                        cv2.imwrite(filename, face_img)

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

camera.release()
cv2.destroyAllWindows()

plt.figure(figsize=(10, 5))
emotions, counts = zip(*emotion_counter.items())
plt.bar(emotions, counts, color='skyblue')
plt.xlabel('Emoções')
plt.ylabel('Ocorrências')
plt.title('Total de Ocorrências das Emoções')
plt.savefig('emocao_grafico_barras.png')
plt.show()

plt.figure(figsize=(10, 6))
for emotion in emotion_labels:
    emotion_times = [time_stamps[i] for i in range(len(emotion_over_time)) if emotion_over_time[i] == emotion]
    emotion_counts = list(range(1, len(emotion_times) + 1))
    plt.plot(emotion_times, emotion_counts, label=emotion)

plt.xlabel('Tempo (s)')
plt.ylabel('Ocorrências acumuladas')
plt.title('Emoções ao longo do tempo')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('emocao_grafico_temporal.png')
plt.show()

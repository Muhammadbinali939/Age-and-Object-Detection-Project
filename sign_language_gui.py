from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load sign language model
model = load_model('sign_language_model.h5')

# Detect sign language in real-time
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img_resized = cv2.resize(frame, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_input)
    cv2.putText(frame, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#Driver code
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit, send
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
socketio = SocketIO(app)

# Data preprocessing for the model
data_generator = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)

# Loading the ASL model
model = load_model('asl_alphabet_model.h5')

# Setting up image and frame sizes
IMAGE_SIZE = 200
CROP_SIZE = 400

# Loading classes for ASL alphabet
with open("classes.txt") as classes_file:
    classes_string = classes_file.readline()
classes = sorted(classes_string.split())

# Preparing cv2 for webcam feed
cap = cv2.VideoCapture(0)

# Variables for storing message and last prediction
message = ''
last_prediction = None

# Flask-SocketIO event for handling message emission
@socketio.on('speak')
def handle_speak():
    global message
    send(message)

# Flask route for the home page
@app.route('/')
def index():
    return render_template('home.html')

# Flask route for the demo page
@app.route('/demo')
def demo():
    return render_template('demo.html')

# Flask route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global message, last_prediction
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Target area for hand gestures
        cv2.rectangle(frame, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)

        # Preprocess the frame before input to the model
        cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
        resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
        reshaped_frame = np.array(resized_frame).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

        # Predict the frame
        prediction = np.array(model.predict(frame_for_model))
        predicted_class = classes[prediction.argmax()]
        frame = cv2.flip(frame, 1)

        # Prepare output based on model's confidence
        prediction_probability = prediction[0, prediction.argmax()]
        if prediction_probability > 0.6:
            # High confidence
            cv2.putText(frame, ' {} - {:.2f}%'.format(predicted_class, prediction_probability * 100), (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
            # Append predicted class to message if different from last prediction
            if predicted_class != last_prediction:
                if predicted_class == 'space':
                    message += ' '
                elif predicted_class == 'nothing':
                    message += ""
                else:
                    message += predicted_class
                last_prediction = predicted_class
        elif 0.2 < prediction_probability <= 0.6:
            # Low confidence
            cv2.putText(frame, ' Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability * 100), (10, 450), 1, 2, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            # No confidence
            cv2.putText(frame, classes[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame as a JPEG image
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Emit the message to the client
        socketio.emit('message', message)

if __name__ == "__main__":
    socketio.run(app, debug=True)

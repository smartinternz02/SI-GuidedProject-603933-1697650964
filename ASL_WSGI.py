from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit , send
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
socketio = SocketIO(app)

# Your existing code for setting up the model and the data generator goes here
# ...
# Prepare data generator for standardizing frames before sending them into the model.
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

# Loading the model.
model = load_model('asl_alphabet_model.h5')
# Setting up the input image size and frame crop size.
IMAGE_SIZE = 200
CROP_SIZE = 400

# Creating list of available classes stored in classes.txt.
classes_file = open("classes.txt")
classes_string = classes_file.readline()
classes = classes_string.split()
classes.sort()  # The predict function sends out output in sorted order.

# Preparing cv2 for webcam feed
cap = cv2.VideoCapture(0)

# Create a variable to store the message and the last prediction
message = ''
last_prediction = None
msgs='A B C'
from flask_socketio import SocketIO, send
@socketio.on('speak')
def handle_speak():
    # Get the message from the server-side variable
    global msgs

    # Emit the message to the client
    send(msgs)

#-----------------------------------------------------
@app.route('/')
def index():
  return render_template('home.html')

@app.route('/demo')
def demo():
  return render_template('demo.html')

@app.route('/video_feed')
def video_feed():
  return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global message
    global last_prediction
    while True:
      # Capture frame-by-frame
      ret, frame = cap.read()

      # Target area where the hand gestures should be.
      cv2.rectangle(frame, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)
      
      # Preprocessing the frame before input to the model.
      cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
      resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
      reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
      frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

      # Predicting the frame.
      prediction = np.array(model.predict(frame_for_model))
      predicted_class = classes[prediction.argmax()]      # Selecting the max confidence index.

      frame = cv2.flip(frame, 1)
      # Preparing output based on the model's confidence.
      prediction_probability = prediction[0, prediction.argmax()]
      if prediction_probability > 0.6:
          # High confidence.
          cv2.putText(frame, '          {} - {:.2f}%'.format(predicted_class, prediction_probability * 100), 
                                      (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
          # Append the predicted class to the message if it's different from the last prediction
          if predicted_class != last_prediction:
              if predicted_class == 'space':
                  message += ' '
              elif predicted_class=='nothing':
                  message +=""
              else:
                  message += predicted_class
              last_prediction = predicted_class
      elif prediction_probability > 0.2 and prediction_probability <= 0.6:
          # Low confidence.
          cv2.putText(frame, '          Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability * 100), 
                                      (10, 450), 1, 2, (0, 255, 255), 2, cv2.LINE_AA)
      else:
          # No confidence.
          cv2.putText(frame, classes[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)

      # Encode the frame as a JPEG image
      _, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()

      # Yield the frame as a Flask response
      yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

      # Emit the message to the client
      socketio.emit('message', message)

if __name__=="__main__":
  socketio.run(app, debug=True)
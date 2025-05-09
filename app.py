from flask import Flask, render_template, Response, request, jsonify
from detect_mask_video import detect_and_predict_mask
import cv2
from tensorflow.keras.models import load_model
import os
from flask_cors import CORS  # Add CORS support

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

face_proto = os.path.join("face_detector", "deploy.prototxt")
face_model = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(face_proto, face_model)
maskNet = load_model("mask_detector.keras")

camera = cv2.VideoCapture(0)
detection_on = False
latest_label = "Waiting..."

def generate_frames():
    global detection_on, latest_label
    while True:
        success, frame = camera.read()
        if not success:
            break

        if detection_on:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                latest_label = label
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle', methods=['POST'])
def toggle():
    global detection_on
    detection_on = not detection_on
    return jsonify({'status': 'on' if detection_on else 'off'})

@app.route('/latest_label')
def get_latest_label():
    return jsonify({'label': latest_label})

if __name__ == "__main__":
    app.run(debug=True)

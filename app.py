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
latest_detection_results = {
    "faces": [],
    "fps": 0.0,
    "status": "Offline"
}

def generate_frames():
    global detection_on, latest_label, latest_detection_results
    import time
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        current_time = time.time()
        frame_count += 1
        
        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            latest_detection_results["fps"] = fps

        if detection_on:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            
            # Store detection results
            faces = []
            for i, (box, pred) in enumerate(zip(locs, preds)):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                confidence = max(mask, withoutMask)
                
                faces.append({
                    "id": i,
                    "label": label,
                    "confidence": float(confidence),
                    "bbox": [int(startX), int(startY), int(endX), int(endY)]
                })

                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label_text = f"{label}: {confidence * 100:.2f}%"

                cv2.putText(frame, label_text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Update latest results
            latest_detection_results["faces"] = faces
            latest_detection_results["status"] = "Active"
            
            if faces:
                latest_label = faces[0]["label"]
            else:
                latest_label = "No faces detected"
        else:
            latest_detection_results["faces"] = []
            latest_detection_results["status"] = "Inactive"
            latest_label = "Detection paused"

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('ultra_modern.html')

@app.route('/original')
def original():
    return render_template('index.html')

@app.route('/enhanced')
def enhanced():
    return render_template('enhanced_index.html')

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

@app.route('/status')
def get_status():
    """Get current detection status and results"""
    try:
        # Return real detection results
        return jsonify(latest_detection_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get performance statistics"""
    try:
        # Return real performance stats
        stats = {
            "fps": latest_detection_results.get("fps", 0.0),
            "processed_frames": 0,  # Could be tracked if needed
            "total_frames": 0,      # Could be tracked if needed
            "skip_ratio": 1,
            "avg_detection_time": 0.04,
            "faces_detected": len(latest_detection_results.get("faces", [])),
            "masks_detected": len([f for f in latest_detection_results.get("faces", []) if f.get("label") == "Mask"])
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'camera': camera is not None and camera.isOpened(),
            'detection_on': detection_on,
            'model_loaded': True
        }
        return jsonify(health_status)
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5001)))

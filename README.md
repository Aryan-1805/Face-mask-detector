Face Mask Detection System
This project implements a face mask detection system using deep learning techniques, OpenCV, and Flask. It identifies whether a person is wearing a face mask in real-time through webcam feed or video files.

Project Structure

train_mask_detector.py: Script to train the face mask detector model using TensorFlow and Keras.
detect_mask_video.py: Script to perform face mask detection in real-time from a webcam feed or video.
app.py: Flask application to serve the face mask detection model via a web interface.
mask_detector.keras: Trained Keras model file.
mask_detector.model: Trained model file (alternative format).
requirements.txt: List of required libraries and dependencies.
plot.png: Graphical representation of training loss and accuracy.
Installation

Clone the repository:
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection
Install the required dependencies:
pip install -r requirements.txt
Training the Model

To train the face mask detector model, use the train_mask_detector.py script:

python train_mask_detector.py --dataset dataset
The script will save the trained model and a plot of the training loss and accuracy.

Running the Detection

Using the Flask App:
Start the Flask server:

python app.py
Visit http://localhost:5000/ in your browser to use the web interface for mask detection.

Using Webcam or Video Feed:
To perform real-time mask detection:

python detect_mask_video.py
This will activate the webcam and display the detection results in real-time.

Results and Accuracy

The plot (plot.png) shows the training loss and accuracy, indicating the model's performance over epochs. The model achieves high accuracy, with loss decreasing significantly after the initial epochs.

Technologies Used

Python: For building and training the model.
TensorFlow/Keras: Deep learning model training.
OpenCV: Real-time video processing.
Flask: Web server for model deployment.
Matplotlib: Plotting training metrics.
Future Improvements

Integrating mask detection on recorded video files.
Enhancing UI with a responsive design.
Implementing multi-person detection.
Extending the model to detect incorrect mask usage.
Acknowledgements

This project utilizes TensorFlow, OpenCV, and Flask for model training, real-time detection, and deployment.

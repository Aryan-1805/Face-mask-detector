
# 😷 Face Mask Detection System

A real-time face mask detection system built using **Deep Learning**, **OpenCV**, and **Flask**. This project identifies whether individuals are wearing face masks via webcam or video input, making it ideal for public safety and monitoring applications.

---

## 📁 Project Structure

| File / Folder            | Description                                                           |
| ------------------------ | --------------------------------------------------------------------- |
| `train_mask_detector.py` | Script to train the mask detection model using TensorFlow and Keras.  |
| `detect_mask_video.py`   | Real-time face mask detection using webcam or video feed.             |
| `app.py`                 | Flask web app that runs the detection system via a browser interface. |
| `mask_detector.keras`    | Trained Keras model file.                                             |
| `mask_detector.model`    | Trained model in an alternate format.                                 |
| `requirements.txt`       | List of dependencies and required Python packages.                    |
| `plot.png`               | Visual representation of model loss and accuracy during training.     |

---

## 🛠 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Training the Model

To train the model from scratch using your own dataset:

```bash
python train_mask_detector.py --dataset dataset
```

* The script will output a trained model and save a plot of the training performance (`plot.png`).

---

## 🧪 Running the Detection

### 🔹 Using the Flask Web Interface

```bash
python app.py
```

* Navigate to `http://localhost:5000/` in your browser.
* Upload images and get real-time detection results in the UI.

### 🔹 Using Webcam or Video Stream

```bash
python detect_mask_video.py
```

* This will launch your webcam and start real-time mask detection on the video feed.

---

## 📊 Results & Accuracy

The model demonstrates high accuracy, as visualized in `plot.png`.

* **Training Loss** drops consistently.
* **Accuracy** increases significantly after the initial few epochs.

---

## 🧰 Technologies Used

* **Python** – Core programming language
* **TensorFlow / Keras** – Deep learning model training
* **OpenCV** – Image and video processing
* **Flask** – Lightweight web framework for deployment
* **Matplotlib** – Visualization of training performance

---

## 🚀 Future Enhancements

* 🎥 Extend detection to uploaded or recorded video files
* 🖥 Improve the frontend with a more dynamic and responsive UI
* 👥 Add support for detecting **multiple faces** at once
* 🩺 Classify **improper mask usage** (e.g., mask below nose)

---

## 🙏 Acknowledgements

Special thanks to the open-source libraries and contributors behind:

* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [Flask](https://flask.palletsprojects.com/)


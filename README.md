# 😷 AI Face Mask Detection System

A **real-time face mask detection system** built using **Deep Learning**, **OpenCV**, and **Flask**. This project identifies whether individuals are wearing face masks via webcam or video input, featuring a **modern web interface** with real-time statistics, dark/light mode, and comprehensive monitoring capabilities.

![Face Mask Detection](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green)
![Flask](https://img.shields.io/badge/Flask-3.1.0-red)

---

## ✨ Features

### 🎯 **Core Detection**
- **Real-time face detection** using OpenCV's DNN module
- **Mask classification** with MobileNetV2 transfer learning
- **Multi-face support** - detect multiple people simultaneously
- **High accuracy** detection with confidence scores

### 🖥️ **Modern Web Interface**
- **Ultra-modern UI** with glassmorphism design
- **Dark/Light mode** toggle with smooth transitions
- **Real-time statistics** dashboard (FPS, face count, accuracy)
- **Live video stream** with detection overlays
- **Responsive design** for all screen sizes
- **Settings modal** with customizable parameters

### 🚀 **Performance & Monitoring**
- **Real-time FPS monitoring** and performance tracking
- **API endpoints** for status, statistics, and health checks
- **Error handling** and graceful degradation
- **Optimized detection** with configurable parameters

---

## 📁 Project Structure

| File / Folder                    | Description                                                           |
| -------------------------------- | --------------------------------------------------------------------- |
| `app.py`                         | **Main Flask application** with real-time detection and API endpoints |
| `detect_mask_video.py`           | Core detection logic with face detection and mask classification      |
| `train_mask_detector.py`         | Model training script using MobileNetV2 transfer learning            |
| `templates/ultra_modern.html`    | **Modern web interface** with dark/light mode and real-time stats    |
| `config.py`                      | Centralized configuration settings                                    |
| `mask_detector.keras`            | **Trained Keras model** (MobileNetV2 + custom head)                  |
| `face_detector/`                 | OpenCV DNN face detection models (Caffe)                             |
| `dataset/`                       | Training data (with_mask/ and without_mask/ folders)                 |
| `requirements.txt`               | Python dependencies and versions                                      |
| `Procfile`                       | Heroku deployment configuration                                       |
| `setup.sh`                       | Automated setup script for virtual environment                       |

---

## 🛠 Quick Start

### 1. **Clone & Setup**
```bash
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection

# Create virtual environment
python -m venv face_mask_env
source face_mask_env/bin/activate  # On Windows: face_mask_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Run the Application**
```bash
python app.py
```

### 3. **Access the Interface**
- Open your browser and go to `http://localhost:5001`
- Click **"Start Detection"** to begin real-time face mask detection
- Toggle between **Dark/Light mode** using the theme switcher
- View **real-time statistics** in the dashboard

---

## 🧠 Model Training

### **Train from Scratch**
```bash
python train_mask_detector.py --dataset dataset
```

### **Training Features**
- **MobileNetV2** base model with transfer learning
- **Data augmentation** (rotation, zoom, shift, shear, flip)
- **Early stopping** and learning rate scheduling
- **Model checkpointing** and performance visualization
- **Automatic plot generation** (`plot.png`)

---

## 🎮 Usage Guide

### **Web Interface Controls**
- **Start/Stop Detection** - Toggle real-time detection on/off
- **Settings** - Configure detection parameters (confidence, max faces, etc.)
- **Theme Toggle** - Switch between dark and light modes
- **Real-time Stats** - Monitor FPS, face count, mask count, and accuracy

### **API Endpoints**
- `GET /` - Main web interface
- `GET /video` - Live video stream (MJPEG)
- `POST /toggle` - Start/stop detection
- `GET /status` - Current detection status and results
- `GET /stats` - Performance statistics
- `GET /health` - System health check

---

## 📊 Performance & Accuracy

### **Model Performance**
- **Training Accuracy**: ~95%+ on validation set
- **Real-time FPS**: 25-30 FPS on modern hardware
- **Detection Speed**: ~40ms per frame
- **Memory Usage**: Optimized for production deployment

### **Supported Formats**
- **Input**: Webcam, video files, image files
- **Output**: Real-time video stream with detection overlays
- **Models**: Keras (.keras), Caffe (.caffemodel, .prototxt)

---

## 🧰 Technology Stack

### **Backend**
- **Python 3.11+** - Core programming language
- **TensorFlow 2.16.2** - Deep learning framework
- **Keras 3.9.2** - High-level neural network API
- **OpenCV 4.11.0** - Computer vision and image processing
- **Flask 3.1.0** - Web framework and API server

### **Frontend**
- **HTML5/CSS3** - Modern web interface
- **JavaScript (ES6+)** - Real-time updates and interactions
- **Font Awesome** - Icons and UI elements
- **Animate.css** - Smooth animations and transitions

### **AI/ML**
- **MobileNetV2** - Pre-trained base model for transfer learning
- **OpenCV DNN** - Face detection using Caffe models
- **Data Augmentation** - Enhanced training with image transformations
- **Transfer Learning** - Efficient training with pre-trained weights

---

## 🚀 Deployment

### **Local Development**
```bash
python app.py
# Runs on http://localhost:5001
```

### **Heroku Deployment**
```bash
# Add Heroku remote
heroku create your-app-name

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1
```

### **Production Considerations**
- **Environment Variables**: Set `PORT` for production
- **Static Files**: Optimize images and assets
- **Monitoring**: Use `/health` endpoint for uptime monitoring
- **Scaling**: Configure worker processes as needed

---

## 🔧 Configuration

### **Detection Parameters** (via Settings)
- **Confidence Threshold**: 0.5 (adjustable)
- **Max Faces**: 10 (configurable)
- **Detection Interval**: 1000ms (customizable)
- **Frame Skip**: 1 (performance optimization)

### **Display Options**
- **Show Confidence Scores**: Toggle on/off
- **Show FPS Counter**: Real-time performance monitoring
- **Show Bounding Boxes**: Visual detection indicators
- **Video Quality**: Medium (720p) default

---

## 🎯 Future Enhancements

### **Planned Features**
- 🎥 **Video File Upload** - Process recorded videos
- 👥 **Multi-person Tracking** - Track individuals across frames
- 🩺 **Mask Quality Assessment** - Detect improper mask usage
- 📱 **Mobile App** - Native iOS/Android applications
- ☁️ **Cloud Integration** - AWS/Azure deployment options
- 🔔 **Notifications** - Real-time alerts and logging
- 📊 **Analytics Dashboard** - Historical data and insights

### **Technical Improvements**
- **GPU Acceleration** - CUDA support for faster processing
- **Model Optimization** - Quantization and pruning
- **Edge Deployment** - Raspberry Pi and IoT devices
- **API Rate Limiting** - Production-ready API management

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Fork the repository
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
python app.py

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

Special thanks to the open-source community and contributors:

- **[TensorFlow](https://www.tensorflow.org/)** - Deep learning framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[Flask](https://flask.palletsprojects.com/)** - Web framework
- **[MobileNetV2](https://arxiv.org/abs/1801.04381)** - Efficient CNN architecture
- **[Font Awesome](https://fontawesome.com/)** - Icons and UI elements

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/face-mask-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/face-mask-detection/discussions)
- **Email**: support@facemaskdetection.com

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by the Face Mask Detection Team

</div>
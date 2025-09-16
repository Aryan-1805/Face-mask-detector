#!/bin/bash

# Face Mask Detection Project Setup Script
echo "🚀 Setting up Face Mask Detection Project..."

# Check if virtual environment exists
if [ ! -d "face_mask_env" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv face_mask_env
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source face_mask_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install opencv-python==4.11.0.86
pip install tensorflow==2.16.2
pip install flask==3.1.0
pip install flask-cors==3.1.0
pip install numpy==1.26.4
pip install pillow==11.2.1
pip install imutils==0.5.4
pip install scikit-learn==1.6.1
pip install matplotlib==3.10.1

# Verify installation
echo "🔍 Verifying installation..."
python -c "import cv2, tensorflow, flask; print('✅ All dependencies installed successfully!')"

# Check model files
echo "🤖 Checking model files..."
if [ -f "mask_detector.keras" ]; then
    echo "✅ Mask detector model found"
else
    echo "⚠️ Mask detector model not found - you may need to train it"
fi

if [ -f "face_detector/deploy.prototxt" ]; then
    echo "✅ Face detector model found"
else
    echo "❌ Face detector model not found"
fi

echo ""
echo "🎉 Setup complete! To start the project:"
echo "1. Activate virtual environment: source face_mask_env/bin/activate"
echo "2. Run the app: python app.py"
echo "3. Open browser: http://localhost:5000"
echo ""
echo "For enhanced features: python enhanced_app.py"

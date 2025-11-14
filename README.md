# 🚁 Object Detection with Distance Estimation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.35+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/badge/uv-package_manager-blue.svg)](https://github.com/astral-sh/uv)

Real-time object detection with monocular depth estimation for drone applications using state-of-the-art AI models from Hugging Face.

![Demo](https://img.shields.io/badge/Status-Active-success)

## ✨ Features

- 🎯 **Real-time Object Detection** using RT-DETR (108 FPS on T4 GPU)
- 📏 **Distance Estimation** from single camera using Depth Anything V2
- 🎨 **Color-coded Bounding Boxes** (Red=Close, Yellow=Medium, Blue=Far)
- 🎥 **Live Webcam Support** with visual depth map overlay
- ⚡ **Fast & Accurate** - Optimized for drone and robotics applications
- 🔧 **Easy Setup** using uv package manager

## 🎬 Demo

The system detects objects in real-time and categorizes their distance using only your laptop camera:
- **RED boxes** = Objects are CLOSE
- **YELLOW boxes** = Objects are MEDIUM distance  
- **BLUE boxes** = Objects are FAR

A live depth map visualization appears in the corner showing the scene's depth structure.

## 📋 Prerequisites

- Python 3.10 or higher
- Webcam/Camera device
- Linux (Fedora/Ubuntu) / macOS / Windows
- [uv package manager](https://github.com/astral-sh/uv)

## 🚀 Quick Start

### 1. Install uv Package Manager

```python
#Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

#Or using pipx
pipx install uv
```


### 2. Clone the Repository

```
git clone https://github.com/HashimCodeDev/Object-Detection.git
cd Object-Detection
```


### 3. Setup Project

```python
#Create directory structure
mkdir -p src/drone_object_detection examples data/output

#Create required files
touch src/drone_object_detection/init.py
touch src/drone_object_detection/distance_detector.py
touch examples/detect_webcam_distance.py

#Make the example script executable
chmod +x examples/detect_webcam_distance.py
```


### 4. Install Dependencies

```python
Add main dependencies
uv add torch torchvision transformers pillow opencv-python numpy timm

Add development dependencies (optional)
uv add --dev pytest black ruff

Verify installation
uv sync
```


### 5. Run the Detection

```python
uv run examples/detect_webcam_distance.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save current frame

## 📁 Project Structure

Object-Detection/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── src/
│ └── drone_object_detection/
│ ├── init.py
│ └── distance_detector.py # Main detection module
├── examples/
│ └── detect_webcam_distance.py # Webcam demo script
├── tests/
│ └── test_detector.py
└── data/
├── input/ # Input images/videos
└── output/ # Saved captures


## 🔧 Configuration

### Camera Permissions (Linux)

If you encounter camera access issues on Linux:

```python
#Add yourself to video group
sudo usermod -aG video $USER

#Fix device permissions
sudo chmod 666 /dev/video*

Log out and log back in for changes to take effect
```

### Model Selection

You can customize the models used:

```python
detector = DroneObjectDistanceDetector(
detection_model="PekingU/rtdetr_r50vd", # Options: r18vd (fastest), r50vd (balanced), r101vd (accurate)
depth_model="depth-anything/Depth-Anything-V2-Small-hf", # Options: Small, Base, Large
confidence_threshold=0.5 # Detection confidence threshold
)
```


## 🤖 Models Used

### Object Detection: RT-DETR
- **RT-DETR (Real-Time Detection Transformer)** from Baidu
- Achieves 53.1% AP with 108 FPS on T4 GPU
- Trained on COCO dataset (80 object classes)
- [Model Card](https://huggingface.co/PekingU/rtdetr_r50vd)

### Depth Estimation: Depth Anything V2
- **Depth Anything V2** - State-of-the-art monocular depth estimation
- Provides relative depth from single RGB image
- Optimized for real-time performance
- [Model Card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)

## 📊 Performance

| Component | Model | FPS (CPU) | FPS (GPU) |
|-----------|-------|-----------|-----------|
| Object Detection | RT-DETR R50 | ~5-10 | ~108 |
| Depth Estimation | Depth Anything V2 Small | ~8-12 | ~60 |
| **Combined** | Both | ~3-5 | ~40-50 |

*Performance varies based on hardware and resolution*

## 🎯 Use Cases

- **Autonomous Drones** - Real-time obstacle detection and distance estimation
- **Robotics** - Navigation and object avoidance
- **Surveillance** - Security monitoring with distance tracking
- **Accessibility** - Visual assistance for navigation
- **Industrial Automation** - Quality control and safety monitoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) by Baidu
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [uv Package Manager](https://github.com/astral-sh/uv) by Astral

## 📧 Contact

**Hashim Mohamed T A**
- GitHub: [@HashimCodeDev](https://github.com/HashimCodeDev)
- Repository: [Object-Detection](https://github.com/HashimCodeDev/Object-Detection)

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

Made with ❤️ for drone and robotics applications

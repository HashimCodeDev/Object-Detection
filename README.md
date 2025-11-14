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


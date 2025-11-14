"""
Available Models Configuration for Object Detection and Depth Estimation

This file contains all supported models organized by performance tier.
Simply import and use the model names in your code.
"""

# ============================================================================
# OBJECT DETECTION MODELS
# ============================================================================

DETECTION_MODELS = {
    # RT-DETR Family (Transformer-based) - RECOMMENDED for Hugging Face
    "rt_detr": {
        "fastest": {
            "name": "PekingU/rtdetr_r18vd",
            "fps_cpu": "8-12",
            "fps_gpu": "92",
            "map": "46.5%",
            "description": "Fastest RT-DETR, best for CPU/laptop",
            "recommended_for": "CPU, Real-time on laptop"
        },
        "balanced": {
            "name": "PekingU/rtdetr_r50vd",
            "fps_cpu": "4-8",
            "fps_gpu": "108",
            "map": "53.1%",
            "description": "Good balance of speed and accuracy",
            "recommended_for": "GPU, Balanced performance"
        },
        "accurate": {
            "name": "PekingU/rtdetr_r101vd",
            "fps_cpu": "2-4",
            "fps_gpu": "74",
            "map": "54.3%",
            "description": "Higher accuracy, slower",
            "recommended_for": "GPU, Better accuracy"
        },
        "best": {
            "name": "PekingU/rtdetr_x",
            "fps_cpu": "1-3",
            "fps_gpu": "74",
            "map": "54.8%",
            "description": "Best RT-DETR accuracy",
            "recommended_for": "GPU, Maximum accuracy"
        }
    },
    
    # RT-DETR V2 (2025 - Latest)
    "rt_detr_v2": {
        "default": {
            "name": "jadechoghari/RT-DETRv2",
            "fps_cpu": "4-8",
            "fps_gpu": "100+",
            "map": "55%+",
            "description": "Latest RT-DETR version with improvements",
            "recommended_for": "GPU, Latest technology"
        }
    },
    
    # YOLOS (YOLO + Transformers)
    "yolos": {
        "small": {
            "name": "hustvl/yolos-small",
            "fps_cpu": "6-10",
            "fps_gpu": "80+",
            "map": "36.1%",
            "description": "Lightweight YOLO transformer",
            "recommended_for": "CPU, Lightweight detection"
        },
        "base": {
            "name": "hustvl/yolos-base",
            "fps_cpu": "3-6",
            "fps_gpu": "50+",
            "map": "42.0%",
            "description": "Standard YOLO transformer",
            "recommended_for": "GPU, Standard detection"
        }
    },
    
    # Original DETR
    "detr": {
        "resnet50": {
            "name": "facebook/detr-resnet-50",
            "fps_cpu": "2-4",
            "fps_gpu": "40+",
            "map": "42.0%",
            "description": "Original DETR model",
            "recommended_for": "Research, Experimentation"
        },
        "resnet101": {
            "name": "facebook/detr-resnet-101",
            "fps_cpu": "1-3",
            "fps_gpu": "30+",
            "map": "43.5%",
            "description": "More accurate original DETR",
            "recommended_for": "Research, Higher accuracy"
        }
    },
    
    # Conditional DETR (Faster convergence)
    "conditional_detr": {
        "resnet50": {
            "name": "microsoft/conditional-detr-resnet-50",
            "fps_cpu": "2-4",
            "fps_gpu": "45+",
            "map": "43.8%",
            "description": "Improved DETR with faster training",
            "recommended_for": "GPU, Better than original DETR"
        }
    }
}

# ============================================================================
# DEPTH ESTIMATION MODELS
# ============================================================================

DEPTH_MODELS = {
    # Depth Anything V2 (2024-2025) - RECOMMENDED
    "depth_anything_v2": {
        "small": {
            "name": "depth-anything/Depth-Anything-V2-Small-hf",
            "fps_cpu": "8-12",
            "fps_gpu": "60+",
            "description": "Fastest depth model, best for CPU",
            "recommended_for": "CPU, Real-time performance",
            "output_type": "relative"
        },
        "base": {
            "name": "depth-anything/Depth-Anything-V2-Base-hf",
            "fps_cpu": "4-8",
            "fps_gpu": "40+",
            "description": "Balanced depth estimation",
            "recommended_for": "GPU, Balanced quality",
            "output_type": "relative"
        },
        "large": {
            "name": "depth-anything/Depth-Anything-V2-Large-hf",
            "fps_cpu": "2-4",
            "fps_gpu": "25+",
            "description": "Best depth quality",
            "recommended_for": "GPU, Maximum quality",
            "output_type": "relative"
        }
    },
    
    # Depth Anything V2 - Metric (Real distance in meters)
    "depth_anything_v2_metric": {
        "indoor_small": {
            "name": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            "fps_cpu": "8-12",
            "fps_gpu": "60+",
            "description": "Metric depth for indoor scenes",
            "recommended_for": "Indoor drone navigation",
            "output_type": "metric (meters)"
        },
        "outdoor_small": {
            "name": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
            "fps_cpu": "8-12",
            "fps_gpu": "60+",
            "description": "Metric depth for outdoor scenes",
            "recommended_for": "Outdoor drone navigation",
            "output_type": "metric (meters)"
        },
        "indoor_base": {
            "name": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
            "fps_cpu": "4-8",
            "fps_gpu": "40+",
            "description": "Better indoor metric depth",
            "recommended_for": "Indoor, better quality",
            "output_type": "metric (meters)"
        },
        "outdoor_base": {
            "name": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
            "fps_cpu": "4-8",
            "fps_gpu": "40+",
            "description": "Better outdoor metric depth",
            "recommended_for": "Outdoor, better quality",
            "output_type": "metric (meters)"
        }
    },
    
    # Depth Anything V1
    "depth_anything_v1": {
        "small": {
            "name": "LiheYoung/depth-anything-small-hf",
            "fps_cpu": "6-10",
            "fps_gpu": "50+",
            "description": "Original Depth Anything small",
            "recommended_for": "Legacy, still good",
            "output_type": "relative"
        },
        "base": {
            "name": "LiheYoung/depth-anything-base-hf",
            "fps_cpu": "3-6",
            "fps_gpu": "35+",
            "description": "Original Depth Anything base",
            "recommended_for": "Legacy, balanced",
            "output_type": "relative"
        },
        "large": {
            "name": "LiheYoung/depth-anything-large-hf",
            "fps_cpu": "1-3",
            "fps_gpu": "20+",
            "description": "Original Depth Anything large",
            "recommended_for": "Legacy, quality",
            "output_type": "relative"
        }
    },
    
    # MiDaS (Intel - Classic)
    "midas": {
        "dpt_large": {
            "name": "Intel/dpt-large",
            "fps_cpu": "3-6",
            "fps_gpu": "30+",
            "description": "High quality MiDaS model",
            "recommended_for": "Classic reliable choice",
            "output_type": "relative"
        },
        "dpt_hybrid": {
            "name": "Intel/dpt-hybrid-midas",
            "fps_cpu": "4-8",
            "fps_gpu": "40+",
            "description": "Balanced MiDaS model",
            "recommended_for": "Faster MiDaS option",
            "output_type": "relative"
        },
        "dpt_swin_tiny": {
            "name": "Intel/dpt-swinv2-tiny-256",
            "fps_cpu": "8-12",
            "fps_gpu": "60+",
            "description": "Tiny fast MiDaS",
            "recommended_for": "CPU, lightweight",
            "output_type": "relative"
        }
    },
    
    # ZoeDepth (Metric depth)
    "zoedepth": {
        "nyu_kitti": {
            "name": "Intel/zoedepth-nyu-kitti",
            "fps_cpu": "3-6",
            "fps_gpu": "35+",
            "description": "Metric depth estimation",
            "recommended_for": "Real distance measurements",
            "output_type": "metric (meters)"
        }
    },
    
    # Marigold (Stable Diffusion-based)
    "marigold": {
        "v1": {
            "name": "prs-eth/marigold-depth-v1-0",
            "fps_cpu": "0.5-1",
            "fps_gpu": "5-10",
            "description": "Very high quality, very slow",
            "recommended_for": "Offline processing, best quality",
            "output_type": "relative"
        }
    }
}

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    "laptop_cpu_fastest": {
        "detection": DETECTION_MODELS["rt_detr"]["fastest"]["name"],
        "depth": DEPTH_MODELS["depth_anything_v2"]["small"]["name"],
        "description": "Optimized for laptop CPU - fastest performance"
    },
    
    "laptop_cpu_balanced": {
        "detection": DETECTION_MODELS["rt_detr"]["fastest"]["name"],
        "depth": DEPTH_MODELS["depth_anything_v2"]["base"]["name"],
        "description": "Laptop CPU with better depth quality"
    },
    
    "gpu_balanced": {
        "detection": DETECTION_MODELS["rt_detr"]["balanced"]["name"],
        "depth": DEPTH_MODELS["depth_anything_v2"]["base"]["name"],
        "description": "Balanced performance on GPU"
    },
    
    "gpu_best_quality": {
        "detection": DETECTION_MODELS["rt_detr"]["accurate"]["name"],
        "depth": DEPTH_MODELS["depth_anything_v2"]["large"]["name"],
        "description": "Best quality on GPU"
    },
    
    "indoor_drone": {
        "detection": DETECTION_MODELS["rt_detr"]["fastest"]["name"],
        "depth": DEPTH_MODELS["depth_anything_v2_metric"]["indoor_small"]["name"],
        "description": "Indoor drone with metric depth"
    },
    
    "outdoor_drone": {
        "detection": DETECTION_MODELS["rt_detr"]["fastest"]["name"],
        "depth": DEPTH_MODELS["depth_anything_v2_metric"]["outdoor_small"]["name"],
        "description": "Outdoor drone with metric depth"
    },
    
    "latest_technology": {
        "detection": DETECTION_MODELS["rt_detr_v2"]["default"]["name"],
        "depth": DEPTH_MODELS["depth_anything_v2"]["large"]["name"],
        "description": "Latest models, requires good GPU"
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def list_all_detection_models():
    """Print all available detection models"""
    print("\n" + "="*80)
    print("OBJECT DETECTION MODELS")
    print("="*80)
    
    for family, variants in DETECTION_MODELS.items():
        print(f"\n📦 {family.upper()}")
        for variant, info in variants.items():
            print(f"   {variant}:")
            print(f"      Model: {info['name']}")
            print(f"      FPS (CPU/GPU): {info['fps_cpu']} / {info['fps_gpu']}")
            print(f"      mAP: {info['map']}")
            print(f"      {info['description']}")

def list_all_depth_models():
    """Print all available depth models"""
    print("\n" + "="*80)
    print("DEPTH ESTIMATION MODELS")
    print("="*80)
    
    for family, variants in DEPTH_MODELS.items():
        print(f"\n📏 {family.upper()}")
        for variant, info in variants.items():
            print(f"   {variant}:")
            print(f"      Model: {info['name']}")
            print(f"      FPS (CPU/GPU): {info['fps_cpu']} / {info['fps_gpu']}")
            print(f"      Output: {info['output_type']}")
            print(f"      {info['description']}")

def list_presets():
    """Print all preset configurations"""
    print("\n" + "="*80)
    print("PRESET CONFIGURATIONS")
    print("="*80)
    
    for name, config in PRESETS.items():
        print(f"\n🎯 {name}")
        print(f"   Detection: {config['detection']}")
        print(f"   Depth: {config['depth']}")
        print(f"   {config['description']}")

def get_preset(preset_name):
    """Get a preset configuration by name"""
    return PRESETS.get(preset_name)

# Quick access shortcuts
FASTEST_CPU = PRESETS["laptop_cpu_fastest"]
BALANCED_GPU = PRESETS["gpu_balanced"]
BEST_QUALITY = PRESETS["gpu_best_quality"]

if __name__ == "__main__":
    # Display all available options
    list_all_detection_models()
    list_all_depth_models()
    list_presets()

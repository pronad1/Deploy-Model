# Spinal Injury Detection System

![Python](https://img.shields.io/badge/python-v3.9-blue)
![Flask](https://img.shields.io/badge/flask-3.0.0-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-red)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-active-success)

AI-powered web application for detecting spinal lesions from DICOM X-ray images using deep learning models.

> ⚠️ **To deploy this app online**, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)  
> 📖 **Documentation**: [DEPLOYMENT.md](DEPLOYMENT.md) | 🚀 **Setup**: [GITHUB_SETUP.md](GITHUB_SETUP.md)

## 📋 Overview

This system uses an ensemble of classification models (DenseNet121, ResNet50, EfficientNetV2-S) and YOLO11 object detection to:
- **Classify** spine X-rays as Normal or Abnormal
- **Detect and localize** specific spinal lesions with bounding boxes
- Provide confidence scores and detailed analysis

### Dataset & Research

Based on the **VinDr-SpineXR** dataset - a large annotated medical image dataset for spinal lesion detection and classification from radiographs.

**Model Performance:**
- **Classification Ensemble**: 91.03% AUROC, 83.09% F1-score
- **Detection (YOLO11)**: 35 epochs, mAP50-95: 18.99%
- Beats baseline paper metrics on all classification measures

## 🚀 Quick Start

### Running Locally

1. **Clone the repository**
```bash
git clone https://github.com/pronad1/Deploy-Model.git
cd Deploy-Model
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open in browser**
```
http://localhost:5000
```

### Deploying Online

**GitHub only stores your code - it doesn't run the app!**

To make your app accessible on the internet, deploy to a hosting platform:

✅ **Render.com** (Free, Recommended) - [See Guide](DEPLOYMENT_GUIDE.md#option-1-rendercom-recommended---free--easy)  
✅ **Railway.app** (Free $5/month credit) - [See Guide](DEPLOYMENT_GUIDE.md#option-2-railwayapp-easy-with-better-free-tier)  
✅ **Hugging Face Spaces** (Free for ML) - [See Guide](DEPLOYMENT_GUIDE.md#option-3-hugging-face-spaces-best-for-ml-apps)

📖 **Full deployment guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

### Prerequisites (for local development)
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Model Files
```
ensemble output/
├── densenet121_balanced/model_best.pth
├── resnet50_optimized/model_best.pth
└── tf_efficientnetv2_s_optimized/model_best.pth

detection output/
└── yolo11/weights/best.pt
```

### Running Locally

```bash
python app.py
```

Visit: `http://localhost:5000`

## 📦 Deployment

### Option 1: Docker (Recommended)

```bash
docker build -t spine-detection .
docker run -p 5000:5000 spine-detection
```

### Option 2: Heroku

```bash
heroku create your-app-name
git push heroku main
```

### Option 3: AWS/Azure/GCP

Use the provided configuration files for cloud deployment.

## 🎯 Usage

1. **Upload DICOM File**
   - Drag & drop or click to browse
   - Only `.dcm` or `.dicom` files accepted
   - Max file size: 16MB

2. **AI Analysis**
   - Automatic validation of DICOM format
   - Classification: Normal vs Abnormal
   - Detection: Lesion localization (if abnormal)

3. **Results**
   - Overall diagnosis with confidence
   - Individual model predictions
   - Annotated image with bounding boxes
   - DICOM metadata display

## 🏗️ Architecture

```
User Upload → DICOM Validation → Preprocessing
                                      ↓
                            ┌─────────┴─────────┐
                            ↓                   ↓
                    Classification        Detection
                    (Ensemble 3 CNNs)     (YOLO11)
                            ↓                   ↓
                    Normal/Abnormal      Lesion Boxes
                            └─────────┬─────────┘
                                      ↓
                               Results Display
```

### Models
- **DenseNet121** (42% weight): Dense connections for feature reuse
- **ResNet50** (26% weight): Residual learning for deep networks
- **EfficientNetV2-S** (32% weight): Efficient scaling
- **YOLO11**: Real-time object detection for lesion localization

## 📊 Dataset Analysis

**VinDr-SpineXR Dataset:**
- Multiple lesion types (fractures, deformities, etc.)
- Bounding box annotations for abnormal cases
- "No finding" label for normal cases
- DICOM format with full medical metadata

**Preprocessing:**
- MONOCHROME1 inversion handling
- Normalization to 0-255 range
- RGB conversion for model compatibility
- Resize to 224×224 for classification

## 🔒 Security & Validation

- **File Type Validation**: Only DICOM files accepted
- **Size Limits**: Max 16MB per upload
- **Format Verification**: PyDICOM validation
- **Automatic Cleanup**: Temporary files removed after processing

## ⚕️ Medical Disclaimer

**This system is for research and educational purposes only.**

- Not FDA approved or clinically validated
- Should NOT replace professional medical diagnosis
- Always consult qualified radiologists/physicians
- Results are probabilistic and may contain errors

## 🛠️ Development

### Project Structure
```
Deploy-Model/
├── 📄 README.md                      # Main documentation
├── 📄 DEPLOYMENT.md                  # Deployment instructions
├── 📄 GITHUB_SETUP.md               # GitHub setup guide
├── 🐍 app.py                        # Flask application
├── 📄 requirements.txt              # Python dependencies
├── 🐳 Dockerfile                    # Docker configuration
├── 📄 Procfile                      # Heroku configuration
├── 📄 runtime.txt                   # Python version
├── 📄 .gitignore                    # Git ignore rules
├── 📁 templates/
│   └── index.html                   # Web interface
├── 📁 static/                       # CSS/JS assets
├── 📁 uploads/                      # Temporary upload folder
│   └── .gitkeep                     # Keep directory in git
├── 📁 ensemble output/
│   ├── densenet121_balanced/
│   │   └── model_best.pth           # 80 MB
│   ├── resnet50_optimized/
│   │   └── model_best.pth           # 26 MB
│   └── tf_efficientnetv2_s_optimized/
│       └── model_best.pth           # 23 MB
├── 📁 detection output/
│   └── yolo11/
│       └── weights/
│           └── best.pt              # 48 MB
└── 📓 vindr-spinexr-dataset-analysis.ipynb
```

### API Endpoints

**POST /upload**
- Upload and analyze DICOM file
- Returns classification + detection results

**GET /health**
- Check system status and model loading

## 📈 Performance Metrics

### Classification (Ensemble)
- **AUROC**: 91.03%
- **F1-Score**: 83.09%
- **Sensitivity**: 84.91%
- **Specificity**: 81.68%
- **Threshold**: 0.449

### Detection (YOLO11)
- **Epochs**: 35
- **mAP50-95**: 18.99%
- **Batch Size**: 12
- **Image Size**: 640×640

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Medical Disclaimer:** This software is for research and educational purposes only. Not for clinical use.

## 🙏 Acknowledgments

- VinDr-SpineXR dataset creators
- PyTorch and Ultralytics teams
- Medical imaging community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for advancing medical AI research**

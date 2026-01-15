from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import pydicom
from pydicom.pixel_data_handlers import pylibjpeg_handler
import numpy as np
from PIL import Image
import io
import os
import sys
import traceback
import warnings
from werkzeug.utils import secure_filename
import timm
from ultralytics import YOLO
import cv2
import base64
from torchvision import transforms

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables
classification_models = {}
detection_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensemble weights from final_results.json
ENSEMBLE_WEIGHTS = {
    'densenet121': 0.42,
    'efficientnet': 0.32,
    'resnet50': 0.26
}
CLASSIFICATION_THRESHOLD = 0.449


def load_models():
    """Load all trained models"""
    global classification_models, detection_model
    
    print("Loading models...")
    print(f"Device: {device}")
    print(f"Current directory: {os.getcwd()}")
    
    # Load Classification Models
    try:
        # DenseNet121
        densenet = timm.create_model('densenet121', pretrained=False, num_classes=1)
        densenet_path = 'ensemble output/densenet121_balanced/model_best.pth'
        print(f"Looking for DenseNet at: {densenet_path}")
        if os.path.exists(densenet_path):
            checkpoint = torch.load(densenet_path, map_location=device, weights_only=False)
            # Handle checkpoint format (with 'model_state_dict' key or direct state dict)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            densenet.load_state_dict(state_dict)
            densenet.eval()
            classification_models['densenet121'] = densenet.to(device)
            print("✓ DenseNet121 loaded")
        else:
            print(f"✗ DenseNet121 not found at {densenet_path}")
        
        # ResNet50
        resnet = timm.create_model('resnet50', pretrained=False, num_classes=1)
        resnet_path = 'ensemble output/resnet50_optimized/model_best.pth'
        print(f"Looking for ResNet50 at: {resnet_path}")
        if os.path.exists(resnet_path):
            checkpoint = torch.load(resnet_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            resnet.load_state_dict(state_dict)
            resnet.eval()
            classification_models['resnet50'] = resnet.to(device)
            print("✓ ResNet50 loaded")
        else:
            print(f"✗ ResNet50 not found at {resnet_path}")
        
        # EfficientNetV2-S
        efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=1)
        efficientnet_path = 'ensemble output/tf_efficientnetv2_s_optimized/model_best.pth'
        print(f"Looking for EfficientNet at: {efficientnet_path}")
        if os.path.exists(efficientnet_path):
            checkpoint = torch.load(efficientnet_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            efficientnet.load_state_dict(state_dict)
            efficientnet.eval()
            classification_models['efficientnet'] = efficientnet.to(device)
            print("✓ EfficientNetV2-S loaded")
        else:
            print(f"✗ EfficientNet not found at {efficientnet_path}")
            
    except Exception as e:
        print(f"Warning: Error loading classification models: {e}")
        traceback.print_exc()
    
    # Load YOLO Detection Model
    try:
        yolo_path = 'detection output/yolo11/weights/best.pt'
        print(f"Looking for YOLO at: {yolo_path}")
        if os.path.exists(yolo_path):
            detection_model = YOLO(yolo_path)
            print("✓ YOLO11 detection model loaded")
        else:
            print(f"✗ YOLO not found at {yolo_path}")
    except Exception as e:
        print(f"Warning: Error loading detection model: {e}")
        traceback.print_exc()
    
    print(f"\nModels loaded: {len(classification_models)} classification models, Detection: {detection_model is not None}")


def validate_dicom(file_path):
    """Validate if file is a proper DICOM file and check if it's a spine image"""
    try:
        ds = pydicom.dcmread(file_path)
        
        # Check if it's a spine image by examining DICOM metadata
        body_part = str(getattr(ds, 'BodyPartExamined', '')).upper()
        study_description = str(getattr(ds, 'StudyDescription', '')).upper()
        series_description = str(getattr(ds, 'SeriesDescription', '')).upper()
        
        # Look for spine-related keywords
        spine_keywords = ['SPINE', 'VERTEBRA', 'LUMBAR', 'THORACIC', 'CERVICAL', 'SPINAL', 
                         'C-SPINE', 'L-SPINE', 'T-SPINE', 'DORSAL', 'SACRAL', 'COCCYX']
        
        # Look for non-spine keywords (things we definitely don't want)
        non_spine_keywords = ['CHEST', 'LUNG', 'BRAIN', 'HEAD', 'SKULL', 'ABDOMEN', 
                             'PELVIS', 'LEG', 'ARM', 'HAND', 'FOOT', 'KNEE', 'SHOULDER',
                             'ELBOW', 'WRIST', 'ANKLE', 'CARDIAC', 'HEART']
        
        # Check if any spine keyword is present
        all_text = f"{body_part} {study_description} {series_description}"
        
        # Check for non-spine keywords first (more restrictive)
        has_non_spine = any(keyword in all_text for keyword in non_spine_keywords)
        has_spine = any(keyword in all_text for keyword in spine_keywords)
        
        # If it explicitly has non-spine keywords, mark as not spine
        # If it has spine keywords, mark as spine
        # If no keywords at all, assume it might be spine (lenient for unlabeled files)
        if has_non_spine:
            is_spine = False
        elif has_spine:
            is_spine = True
        else:
            # No clear indicators - assume it's okay (lenient mode)
            is_spine = True
        
        return True, ds, is_spine, body_part, str(getattr(ds, 'Modality', 'Unknown'))
    except Exception as e:
        return False, str(e), False, '', ''


def preprocess_dicom(dicom_path):
    """Read and preprocess DICOM image"""
    # Configure pydicom to use pylibjpeg for JPEG 2000 (better handling)
    pydicom.config.pixel_data_handlers = [pylibjpeg_handler]
    
    # Suppress JPEG 2000 bit depth warnings (these are harmless but noisy)
    warnings.filterwarnings('ignore', message='.*Bits Stored.*', category=UserWarning)
    
    # Validate DICOM and check if it's a spine image
    is_valid, result, is_spine, body_part, modality = validate_dicom(dicom_path)
    
    if not is_valid:
        raise ValueError(f"Invalid DICOM file: {result}")
    
    # Check if it's a spine image
    if not is_spine:
        if body_part and body_part != 'UNKNOWN' and body_part != '':
            raise ValueError(f"This appears to be a {body_part} {modality} scan. This system only analyzes spine X-rays and MRIs. Please upload a spine-related DICOM file.")
        else:
            raise ValueError("This DICOM file does not appear to be a spine X-ray or MRI. This system is specifically designed for spine analysis only.")
    
    ds = result
    pixel_array = ds.pixel_array
    
    # Handle MONOCHROME1 (inverted grayscale)
    # MONOCHROME1 means "0 is White", we want "0 is Black"
    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.amax(pixel_array) - pixel_array
    
    # Robust normalization to 0-255 range (fixes display issues on Linux/deployment)
    # Convert to float for precise calculations
    pixel_array = pixel_array.astype(np.float64)
    
    # Remove negative values and normalize
    pixel_array = np.maximum(pixel_array, 0)
    
    # Normalize to 0-255 range
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    
    if pixel_max > pixel_min:  # Avoid division by zero
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)
    
    # Convert to uint8 for consistency
    pixel_array = np.uint8(pixel_array)
    
    return pixel_array, ds


def classify_image(pixel_array):
    """Run ensemble classification"""
    # Transform for models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert grayscale to RGB
    if len(pixel_array.shape) == 2:
        rgb_image = np.stack([pixel_array] * 3, axis=-1)
    else:
        rgb_image = pixel_array
    
    pil_image = Image.fromarray(rgb_image)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Get predictions from each model
    predictions = {}
    with torch.no_grad():
        if 'densenet121' in classification_models:
            pred = torch.sigmoid(classification_models['densenet121'](input_tensor)).item()
            predictions['densenet121'] = pred
        
        if 'resnet50' in classification_models:
            pred = torch.sigmoid(classification_models['resnet50'](input_tensor)).item()
            predictions['resnet50'] = pred
        
        if 'efficientnet' in classification_models:
            pred = torch.sigmoid(classification_models['efficientnet'](input_tensor)).item()
            predictions['efficientnet'] = pred
    
    # Ensemble prediction
    ensemble_score = 0
    total_weight = 0
    for model_name, weight in ENSEMBLE_WEIGHTS.items():
        if model_name in predictions:
            ensemble_score += predictions[model_name] * weight
            total_weight += weight
    
    if total_weight > 0:
        ensemble_score /= total_weight
    
    is_abnormal = ensemble_score > CLASSIFICATION_THRESHOLD
    
    return {
        'ensemble_score': float(ensemble_score),
        'is_abnormal': bool(is_abnormal),
        'confidence': float(max(ensemble_score, 1 - ensemble_score) * 100),
        'individual_predictions': predictions
    }


def detect_lesions(pixel_array):
    """Run YOLO detection"""
    if detection_model is None:
        return None
    
    # Convert to RGB for YOLO
    if len(pixel_array.shape) == 2:
        rgb_image = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = pixel_array
    
    # Run detection
    results = detection_model(rgb_image, conf=0.25)
    
    detections = []
    annotated_image = rgb_image.copy()
    
    if len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Get class name if available
                class_name = result.names[cls] if hasattr(result, 'names') else f"Class_{cls}"
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': class_name
                })
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(annotated_image, f"{class_name} {conf:.2f}", 
                           (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 0, 0), 2)
    
    # Convert annotated image to base64
    _, buffer = cv2.imencode('.png', annotated_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'num_detections': len(detections),
        'detections': detections,
        'annotated_image': img_base64
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Processing file: {filename}")
        # Validate DICOM - validation is now done in preprocess_dicom
        # Just do a quick check if file can be read
        try:
            pydicom.dcmread(filepath)
        except Exception as e:
            os.remove(filepath)
            return jsonify({
                'error': 'Invalid file format',
                'message': 'Please upload a valid DICOM (.dcm or .dicom) file',
                'details': str(e)
            }), 400
        
        # Process DICOM
        try:
            pixel_array, dicom_data = preprocess_dicom(filepath)
        except ValueError as ve:
            # Handle non-spine DICOM files
            os.remove(filepath)
            error_message = str(ve)
            return jsonify({
                'error': 'Invalid spine DICOM',
                'message': error_message
            }), 400
        
        # Run classification
        classification_result = classify_image(pixel_array)
        
        # Run detection if abnormal
        detection_result = None
        if classification_result['is_abnormal']:
            detection_result = detect_lesions(pixel_array)
        
        # Create preview image
        _, buffer = cv2.imencode('.png', pixel_array)
        preview_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get DICOM metadata
        metadata = {
            'patient_id': str(dicom_data.PatientID) if hasattr(dicom_data, 'PatientID') else 'N/A',
            'study_date': str(dicom_data.StudyDate) if hasattr(dicom_data, 'StudyDate') else 'N/A',
            'modality': str(dicom_data.Modality) if hasattr(dicom_data, 'Modality') else 'N/A',
            'image_size': f"{pixel_array.shape[1]}x{pixel_array.shape[0]}"
        }
        
        result = {
            'success': True,
            'filename': filename,
            'metadata': metadata,
            'classification': classification_result,
            'detection': detection_result,
            'preview_image': preview_base64
        }
        
        # Clean up
        os.remove(filepath)
        
        print(f"Successfully processed {filename}")
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        # Log the full error
        print(f"Error processing upload: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'error': 'Processing failed',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/health')
def health():
    model_info = {
        'classification_models': list(classification_models.keys()),
        'detection_model_loaded': detection_model is not None,
        'device': str(device),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    }
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'classification': len(classification_models),
            'detection': detection_model is not None
        },
        'details': model_info
    })


if __name__ == '__main__':
    load_models()
    # Use environment variables for production
    import os
    port = int(os.environ.get('PORT', 5000))
    # Disable debug mode to avoid auto-reloader issues
    app.run(debug=False, host='0.0.0.0', port=port)


# Vision Service - Plant Disease Detection API
# Port: 5001
# Model: ViT-B/16 Finetuned

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import traceback

# ============================================================================
# MODEL DEFINITION - ViT
# ============================================================================
class SimpleViT(nn.Module):
    def __init__(self, num_classes):
        super(SimpleViT, self).__init__()
        self.vit = models.vit_b_16()
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = "./modelsCP/vit_finetune/vit_finetune_best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global model variables
vision_model = None
class_names = []
model_accuracy = 0.0

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_vision_model():
    global vision_model, class_names, model_accuracy
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[VISION SERVICE] ERROR: Model checkpoint not found at {CHECKPOINT_PATH}")
        return False
    
    try:
        print(f"[VISION SERVICE] Loading ViT model from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        num_classes = checkpoint['num_classes']
        vision_model = SimpleViT(num_classes=num_classes).to(DEVICE)
        vision_model.load_state_dict(checkpoint['model_state_dict'])
        vision_model.eval()
        
        class_names = checkpoint['class_names']
        model_accuracy = checkpoint.get('accuracy', 0.0)
        
        print(f"[VISION SERVICE] Model loaded successfully!")
        print(f"[VISION SERVICE] - Accuracy: {model_accuracy:.2f}%")
        print(f"[VISION SERVICE] - Classes: {num_classes}")
        print(f"[VISION SERVICE] - Device: {DEVICE}")
        return True
        
    except Exception as e:
        print(f"[VISION SERVICE] ERROR loading model: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_disease_name(raw_name):
    """Clean disease name for better readability"""
    if not raw_name:
        return None
    
    # Replace underscores with spaces
    cleaned = raw_name.replace('___', ' - ').replace('_', ' ')
    
    # Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned.strip()

def preprocess_image(image_path):
    """Preprocess image for ViT model"""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        return tensor
        
    except Exception as e:
        print(f"[VISION SERVICE] Error preprocessing image: {e}")
        return None

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'vision',
        'model_loaded': vision_model is not None,
        'device': str(DEVICE),
        'accuracy': model_accuracy
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main API endpoint - Receives image and/or text, returns AI response
    
    Input (multipart/form-data):
        - image: File (optional) - Plant disease image
        - query: String (optional) - User question text
    
    Output:
        {
            "success": true/false,
            "vision_result": {
                "disease": "cleaned disease name",
                "confidence": 0.95
            },
            "ai_response": "Full answer from LLM",
            "error": "error message (if failed)"
        }
    
    Flow:
        1. If image provided: Vision predicts disease
        2. Format query with disease context (if available)
        3. Call LLM Service with formatted query
        4. Return combined result
    """
    import requests
    
    LLM_SERVICE_URL = "http://localhost:5002/get_rag_response"
    
    # Get inputs
    image_file = request.files.get('image')
    user_query = request.form.get('query', '').strip()
    
    # Validate inputs
    if not image_file and not user_query:
        return jsonify({
            'success': False,
            'error': 'Either image or query must be provided'
        }), 400
    
    vision_result = None
    identified_disease = None
    
    # Step 1: Process image if provided
    if image_file and image_file.filename != '':
        if not allowed_file(image_file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'
            }), 400
        
        if vision_model is None:
            return jsonify({
                'success': False,
                'error': 'Vision model not loaded'
            }), 500
        
        try:
            # Save temporary file
            temp_path = f"temp_{os.urandom(8).hex()}.jpg"
            image_file.save(temp_path)
            
            # Preprocess and predict
            tensor = preprocess_image(temp_path)
            
            if tensor is None:
                os.remove(temp_path)
                return jsonify({
                    'success': False,
                    'error': 'Failed to preprocess image'
                }), 500
            
            # Predict disease
            with torch.no_grad():
                outputs = vision_model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                pred_idx = predicted_idx.item()
                confidence_score = confidence.item()
                raw_disease = class_names[pred_idx]
                identified_disease = clean_disease_name(raw_disease)
            
            # Clean up temp file
            os.remove(temp_path)
            
            vision_result = {
                'disease': identified_disease,
                'raw_disease': raw_disease,
                'confidence': float(confidence_score)
            }
            
            print(f"[VISION SERVICE] Predicted: {identified_disease} ({confidence_score:.4f})")
            
        except Exception as e:
            # Clean up temp file if exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(f"[VISION SERVICE] Error during prediction: {e}")
            traceback.print_exc()
            
            return jsonify({
                'success': False,
                'error': f'Vision prediction failed: {str(e)}'
            }), 500
    
    # Step 2: Format query for LLM
    # If no query provided but disease identified, create default query
    if not user_query and identified_disease:
        user_query = "Thong tin ve benh nay"
    
    if not user_query:
        return jsonify({
            'success': False,
            'error': 'Query is required when no image is provided'
        }), 400
    
    # Step 3: Call LLM Service
    try:
        print(f"[VISION SERVICE] Calling LLM Service with query: '{user_query}'")
        if identified_disease:
            print(f"[VISION SERVICE] Disease context: '{identified_disease}'")
        
        llm_payload = {
            'query': user_query,
            'identified_disease': identified_disease
        }
        
        llm_response = requests.post(
            LLM_SERVICE_URL,
            json=llm_payload,
            headers={'Content-Type': 'application/json'},
            timeout=190
        )
        
        if llm_response.status_code != 200:
            print(f"[VISION SERVICE] LLM Service error: {llm_response.status_code}")
            return jsonify({
                'success': False,
                'vision_result': vision_result,
                'error': f'LLM Service returned status {llm_response.status_code}'
            }), 500
        
        llm_data = llm_response.json()
        
        if not llm_data.get('success'):
            print(f"[VISION SERVICE] LLM Service failed: {llm_data.get('error')}")
            return jsonify({
                'success': False,
                'vision_result': vision_result,
                'error': f"LLM Service error: {llm_data.get('error')}"
            }), 500
        
        ai_response = llm_data.get('answer', 'No response from LLM')
        
        print(f"[VISION SERVICE] LLM response received (length: {len(ai_response)} chars)")
        
        # Step 4: Return combined result
        return jsonify({
            'success': True,
            'vision_result': vision_result,
            'ai_response': ai_response
        })
        
    except requests.exceptions.Timeout:
        print(f"[VISION SERVICE] LLM Service timeout")
        return jsonify({
            'success': False,
            'vision_result': vision_result,
            'error': 'LLM Service timeout'
        }), 504
        
    except requests.exceptions.ConnectionError:
        print(f"[VISION SERVICE] Cannot connect to LLM Service")
        return jsonify({
            'success': False,
            'vision_result': vision_result,
            'error': 'Cannot connect to LLM Service. Make sure it is running on port 5002.'
        }), 503
        
    except Exception as e:
        print(f"[VISION SERVICE] Error calling LLM Service: {e}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'vision_result': vision_result,
            'error': f'Error calling LLM Service: {str(e)}'
        }), 500

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("VISION SERVICE - Plant Disease Detection")
    print("=" * 60)
    
    # Load model
    if not load_vision_model():
        print("[VISION SERVICE] CRITICAL: Failed to load model. Exiting.")
        exit(1)
    
    print("\n[VISION SERVICE] Starting Flask server on port 5003...")
    print("[VISION SERVICE] API Endpoints:")
    print("  - GET  /health  : Health check")
    print("  - POST /predict : Predict disease from image")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5003)

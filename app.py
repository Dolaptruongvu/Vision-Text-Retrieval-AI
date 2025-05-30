import os
import sys
import uuid 
import re 
import requests 

# --- Thêm đường dẫn ---
current_script_directory = os.path.dirname(os.path.abspath(__file__))
path_to_k_directory_containing_cai = os.path.join(current_script_directory, 'k')
if path_to_k_directory_containing_cai not in sys.path:
    sys.path.insert(0, path_to_k_directory_containing_cai)
    print(f"FRONTEND: Đã thêm '{path_to_k_directory_containing_cai}' vào sys.path")

import numpy as np
from PIL import Image
from tensorflow import keras
import cai.datasets
import cai.layers

from flask import Flask, request, render_template, session, redirect, url_for

# --- Cấu hình ---
UPLOAD_FOLDER = 'uploads' 
STATIC_FOLDER = 'static' 
app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = os.path.join(STATIC_FOLDER, UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
RAG_API_URL = "http://localhost:5002/get_rag_response" 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.secret_key = os.urandom(24)

# --- Model Vision ---
INPUT_SHAPE = (160, 160, 3)
CLASSES = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight',  
    'Potato___healthy',      
    'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 
    'Tomato_Late_blight',    
    'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'     
]
MODEL_PATH = os.path.join(current_script_directory, "modelsCP", "two-path-inception-v2.7-True-0.2-best_result.keras")
vision_model = None

def load_vision_model():
    global vision_model
    if vision_model is None:
        if not os.path.exists(MODEL_PATH):
            print(f"FRONTEND CRITICAL ERROR: Vision Model file not found at {MODEL_PATH}")
            return None
        print(f"FRONTEND: Loading Vision Keras model from {MODEL_PATH}...")
        try:
            custom_objects = {'CopyChannels': cai.layers.CopyChannels}
            vision_model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            print("FRONTEND: Vision Model loaded successfully.")
        except Exception as e:
            print(f"FRONTEND: Error loading vision model: {e}")
            import traceback
            traceback.print_exc()
    return vision_model

vision_model = load_vision_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path_for_cai, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1])):
    try:
        img_input_array = cai.datasets.load_images_from_files(
            file_names=[image_path_for_cai], target_size=target_size,
            lab=True, smart_resize=True, rescale=True, bipolar=False)
        return img_input_array
    except Exception as e:
        print(f"FRONTEND: Error during CAI preprocessing: {e}")
        return None

def clean_disease_name(raw_name):
    if not raw_name: return None
    name = re.sub(r'\([^)]*\)', '', raw_name) 
    name = name.replace('___', ' ').replace('_', ' ')
    name = ' '.join(name.split())
    return name.strip()

@app.route('/', methods=['GET', 'POST'])
def chat_interface():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'session_id' not in session: 
        session['session_id'] = str(uuid.uuid4())
        print(f"FRONTEND: New session started: {session['session_id']}")

    # In ra lịch sử chat ở đầu mỗi request (cả GET và POST) để debug
    # print(f"FRONTEND DEBUG: Current chat_history at start of request: {session.get('chat_history')}")


    if request.method == 'POST':
        user_text_query_original = request.form.get('user_query', '').strip()
        image_file = request.files.get('image_file')
        
        print(f"FRONTEND DEBUG: POST received - User Text: '{user_text_query_original}', Image File: {image_file.filename if image_file else 'None'}")

        current_interaction = {
            'user_query': user_text_query_original,
            'uploaded_image_url': None,
            'original_image_filename': None,
            'vision_analysis': None,
            'ai_response': None 
        }
        
        raw_disease_name_from_vision = None
        cleaned_disease_name_for_rag = None
        
        upload_folder_abs = os.path.join(current_script_directory, app.static_folder, UPLOAD_FOLDER)
        os.makedirs(upload_folder_abs, exist_ok=True)

        if image_file and image_file.filename != '':
            current_interaction['original_image_filename'] = image_file.filename
            if allowed_file(image_file.filename):
                unique_filename_ext = os.path.splitext(image_file.filename)[1]
                unique_filename_on_server = str(uuid.uuid4()) + unique_filename_ext
                filename_on_server_path = os.path.join(upload_folder_abs, unique_filename_on_server)
                
                try:
                    image_file.save(filename_on_server_path)
                    current_interaction['uploaded_image_url'] = url_for('static', filename=os.path.join(UPLOAD_FOLDER, unique_filename_on_server))
                    print(f"FRONTEND DEBUG: Image saved. URL: {current_interaction['uploaded_image_url']}")
                    
                    img_array = preprocess_image(filename_on_server_path, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))
                    if img_array is not None and vision_model is not None:
                        pred = vision_model.predict(img_array)
                        predicted_idx = np.argmax(pred[0])
                        raw_disease_name_from_vision = CLASSES[predicted_idx]
                        cleaned_disease_name_for_rag = clean_disease_name(raw_disease_name_from_vision)
                        confidence = pred[0][predicted_idx]
                        current_interaction['vision_analysis'] = f"Phân tích hình ảnh: {cleaned_disease_name_for_rag} (Độ tin cậy: {confidence:.4f})."
                        print(f"FRONTEND DEBUG: Vision analysis: {current_interaction['vision_analysis']}")
                    elif vision_model is None: current_interaction['vision_analysis'] = "Model vision chưa được tải."
                    else: current_interaction['vision_analysis'] = "Lỗi tiền xử lý ảnh."
                except Exception as e:
                    print(f"FRONTEND ERROR: Saving or processing image: {e}")
                    current_interaction['vision_analysis'] = "Lỗi xử lý file ảnh."
            else:
                current_interaction['vision_analysis'] = "Loại tệp ảnh không hợp lệ."
        
        query_to_rag_service = user_text_query_original
        if not user_text_query_original.strip() and cleaned_disease_name_for_rag:
             query_to_rag_service = f"Thông tin về bệnh {cleaned_disease_name_for_rag}"
        
        # Chỉ gọi RAG service nếu thực sự có gì đó để hỏi
        if query_to_rag_service: 
            try:
                payload = {
                    'query': query_to_rag_service,
                    'identified_disease': cleaned_disease_name_for_rag, 
                }
                print(f"FRONTEND DEBUG: Sending to RAG API -> Payload: {payload}")
                response_from_rag = requests.post(RAG_API_URL, json=payload, timeout=190) 
                response_from_rag.raise_for_status()
                rag_json_response = response_from_rag.json()
                current_interaction['ai_response'] = rag_json_response.get("answer", "Không nhận được phản hồi hợp lệ từ RAG service.")
                print(f"FRONTEND DEBUG: Received from RAG API: {current_interaction['ai_response']}")
            except Exception as e: 
                error_msg = f"Lỗi khi giao tiếp với dịch vụ AI: {e}"
                print(f"FRONTEND ERROR: Calling RAG API: {error_msg}")
                current_interaction['ai_response'] = error_msg
        elif not user_text_query_original and not image_file: 
            current_interaction['user_query'] = "" 
            current_interaction['ai_response'] = "Chào bạn! Vui lòng nhập câu hỏi hoặc tải ảnh lên."
            print(f"FRONTEND DEBUG: No input from user. Default AI response: {current_interaction['ai_response']}")
        else: # Có ảnh nhưng không tạo được query_to_rag_service (ví dụ ảnh lỗi và không có text)
            current_interaction['ai_response'] = "Không có đủ thông tin để xử lý yêu cầu từ ảnh và/hoặc văn bản."
            print(f"FRONTEND DEBUG: Insufficient info for RAG. AI response: {current_interaction['ai_response']}")

        
        # Chỉ thêm vào lịch sử nếu có input ban đầu từ người dùng (text hoặc ảnh hợp lệ)
        if user_text_query_original or current_interaction.get('original_image_filename'):
            session['chat_history'].append(current_interaction)
            session.modified = True
            print(f"FRONTEND DEBUG: Interaction appended to chat_history. New history length: {len(session['chat_history'])}")
        else:
            print("FRONTEND DEBUG: No significant user input, not adding to chat_history this turn.")
        
        return redirect(url_for('chat_interface'))

    # Cho cả GET và POST (sau redirect), truyền lịch sử chat cho template
    chat_history_to_render = session.get('chat_history', [])
    print(f"FRONTEND DEBUG: Rendering template with chat_history_display: {chat_history_to_render}")
    return render_template('index.html', chat_history_display=chat_history_to_render)

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session.pop('chat_history', None) 
    session.pop('session_id', None) 
    print("FRONTEND: Chat history cleared.")
    return redirect(url_for('chat_interface'))

if __name__ == '__main__':
    if vision_model is None:
        print("FRONTEND CRITICAL: Vision Model không thể tải. Ứng dụng sẽ thoát.")
        exit()
    static_uploads_dir = os.path.join(current_script_directory, app.static_folder, UPLOAD_FOLDER)
    if not os.path.exists(static_uploads_dir):
        os.makedirs(static_uploads_dir)
        print(f"Created directory: {static_uploads_dir}")
    app.run(debug=True, host='0.0.0.0', port=5001)
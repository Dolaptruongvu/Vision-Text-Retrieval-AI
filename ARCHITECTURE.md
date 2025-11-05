# Backend API Architecture

## 🏗️ System Architecture

### NEW Architecture (Correct Flow)
```
┌──────────────┐
│   Frontend   │ (React + Vite)
│ Port: 5173   │
└──────┬───────┘
       │ HTTP + JWT Token
       ▼
┌──────────────┐
│  Backend API │ (Node.js + Express)
│ Port: 5004   │
│              │
│ - Verify JWT │
│ - Upload file│
│ - Forward    │
└──────┬───────┘
       │ HTTP + X-Internal-Request header
       ▼
┌──────────────┐
│Vision Service│ (Python + Flask)
│ Port: 5003   │
│              │
│ - Process img│
│ - Predict    │
└──────┬───────┘
       │ HTTP
       ▼
┌──────────────┐
│ LLM Service  │ (Python + Flask + RAG)
│ Port: 5002   │
│              │
│ - RAG query  │
│ - Generate   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   MongoDB    │
│ Port: 27017  │
│              │
│ - Save result│
└──────────────┘
```

### OLD Architecture (Incorrect - Direct Call)
```
Frontend → Vision Service (WRONG! No centralized auth, hard to manage)
```

---

## 📡 API Flow

### 1. **POST /api/disease/predict** (NEW)

**Request Flow:**
```
1. Frontend sends FormData (image + query) with JWT token in Authorization header
2. Backend receives request
   - Middleware: protect() verifies JWT token
   - Middleware: multer handles file upload
3. Backend forwards to Vision Service
   - Adds X-Internal-Request: true header
   - Adds userId and username from JWT to FormData
4. Vision Service processes
   - Detects internal request (skips JWT verification)
   - Predicts disease from image
   - Calls LLM Service for AI response
   - Saves to MongoDB
5. Backend receives result
   - Cleans up uploaded file
   - Returns to frontend
```

**Request (Frontend):**
```javascript
// FormData
const formData = new FormData();
formData.append('image', file);  // Optional
formData.append('query', text);  // Optional

// Headers (auto-injected by axios interceptor)
Authorization: Bearer <JWT_TOKEN>

// Endpoint
POST http://localhost:5004/api/disease/predict
```

**Response:**
```json
{
  "success": true,
  "data": {
    "vision_result": {
      "disease": "Late Blight (Bệnh Mốc Sương)",
      "raw_disease": "Tomato___Late_blight",
      "confidence": 0.9876
    },
    "ai_response": "Bệnh mốc sương (Late Blight) là...",
    "saved_to_db": true,
    "record_id": "507f1f77bcf86cd799439011"
  }
}
```

---

## 🔐 Authentication Flow

### Vision Service Authentication Logic
```python
def is_internal_request(request):
    """Check if request is from Backend (internal)"""
    internal_header = request.headers.get('X-Internal-Request', '').lower()
    return internal_header == 'true'

# In predict endpoint:
if is_internal_request(request):
    # Trust user info from Backend (already verified JWT)
    user_id = request.form.get('userId')
    username = request.form.get('username')
else:
    # External request - verify JWT token (backward compatibility)
    verify_user_token(token)
```

**Why this approach?**
- ✅ **Centralized auth**: Backend verifies JWT once, no redundant checks
- ✅ **Secure**: Vision Service only trusts requests with X-Internal-Request header
- ✅ **Flexible**: Still supports direct calls (for testing) with JWT
- ✅ **Clean**: Vision Service focuses on prediction, Backend handles auth

---

## 📁 File Upload Handling

### Backend (diseaseController.js)
```javascript
// Multer saves file to uploads/ folder
const imageFile = req.file; 
// { path, originalname, mimetype, size }

// Forward to Vision Service
const formData = new FormData();
formData.append('image', fs.createReadStream(imageFile.path), {
  filename: imageFile.originalname,
  contentType: imageFile.mimetype
});

// Cleanup after response
if (fs.existsSync(imageFile.path)) {
  fs.unlinkSync(imageFile.path);
}
```

### Vision Service (visionService.py)
```python
# Receive file from Backend
image_file = request.files.get('image')

# Save temporarily for processing
temp_path = f"temp_{os.urandom(8).hex()}.jpg"
image_file.save(temp_path)

# Process with model
tensor = preprocess_image(temp_path)
outputs = vision_model(tensor)

# Cleanup
os.remove(temp_path)
```

---

## 🧪 Testing

### Test with testAPI.py
```bash
cd ThesisLLM
python testAPI.py
```

**Flow:**
```
1. Login → Get JWT token
2. Get monthly stats (before)
3. For each test case:
   - Call Backend API with JWT
   - Backend forwards to Vision
   - Vision predicts + calls LLM
   - Save to MongoDB
   - Return result
4. Get monthly stats (after)
5. Show trend changes
```

### Test with Frontend
```bash
# Terminal 1: Backend
cd ThesisApp/ThesisBE
npm run dev

# Terminal 2: Vision
cd ThesisVision
python visionService.py

# Terminal 3: LLM
cd ThesisLLM
python RAGserver.py

# Terminal 4: Frontend
cd ThesisApp/ThesisFE
npm run dev

# Open browser: http://localhost:5173
```

---

## ⚠️ Important Notes

### 1. **Uploads Folder**
```bash
ThesisBE/
  uploads/        # MUST exist (created automatically)
  .gitignore      # Add uploads/* to prevent committing uploaded files
```

### 2. **Environment Variables**
```env
# .env
VISION_SERVICE_URL=http://localhost:5003
PORT=5004
```

### 3. **CORS Configuration**
```javascript
// server.js
app.use(cors({
  origin: ['http://localhost:5173'],  // Frontend URL
  credentials: true
}));
```

### 4. **Error Handling**
```javascript
// Backend catches Vision Service errors:
- ECONNABORTED → 504 Gateway Timeout
- ECONNREFUSED → 503 Service Unavailable
- Other errors → 500 Internal Server Error
```

---

## 📊 Benefits of New Architecture

| Aspect | OLD (Direct) | NEW (Through Backend) |
|--------|--------------|----------------------|
| **Authentication** | Vision verifies JWT | Backend verifies JWT (once) |
| **File Upload** | Frontend → Vision | Frontend → Backend → Vision |
| **Error Handling** | Frontend handles | Backend handles + cleans up |
| **Security** | Token exposed to Vision | Token only in Backend |
| **Monitoring** | Hard to track requests | Backend logs all requests |
| **Rate Limiting** | Must implement in Vision | Backend handles rate limiting |
| **CORS** | Vision needs CORS config | Only Backend needs CORS |

---

## 🚀 Deployment Considerations

### Production Setup
```
Frontend (Vercel)
    ↓ HTTPS
Backend (Railway/Heroku) + API Gateway
    ↓ Internal HTTP
Vision Service (Private network)
    ↓ Internal HTTP
LLM Service (Private network)
    ↓ Internal HTTP
MongoDB Atlas
```

**Security in Production:**
- Frontend: Only knows Backend URL
- Vision/LLM: Not exposed to internet
- Backend: Acts as API Gateway
- All internal communication via private network
- JWT tokens never leave Backend ↔ Frontend communication

---

## 📝 Summary

**Key Changes:**
1. ✅ Frontend calls Backend API (not Vision Service directly)
2. ✅ Backend verifies JWT and forwards to Vision Service
3. ✅ Vision Service trusts internal requests from Backend
4. ✅ File uploads handled by Backend with multer
5. ✅ Cleanup and error handling centralized in Backend

**This is the correct microservices architecture!** 🎉

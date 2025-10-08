# 🔐 Vision Service Authentication

## Overview

Vision Service now requires JWT authentication. Users must login to get a token before using the prediction API.

## 🔄 Authentication Flow

```
1. User Login (Backend)
   ↓
2. Receive JWT Token
   ↓
3. Send Token with Prediction Request (Vision Service)
   ↓
4. Vision Service Verifies Token (with Backend)
   ↓
5. Process Prediction
   ↓
6. Save Result to MongoDB (with User ID)
   ↓
7. Return Response
```

## 🚀 Quick Start

### 1. Start All Services

```powershell
# Terminal 1: Start MongoDB
net start MongoDB

# Terminal 2: Start Backend
cd c:\wd\ThesisLLM\ThesisApp\ThesisBE
npm run dev

# Terminal 3: Start LLM Service
cd c:\wd\ThesisLLM\ThesisLLM
python RAGserver.py

# Terminal 4: Start Vision Service
cd c:\wd\ThesisLLM\ThesisVision
python visionService.py
```

### 2. Register User (First Time Only)

```powershell
cd c:\wd\ThesisLLM\ThesisApp\ThesisBE
python test_api.py
# This will register test@example.com / password123
```

### 3. Test Authenticated Prediction

```powershell
# Option 1: Use test script in Vision folder
cd c:\wd\ThesisLLM\ThesisVision
python test_auth_predict.py

# Option 2: Use updated testAPI script
cd c:\wd\ThesisLLM\ThesisLLM
python testAPI.py
```

## 📡 API Usage

### Method 1: Token in Form Data (Recommended for multipart/form-data)

```python
import requests

# 1. Login to get token
response = requests.post(
    'http://localhost:5004/api/auth/login',
    json={
        'email': 'test@example.com',
        'password': 'password123'
    }
)
token = response.json()['data']['token']

# 2. Use token in prediction request
files = {'image': open('plant.jpg', 'rb')}
data = {
    'query': 'What disease is this?',
    'token': token  # Include token here
}

response = requests.post(
    'http://localhost:5003/predict',
    files=files,
    data=data
)

result = response.json()
print(f"Disease: {result['vision_result']['disease']}")
print(f"Saved to DB: {result['saved_to_db']}")
print(f"Record ID: {result['record_id']}")
```

### Method 2: Token in Authorization Header

```python
import requests

# 1. Login to get token
response = requests.post(
    'http://localhost:5004/api/auth/login',
    json={
        'email': 'test@example.com',
        'password': 'password123'
    }
)
token = response.json()['data']['token']

# 2. Use token in header
files = {'image': open('plant.jpg', 'rb')}
data = {'query': 'What disease is this?'}
headers = {'Authorization': f'Bearer {token}'}

response = requests.post(
    'http://localhost:5003/predict',
    files=files,
    data=data,
    headers=headers
)

result = response.json()
```

### Method 3: cURL Examples

```bash
# Login
curl -X POST http://localhost:5004/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Predict with token in form data
curl -X POST http://localhost:5003/predict \
  -F "image=@plant.jpg" \
  -F "query=What disease is this?" \
  -F "token=YOUR_JWT_TOKEN"

# Predict with Authorization header
curl -X POST http://localhost:5003/predict \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "image=@plant.jpg" \
  -F "query=What disease is this?"
```

## 🔒 Response Structure

### Success Response (200)

```json
{
  "success": true,
  "user": {
    "id": "user_id_from_mongodb",
    "username": "testuser",
    "email": "test@example.com"
  },
  "vision_result": {
    "disease": "Tomato - Late blight",
    "confidence": 0.9852,
    "raw_disease": "Tomato___Late_blight"
  },
  "ai_response": "Dựa trên kết quả phân tích hình ảnh...",
  "saved_to_db": true,
  "record_id": "mongodb_record_id"
}
```

### Error Response (401 Unauthorized)

```json
{
  "success": false,
  "error": "Authentication required. Please login first.",
  "message": "Please login first to use the prediction service",
  "code": "AUTHENTICATION_REQUIRED"
}
```

## 🛠️ Configuration

### Disable Authentication (Development Only)

Create `.env` file in ThesisVision:

```env
REQUIRE_AUTH=false
```

**⚠️ WARNING:** Only use this for testing. Never deploy to production without authentication!

### Backend URL Configuration

In `visionService.py`:

```python
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5004')
```

## 🧪 Testing Scripts

### 1. test_auth_predict.py

Complete authentication flow test:
- ✅ Login and get token
- ✅ Test without token (should fail)
- ✅ Test with token in form data (should succeed)
- ✅ Test with Authorization header (should succeed)
- ✅ Verify data saved to MongoDB

```powershell
cd c:\wd\ThesisLLM\ThesisVision
python test_auth_predict.py
```

### 2. testAPI.py (Updated)

Now includes automatic login:
- ✅ Auto-login before tests
- ✅ All test cases use authenticated requests
- ✅ Shows saved records in database

```powershell
cd c:\wd\ThesisLLM\ThesisLLM
python testAPI.py
```

## 🔍 Troubleshooting

### Error: "Authentication required"

**Solution:**
1. Make sure you're sending the token
2. Check token format: `Bearer YOUR_TOKEN` for header, or plain token for form data
3. Verify backend is running on port 5004

### Error: "Invalid or expired token"

**Solution:**
1. Login again to get a fresh token
2. Tokens expire after 7 days (configurable in backend .env)
3. Check if user still exists and is active

### Error: "Authentication service unavailable"

**Solution:**
1. Ensure backend is running: `npm run dev` in ThesisBE
2. Check BACKEND_URL in visionService.py
3. Verify network connectivity between services

## 📊 Database Integration

When authenticated, predictions are automatically saved to MongoDB:

```javascript
{
  _id: ObjectId,
  userId: ObjectId,  // ✅ Linked to authenticated user
  diseaseName: "Tomato - Late blight",
  confidence: 0.9852,
  userQuery: "What disease is this?",
  aiResponse: "...",
  imageName: "plant.jpg",
  processingTime: 1500,
  detectionDate: ISODate(),
  saved_to_db: true
}
```

View your history:
```bash
curl http://localhost:5004/api/disease/history \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 🎯 Benefits

1. **User Tracking**: Know who made which predictions
2. **Personalization**: Tailor responses based on user history
3. **Analytics**: Track usage per user
4. **Security**: Prevent unauthorized API access
5. **Rate Limiting**: Control API usage (future feature)
6. **Audit Trail**: Complete history of predictions

## 📝 Next Steps

- [ ] Implement rate limiting per user
- [ ] Add API key support for external clients
- [ ] Create user dashboard to view history
- [ ] Add prediction analytics
- [ ] Implement user feedback system

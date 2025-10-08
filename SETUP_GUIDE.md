# 🚀 QUICK START GUIDE - ThesisBE Backend

## 📋 Prerequisites

1. **Node.js** (v16 or higher)
2. **MongoDB** (running on localhost:27017)
3. **Python** (for testing script)

## 🛠️ Setup Steps

### 1. Install MongoDB (if not installed)

**Windows:**
```powershell
# Download from: https://www.mongodb.com/try/download/community
# Or use chocolatey:
choco install mongodb

# Start MongoDB service
net start MongoDB
```

### 2. Install Backend Dependencies

```powershell
cd c:\wd\ThesisLLM\ThesisApp\ThesisBE
npm install
```

### 3. Configure Environment

The `.env` file is already created with default settings:
```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/plantAI
JWT_SECRET=your_super_secret_jwt_key_change_this_in_production_12345
JWT_EXPIRE=7d
PORT=5004
```

### 4. Start Backend Server

```powershell
# Development mode (with auto-reload)
npm run dev

# Production mode
npm start
```

Server will start on: **http://localhost:5004**

### 5. Test Backend API

```powershell
# In a new terminal
python test_api.py
```

This will test:
- ✅ Health check
- ✅ User registration
- ✅ User login
- ✅ Get profile
- ✅ Save disease history
- ✅ Get disease history
- ✅ Get statistics
- ✅ Get disease detail

## 📡 API Endpoints

### Authentication

**Register:**
```bash
POST http://localhost:5004/api/auth/register
Content-Type: application/json

{
  "username": "testuser",
  "email": "test@example.com",
  "password": "password123"
}
```

**Login:**
```bash
POST http://localhost:5004/api/auth/login
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "password123"
}
```

**Get Profile:**
```bash
GET http://localhost:5004/api/auth/me
Authorization: Bearer YOUR_JWT_TOKEN
```

### Disease History

**Save Detection Result:**
```bash
POST http://localhost:5004/api/disease/save
Authorization: Bearer YOUR_JWT_TOKEN
Content-Type: application/json

{
  "diseaseName": "Tomato - Late blight",
  "diseaseNameRaw": "Tomato___Late_blight",
  "confidence": 0.9852,
  "userQuery": "What is this disease?",
  "aiResponse": "This is late blight...",
  "imageName": "tomato.jpg",
  "processingTime": 1500
}
```

**Get History:**
```bash
GET http://localhost:5004/api/disease/history?limit=20&page=1
Authorization: Bearer YOUR_JWT_TOKEN
```

**Get Statistics:**
```bash
GET http://localhost:5004/api/disease/stats
Authorization: Bearer YOUR_JWT_TOKEN
```

**Get Detail:**
```bash
GET http://localhost:5004/api/disease/:id
Authorization: Bearer YOUR_JWT_TOKEN
```

## 🔗 Integration with Vision Service

Vision Service automatically saves results to backend when user provides JWT token.

**Example request to Vision Service:**
```python
import requests

files = {'image': open('tomato.jpg', 'rb')}
data = {
    'query': 'What is this disease?',
    'token': 'YOUR_JWT_TOKEN'  # Add this to save to database
}

response = requests.post(
    'http://localhost:5003/predict',
    files=files,
    data=data
)

result = response.json()
print(f"Saved to DB: {result.get('saved_to_db')}")
print(f"Record ID: {result.get('record_id')}")
```

## 🗄️ Database Models

### User
- `username`, `email`, `password` (hashed with bcrypt)
- `role` (user/admin)
- `isActive`, `createdAt`, `lastLogin`

### DiseaseHistory
- `userId` (ref to User)
- `diseaseName`, `diseaseNameRaw`, `confidence`
- `userQuery`, `aiResponse`
- `imageUrl`, `imageName`
- `processingTime`, `detectionDate`
- `userFeedback` (for accuracy tracking)

### ChatSession (future implementation)
- `userId`, `title`
- `messages[]` (role, content, timestamp)
- `isActive`, `isPinned`, `isArchived`

## 📊 MongoDB Commands

```bash
# Connect to MongoDB
mongo

# Use database
use plantAI

# Show collections
show collections

# Find users
db.users.find().pretty()

# Find disease history
db.diseasehistories.find().pretty()

# Count documents
db.diseasehistories.countDocuments()

# Get statistics
db.diseasehistories.aggregate([
  { $group: { _id: "$diseaseName", count: { $sum: 1 } } }
])
```

## 🐛 Troubleshooting

**MongoDB not running:**
```powershell
# Windows
net start MongoDB

# Or manually
mongod --dbpath C:\data\db
```

**Port 5004 already in use:**
```powershell
# Find process
netstat -ano | findstr :5004

# Kill process
taskkill /PID <PID> /F
```

**Cannot connect to database:**
- Check if MongoDB is running
- Verify MONGODB_URI in `.env` file
- Check MongoDB logs

## 🎯 Next Steps

1. ✅ Backend API is ready
2. 🔄 Test integration with Vision Service (port 5003)
3. 📱 Build React frontend (ThesisFE)
4. 📊 Create dashboard for analytics
5. 💬 Implement chat session features

## 📝 Notes

- JWT token expires in 7 days (configurable in `.env`)
- Passwords are hashed with bcrypt (10 salt rounds)
- All disease routes require authentication
- Vision Service integration is optional (works without token too)

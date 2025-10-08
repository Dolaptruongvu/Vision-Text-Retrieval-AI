# ThesisBE - Backend API

Backend service for Plant Disease Detection System with user authentication and data management.

## 🚀 Features

- **User Authentication**: JWT-based login/register
- **Disease History**: Store and retrieve disease detection results
- **Dashboard Analytics**: Statistics and trends for detected diseases
- **Chat Sessions**: Store conversation history (future implementation)

## 📦 Tech Stack

- Node.js + Express.js
- MongoDB + Mongoose
- JWT Authentication
- bcryptjs for password hashing

## 🛠️ Installation

```bash
# Install dependencies
npm install

# Start MongoDB (make sure MongoDB is running on localhost:27017)

# Run development server
npm run dev

# Run production server
npm start
```

## 📝 Environment Variables

Create a `.env` file:

```env
MONGODB_URI=mongodb://localhost:27017/anders
JWT_SECRET=your_secret_key
JWT_EXPIRE=7d
PORT=5004
```

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user profile (Protected)

### Disease History
- `POST /api/disease/save` - Save disease detection result (Protected)
- `GET /api/disease/history` - Get user's disease history (Protected)
- `GET /api/disease/stats` - Get dashboard statistics (Protected)
- `GET /api/disease/:id` - Get specific disease record (Protected)

## 🧪 Testing

### 1. Register User
```bash
curl -X POST http://localhost:5004/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'
```

### 2. Login
```bash
curl -X POST http://localhost:5004/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123"
  }'
```

### 3. Save Disease History (use token from login)
```bash
curl -X POST http://localhost:5004/api/disease/save \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "diseaseName": "Tomato - Late blight",
    "confidence": 0.98,
    "userQuery": "What is this disease?",
    "aiResponse": "This is late blight..."
  }'
```

## 📊 Database Models

### User
- username, email, password (hashed)
- role (user/admin)
- isActive, createdAt, lastLogin

### DiseaseHistory
- userId (ref to User)
- diseaseName, confidence
- userQuery, aiResponse
- imageUrl, imageName
- detectionDate, isReviewed
- userFeedback (for accuracy tracking)

### ChatSession (future)
- userId, title
- messages array (role, content, timestamp)
- isActive, isPinned, isArchived

## 🔗 Integration with Vision Service

Update `visionService.py` to automatically save results to backend after prediction.

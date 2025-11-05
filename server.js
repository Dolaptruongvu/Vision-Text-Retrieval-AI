// Main Server File
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./config/database');

// Load environment variables
dotenv.config();

// Connect to MongoDB
connectDB();

// Initialize Express app
const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging middleware
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/disease', require('./routes/disease'));
app.use('/api/chat', require('./routes/chat'));

// Health check endpoint
app.get('/health', (req, res) => {
  const mongoose = require('mongoose');
  const dbState = mongoose.connection.readyState;
  const dbStates = {
    0: 'Disconnected',
    1: 'Connected',
    2: 'Connecting',
    3: 'Disconnecting'
  };

  res.json({
    success: true,
    message: 'ThesisBE API is running',
    timestamp: new Date().toISOString(),
    database: {
      status: dbStates[dbState] || 'Unknown',
      name: mongoose.connection.name || 'N/A',
      host: mongoose.connection.host || 'N/A'
    },
    server: {
      port: process.env.PORT,
      environment: process.env.NODE_ENV
    }
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'Welcome to ThesisBE API',
    version: '1.0.0',
    endpoints: {
      auth: {
        register: 'POST /api/auth/register',
        login: 'POST /api/auth/login',
        profile: 'GET /api/auth/me'
      },
      disease: {
        save: 'POST /api/disease/save',
        history: 'GET /api/disease/history',
        stats: 'GET /api/disease/stats',
        detail: 'GET /api/disease/:id'
      }
    }
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found'
  });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || 'Internal server error',
    error: process.env.NODE_ENV === 'development' ? err : {}
  });
});

// Start server
const PORT = process.env.PORT || 5004;
app.listen(PORT, () => {
  console.log('\n' + '='.repeat(60));
  console.log('🚀 THESIS BACKEND API - READY');
  console.log('='.repeat(60));
  console.log(`📡 Server: http://localhost:${PORT}`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV}`);
  console.log(`🗄️  Database URI: ${process.env.MONGODB_URI}`);
  console.log(`🔐 JWT Secret: ${process.env.JWT_SECRET.substring(0, 20)}...`);
  console.log('='.repeat(60));
  console.log('\n📚 API Endpoints:');
  console.log('   Health Check:  GET  http://localhost:' + PORT + '/health');
  console.log('   Register:      POST http://localhost:' + PORT + '/api/auth/register');
  console.log('   Login:         POST http://localhost:' + PORT + '/api/auth/login');
  console.log('   Profile:       GET  http://localhost:' + PORT + '/api/auth/me');
  console.log('   Save Disease:  POST http://localhost:' + PORT + '/api/disease/save');
  console.log('   Get History:   GET  http://localhost:' + PORT + '/api/disease/history');
  console.log('   Get Stats:     GET  http://localhost:' + PORT + '/api/disease/stats');
  console.log('='.repeat(60) + '\n');
});

// Vision Auth Middleware - Verify JWT token with Backend
const verifyToken = async (req, res, next) => {
  const axios = require('axios');
  const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5004';
  
  try {
    // Get token from request
    let token = null;
    
    // Check Authorization header
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    }
    // Check form data (for multipart requests)
    else if (req.body && req.body.token) {
      token = req.body.token;
    }
    
    if (!token) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required. Please login first.',
        code: 'NO_TOKEN'
      });
    }
    
    // Verify token with backend
    const response = await axios.get(`${BACKEND_URL}/api/auth/me`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.data.success) {
      // Attach user info to request
      req.user = response.data.data.user;
      req.token = token;
      next();
    } else {
      return res.status(401).json({
        success: false,
        error: 'Invalid token',
        code: 'INVALID_TOKEN'
      });
    }
    
  } catch (error) {
    console.error('[VISION SERVICE] Token verification failed:', error.message);
    
    if (error.response && error.response.status === 401) {
      return res.status(401).json({
        success: false,
        error: 'Invalid or expired token. Please login again.',
        code: 'TOKEN_EXPIRED'
      });
    }
    
    return res.status(500).json({
      success: false,
      error: 'Authentication service unavailable',
      code: 'AUTH_SERVICE_ERROR'
    });
  }
};

module.exports = { verifyToken };

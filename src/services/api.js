import axios from 'axios';

const API_BASE_URL = 'http://localhost:5004/api';
const VISION_SERVICE_URL = 'http://localhost:5003';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 200000, // 200 seconds for long LLM responses
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ============================================================================
// AUTH APIs
// ============================================================================
export const authAPI = {
  register: async (userData) => {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },

  login: async (credentials) => {
    const response = await api.post('/auth/login', credentials);
    if (response.data.success) {
      localStorage.setItem('token', response.data.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.data.user));
    }
    return response.data;
  },

  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },

  getProfile: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },
};

// ============================================================================
// VISION + LLM APIs (Chat)
// ============================================================================
export const chatAPI = {
  predict: async (formData) => {
    // NEW ARCHITECTURE: Frontend → Backend → Vision Service → LLM Service
    // Backend API handles authentication and forwards to Vision Service
    const response = await api.post('/disease/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 200000, // 200 seconds for LLM processing
    });
    return response.data;
  },
};

// ============================================================================
// DISEASE HISTORY APIs
// ============================================================================
export const diseaseAPI = {
  getHistory: async (params = {}) => {
    const response = await api.get('/disease/history', { params });
    return response.data;
  },

  getStats: async () => {
    const response = await api.get('/disease/stats');
    return response.data;
  },

  getMonthlyStats: async (year, month) => {
    const params = {};
    if (year) params.year = year;
    if (month) params.month = month;
    const response = await api.get('/disease/monthly-stats', { params });
    return response.data;
  },

  getDetail: async (id) => {
    const response = await api.get(`/disease/${id}`);
    return response.data;
  },
};

// ============================================================================
// CHAT SESSION APIs
// ============================================================================
export const sessionAPI = {
  // Get all sessions
  getSessions: async () => {
    const response = await api.get('/chat/sessions');
    return response.data;
  },

  // Get single session
  getSession: async (id) => {
    const response = await api.get(`/chat/sessions/${id}`);
    return response.data;
  },

  // Create new session
  createSession: async (title = 'New Chat') => {
    const response = await api.post('/chat/sessions', { title });
    return response.data;
  },

  // Add message to session
  addMessage: async (sessionId, message) => {
    const response = await api.post(`/chat/sessions/${sessionId}/messages`, message);
    return response.data;
  },

  // Delete session
  deleteSession: async (id) => {
    const response = await api.delete(`/chat/sessions/${id}`);
    return response.data;
  },

  // Clear session messages
  clearSession: async (id) => {
    const response = await api.put(`/chat/sessions/${id}/clear`);
    return response.data;
  },
};

export default api;

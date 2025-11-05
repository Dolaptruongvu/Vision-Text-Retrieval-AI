// Chat Session Routes
const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth');
const {
  getSessions,
  getSession,
  createSession,
  addMessage,
  deleteSession,
  clearSession
} = require('../controllers/chatController');

// All routes require authentication
router.use(protect);

// Session CRUD
router.get('/sessions', getSessions);
router.post('/sessions', createSession);
router.get('/sessions/:id', getSession);
router.delete('/sessions/:id', deleteSession);

// Session operations
router.post('/sessions/:id/messages', addMessage);
router.put('/sessions/:id/clear', clearSession);

module.exports = router;

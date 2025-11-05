// Chat Session Controller
const ChatSession = require('../models/ChatSession');
const mongoose = require('mongoose');

// @desc    Get all chat sessions for user
// @route   GET /api/chat/sessions
// @access  Private
exports.getSessions = async (req, res) => {
  try {
    const sessions = await ChatSession.find({ userId: req.user.id })
      .select('title lastMessageAt messages isActive')
      .sort('-lastMessageAt')
      .limit(50);

    res.status(200).json({
      success: true,
      data: sessions.map(s => ({
        id: s._id,
        title: s.title,
        lastMessageAt: s.lastMessageAt,
        messageCount: s.messages.length,
        isActive: s.isActive,
        preview: s.messages.length > 0 ? s.messages[s.messages.length - 1].content.substring(0, 50) : ''
      }))
    });
  } catch (error) {
    console.error('Get sessions error:', error);
    res.status(500).json({ success: false, message: 'Error fetching sessions' });
  }
};

// @desc    Get single chat session
// @route   GET /api/chat/sessions/:id
// @access  Private
exports.getSession = async (req, res) => {
  try {
    const session = await ChatSession.findOne({
      _id: req.params.id,
      userId: req.user.id
    });

    if (!session) {
      return res.status(404).json({ success: false, message: 'Session not found' });
    }

    res.status(200).json({ success: true, data: session });
  } catch (error) {
    console.error('Get session error:', error);
    res.status(500).json({ success: false, message: 'Error fetching session' });
  }
};

// @desc    Create new chat session
// @route   POST /api/chat/sessions
// @access  Private
exports.createSession = async (req, res) => {
  try {
    const { title } = req.body;

    const session = new ChatSession({
      userId: req.user.id,
      title: title || 'New Chat',
      messages: []
    });

    await session.save();

    res.status(201).json({
      success: true,
      data: {
        id: session._id,
        title: session.title,
        messages: session.messages
      }
    });
  } catch (error) {
    console.error('Create session error:', error);
    res.status(500).json({ success: false, message: 'Error creating session' });
  }
};

// @desc    Add message to session
// @route   POST /api/chat/sessions/:id/messages
// @access  Private
exports.addMessage = async (req, res) => {
  try {
    const { role, content, relatedDiseaseId } = req.body;

    if (!role || !content) {
      return res.status(400).json({ success: false, message: 'Role and content required' });
    }

    const session = await ChatSession.findOne({
      _id: req.params.id,
      userId: req.user.id
    });

    if (!session) {
      return res.status(404).json({ success: false, message: 'Session not found' });
    }

    session.messages.push({ role, content, relatedDiseaseId });
    session.lastMessageAt = new Date();

    // Auto-generate title from first user message
    if (session.messages.length === 1 && role === 'user' && session.title === 'New Chat') {
      session.title = content.substring(0, 50) + (content.length > 50 ? '...' : '');
    }

    await session.save();

    res.status(200).json({ success: true, data: session });
  } catch (error) {
    console.error('Add message error:', error);
    res.status(500).json({ success: false, message: 'Error adding message' });
  }
};

// @desc    Delete chat session
// @route   DELETE /api/chat/sessions/:id
// @access  Private
exports.deleteSession = async (req, res) => {
  try {
    const session = await ChatSession.findOneAndDelete({
      _id: req.params.id,
      userId: req.user.id
    });

    if (!session) {
      return res.status(404).json({ success: false, message: 'Session not found' });
    }

    res.status(200).json({ success: true, message: 'Session deleted' });
  } catch (error) {
    console.error('Delete session error:', error);
    res.status(500).json({ success: false, message: 'Error deleting session' });
  }
};

// @desc    Clear all messages in session
// @route   PUT /api/chat/sessions/:id/clear
// @access  Private
exports.clearSession = async (req, res) => {
  try {
    const session = await ChatSession.findOne({
      _id: req.params.id,
      userId: req.user.id
    });

    if (!session) {
      return res.status(404).json({ success: false, message: 'Session not found' });
    }

    session.messages = [];
    session.title = 'New Chat';
    await session.save();

    res.status(200).json({ success: true, data: session });
  } catch (error) {
    console.error('Clear session error:', error);
    res.status(500).json({ success: false, message: 'Error clearing session' });
  }
};

// Chat Session Model - Store chat conversations like ChatGPT
const mongoose = require('mongoose');

const messageSchema = new mongoose.Schema({
  role: {
    type: String,
    enum: ['user', 'assistant', 'system'],
    required: true
  },
  content: {
    type: String,
    required: true
  },
  timestamp: {
    type: Date,
    default: Date.now
  },
  // Optional: link to disease detection if message is about specific disease
  relatedDiseaseId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'DiseaseHistory',
    default: null
  }
});

const chatSessionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  title: {
    type: String,
    default: 'New Chat',
    maxlength: 100
  },
  messages: [messageSchema],
  // Session metadata
  isActive: {
    type: Boolean,
    default: true
  },
  lastMessageAt: {
    type: Date,
    default: Date.now
  },
  // Summary or context (optional)
  summary: {
    type: String,
    default: null
  },
  tags: [{
    type: String
  }],
  // For organization
  isPinned: {
    type: Boolean,
    default: false
  },
  isArchived: {
    type: Boolean,
    default: false
  }
}, {
  timestamps: true
});

// Index for efficient queries
chatSessionSchema.index({ userId: 1, lastMessageAt: -1 });
chatSessionSchema.index({ userId: 1, isActive: 1 });

// Update lastMessageAt when adding messages
chatSessionSchema.pre('save', function(next) {
  if (this.messages && this.messages.length > 0) {
    this.lastMessageAt = this.messages[this.messages.length - 1].timestamp;
  }
  next();
});

module.exports = mongoose.model('ChatSession', chatSessionSchema);

// Disease History Model - Store prediction results
const mongoose = require('mongoose');

const diseaseHistorySchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  // Vision prediction result
  diseaseName: {
    type: String,
    required: true
  },
  diseaseNameRaw: {
    type: String // Original model output (e.g., "Tomato___Late_blight")
  },
  confidence: {
    type: Number,
    required: true,
    min: 0,
    max: 1
  },
  // User query and AI response
  userQuery: {
    type: String,
    default: ''
  },
  aiResponse: {
    type: String,
    default: ''
  },
  // Image info (optional - can store URL or base64)
  imageUrl: {
    type: String,
    default: null
  },
  imageName: {
    type: String,
    default: null
  },
  // Metadata
  modelUsed: {
    type: String,
    default: 'ViT-B/16'
  },
  processingTime: {
    type: Number, // in milliseconds
    default: 0
  },
  // For analytics
  detectionDate: {
    type: Date,
    default: Date.now,
    index: true
  },
  // Status
  isReviewed: {
    type: Boolean,
    default: false
  },
  userFeedback: {
    isCorrect: {
      type: Boolean,
      default: null
    },
    actualDisease: {
      type: String,
      default: null
    },
    notes: {
      type: String,
      default: null
    }
  }
}, {
  timestamps: true // Adds createdAt and updatedAt
});

// Index for efficient queries
diseaseHistorySchema.index({ userId: 1, detectionDate: -1 });
diseaseHistorySchema.index({ diseaseName: 1 });

module.exports = mongoose.model('DiseaseHistory', diseaseHistorySchema);

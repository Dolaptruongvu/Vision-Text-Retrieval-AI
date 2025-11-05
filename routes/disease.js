// Disease Routes
const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const {
  saveDiseaseHistory,
  getDiseaseHistory,
  getDiseaseStats,
  getMonthlyDiseaseStats,
  getDiseaseDetail,
  predictDisease
} = require('../controllers/diseaseController');
const { protect } = require('../middleware/auth');

// Configure multer for file upload
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/') // Make sure this folder exists
  },
  filename: function (req, file, cb) {
    // Generate unique filename
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'image-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 16 * 1024 * 1024 // 16MB max
  },
  fileFilter: function (req, file, cb) {
    // Accept images only
    const allowedTypes = /jpeg|jpg|png/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only .png, .jpg and .jpeg format allowed!'));
    }
  }
});

// All disease routes are protected
router.use(protect);

// NEW: Predict disease (Frontend calls this, Backend forwards to Vision Service)
router.post('/predict', upload.single('image'), predictDisease);

router.post('/save', saveDiseaseHistory);
router.get('/history', getDiseaseHistory);
router.get('/stats', getDiseaseStats);
router.get('/monthly-stats', getMonthlyDiseaseStats);
router.get('/:id', getDiseaseDetail);

module.exports = router;

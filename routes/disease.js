// Disease Routes
const express = require('express');
const router = express.Router();
const {
  saveDiseaseHistory,
  getDiseaseHistory,
  getDiseaseStats,
  getDiseaseDetail
} = require('../controllers/diseaseController');
const { protect } = require('../middleware/auth');

// All disease routes are protected
router.use(protect);

router.post('/save', saveDiseaseHistory);
router.get('/history', getDiseaseHistory);
router.get('/stats', getDiseaseStats);
router.get('/:id', getDiseaseDetail);

module.exports = router;

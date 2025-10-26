// Disease Routes
const express = require('express');
const router = express.Router();
const {
  saveDiseaseHistory,
  getDiseaseHistory,
  getDiseaseStats,
  getMonthlyDiseaseStats,
  getDiseaseDetail
} = require('../controllers/diseaseController');
const { protect } = require('../middleware/auth');

// All disease routes are protected
router.use(protect);

router.post('/save', saveDiseaseHistory);
router.get('/history', getDiseaseHistory);
router.get('/stats', getDiseaseStats);
router.get('/monthly-stats', getMonthlyDiseaseStats);
router.get('/:id', getDiseaseDetail);

module.exports = router;

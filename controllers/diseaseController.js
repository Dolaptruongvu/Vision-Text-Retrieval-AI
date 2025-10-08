// Disease Detection Controller
const DiseaseHistory = require('../models/DiseaseHistory');

// @desc    Save disease detection result
// @route   POST /api/disease/save
// @access  Private
exports.saveDiseaseHistory = async (req, res) => {
  try {
    const {
      diseaseName,
      diseaseNameRaw,
      confidence,
      userQuery,
      aiResponse,
      imageUrl,
      imageName,
      processingTime
    } = req.body;

    // Validate required fields
    if (!diseaseName || confidence === undefined) {
      return res.status(400).json({
        success: false,
        message: 'Please provide diseaseName and confidence'
      });
    }

    // Create disease history record
    const diseaseRecord = await DiseaseHistory.create({
      userId: req.user.id,
      diseaseName,
      diseaseNameRaw: diseaseNameRaw || diseaseName,
      confidence,
      userQuery: userQuery || '',
      aiResponse: aiResponse || '',
      imageUrl: imageUrl || null,
      imageName: imageName || null,
      processingTime: processingTime || 0
    });

    res.status(201).json({
      success: true,
      message: 'Disease history saved successfully',
      data: {
        id: diseaseRecord._id,
        diseaseName: diseaseRecord.diseaseName,
        confidence: diseaseRecord.confidence,
        detectionDate: diseaseRecord.detectionDate
      }
    });
  } catch (error) {
    console.error('Save disease history error:', error);
    res.status(500).json({
      success: false,
      message: 'Error saving disease history',
      error: error.message
    });
  }
};

// @desc    Get user's disease history
// @route   GET /api/disease/history
// @access  Private
exports.getDiseaseHistory = async (req, res) => {
  try {
    const { limit = 20, page = 1 } = req.query;

    const skip = (page - 1) * limit;

    const history = await DiseaseHistory.find({ userId: req.user.id })
      .sort({ detectionDate: -1 })
      .limit(parseInt(limit))
      .skip(skip);

    const total = await DiseaseHistory.countDocuments({ userId: req.user.id });

    res.status(200).json({
      success: true,
      data: {
        history,
        pagination: {
          total,
          page: parseInt(page),
          limit: parseInt(limit),
          pages: Math.ceil(total / limit)
        }
      }
    });
  } catch (error) {
    console.error('Get disease history error:', error);
    res.status(500).json({
      success: false,
      message: 'Error getting disease history',
      error: error.message
    });
  }
};

// @desc    Get disease statistics for dashboard
// @route   GET /api/disease/stats
// @access  Private
exports.getDiseaseStats = async (req, res) => {
  try {
    const userId = req.user.id;

    // Total detections
    const totalDetections = await DiseaseHistory.countDocuments({ userId });

    // Disease distribution
    const diseaseDistribution = await DiseaseHistory.aggregate([
      { $match: { userId: userId } },
      {
        $group: {
          _id: '$diseaseName',
          count: { $sum: 1 },
          avgConfidence: { $avg: '$confidence' }
        }
      },
      { $sort: { count: -1 } },
      { $limit: 10 }
    ]);

    // Recent detections (last 7 days)
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

    const recentDetections = await DiseaseHistory.countDocuments({
      userId,
      detectionDate: { $gte: sevenDaysAgo }
    });

    // Detection trend (last 30 days)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const detectionTrend = await DiseaseHistory.aggregate([
      {
        $match: {
          userId: userId,
          detectionDate: { $gte: thirtyDaysAgo }
        }
      },
      {
        $group: {
          _id: {
            $dateToString: { format: '%Y-%m-%d', date: '$detectionDate' }
          },
          count: { $sum: 1 }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    res.status(200).json({
      success: true,
      data: {
        totalDetections,
        recentDetections,
        diseaseDistribution,
        detectionTrend
      }
    });
  } catch (error) {
    console.error('Get disease stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Error getting disease statistics',
      error: error.message
    });
  }
};

// @desc    Get single disease record detail
// @route   GET /api/disease/:id
// @access  Private
exports.getDiseaseDetail = async (req, res) => {
  try {
    const disease = await DiseaseHistory.findOne({
      _id: req.params.id,
      userId: req.user.id
    });

    if (!disease) {
      return res.status(404).json({
        success: false,
        message: 'Disease record not found'
      });
    }

    res.status(200).json({
      success: true,
      data: { disease }
    });
  } catch (error) {
    console.error('Get disease detail error:', error);
    res.status(500).json({
      success: false,
      message: 'Error getting disease detail',
      error: error.message
    });
  }
};

// Disease Detection Controller
const DiseaseHistory = require('../models/DiseaseHistory');
const mongoose = require('mongoose');

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

// @desc    Get monthly disease statistics (percentage distribution)
// @route   GET /api/disease/monthly-stats
// @access  Private
exports.getMonthlyDiseaseStats = async (req, res) => {
  try {
    const userId = req.user.id;
    const { year, month } = req.query;

    // Default to current month if not specified
    const now = new Date();
    const targetYear = year ? parseInt(year) : now.getFullYear();
    const targetMonth = month ? parseInt(month) : now.getMonth() + 1; // JS months are 0-indexed

    // Calculate start and end of month IN UTC (to match MongoDB storage)
    const startOfMonth = new Date(Date.UTC(targetYear, targetMonth - 1, 1, 0, 0, 0, 0));
    const endOfMonth = new Date(Date.UTC(targetYear, targetMonth, 0, 23, 59, 59, 999));

    console.log(`[DISEASE STATS] Calculating for ${targetYear}-${targetMonth}`);
    console.log(`[DISEASE STATS] Date range (UTC): ${startOfMonth.toISOString()} to ${endOfMonth.toISOString()}`);
    console.log(`[DISEASE STATS] UserId:`, userId);

    // Get disease distribution for the month
    const diseaseStats = await DiseaseHistory.aggregate([
      {
        $match: {
          userId: new mongoose.Types.ObjectId(userId), // Convert string to ObjectId
          detectionDate: {
            $gte: startOfMonth,
            $lte: endOfMonth
          }
        }
      },
      {
        $group: {
          _id: '$diseaseName',
          count: { $sum: 1 },
          avgConfidence: { $avg: '$confidence' },
          diseaseNameRaw: { $first: '$diseaseNameRaw' }
        }
      },
      {
        $sort: { count: -1 }
      }
    ]);

    // Calculate total detections for percentage
    const totalDetections = diseaseStats.reduce((sum, item) => sum + item.count, 0);

    // Format results with percentage
    const diseaseDistribution = diseaseStats.map(item => ({
      diseaseName: item._id,
      diseaseNameRaw: item.diseaseNameRaw || item._id,
      count: item.count,
      percentage: totalDetections > 0 ? ((item.count / totalDetections) * 100).toFixed(2) : 0,
      avgConfidence: item.avgConfidence ? (item.avgConfidence * 100).toFixed(2) : 0
    }));

    // Get previous month stats for comparison (also in UTC)
    const prevMonthStart = new Date(Date.UTC(targetYear, targetMonth - 2, 1, 0, 0, 0, 0));
    const prevMonthEnd = new Date(Date.UTC(targetYear, targetMonth - 1, 0, 23, 59, 59, 999));

    const prevMonthTotal = await DiseaseHistory.countDocuments({
      userId: new mongoose.Types.ObjectId(userId), // Convert string to ObjectId
      detectionDate: {
        $gte: prevMonthStart,
        $lte: prevMonthEnd
      }
    });

    // Calculate trend
    const trend = prevMonthTotal > 0 
      ? (((totalDetections - prevMonthTotal) / prevMonthTotal) * 100).toFixed(2)
      : totalDetections > 0 ? 100 : 0;

    res.status(200).json({
      success: true,
      data: {
        period: {
          year: targetYear,
          month: targetMonth,
          monthName: new Date(targetYear, targetMonth - 1).toLocaleString('vi-VN', { month: 'long' }),
          startDate: startOfMonth,
          endDate: endOfMonth
        },
        summary: {
          totalDetections,
          previousMonthTotal: prevMonthTotal,
          trend: `${trend > 0 ? '+' : ''}${trend}%`,
          trendValue: parseFloat(trend)
        },
        diseaseDistribution,
        topDiseases: diseaseDistribution.slice(0, 5),
        message: totalDetections === 0 
          ? `Không có dữ liệu phát hiện bệnh trong tháng ${targetMonth}/${targetYear}`
          : `Tháng ${targetMonth}/${targetYear}: ${diseaseDistribution[0]?.diseaseName} chiếm ${diseaseDistribution[0]?.percentage}% (${diseaseDistribution[0]?.count}/${totalDetections} trường hợp)`
      }
    });

  } catch (error) {
    console.error('Get monthly disease stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Error getting monthly disease statistics',
      error: error.message
    });
  }
};

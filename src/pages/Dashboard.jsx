import { useState, useEffect } from 'react';
import { diseaseAPI } from '../services/api';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { BarChart3, TrendingUp, TrendingDown, Calendar, Loader2 } from 'lucide-react';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316'];

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [selectedMonth, setSelectedMonth] = useState(new Date().getMonth() + 1);

  useEffect(() => {
    fetchStats();
  }, [selectedYear, selectedMonth]);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await diseaseAPI.getMonthlyStats(selectedYear, selectedMonth);
      
      console.log('[DASHBOARD] API Response:', response); // Debug log
      console.log('[DASHBOARD] Stats data:', response?.data); // Debug nested data
      
      if (response.success && response.data) {
        // Backend trả về: { success: true, data: { period, summary, diseaseDistribution, ... } }
        // response.data = { period, summary, diseaseDistribution, topDiseases, message }
        setStats(response.data);
        setError(null);
      } else {
        setError(response.message || 'Không thể tải dữ liệu');
        setStats(null);
      }
    } catch (err) {
      console.error('[DASHBOARD] Error:', err); // Debug log
      setError(err.response?.data?.message || 'Lỗi kết nối server');
    } finally {
      setLoading(false);
    }
  };

  const years = Array.from({ length: 5 }, (_, i) => new Date().getFullYear() - i);
  const months = [
    { value: 1, label: 'Tháng 1' },
    { value: 2, label: 'Tháng 2' },
    { value: 3, label: 'Tháng 3' },
    { value: 4, label: 'Tháng 4' },
    { value: 5, label: 'Tháng 5' },
    { value: 6, label: 'Tháng 6' },
    { value: 7, label: 'Tháng 7' },
    { value: 8, label: 'Tháng 8' },
    { value: 9, label: 'Tháng 9' },
    { value: 10, label: 'Tháng 10' },
    { value: 11, label: 'Tháng 11' },
    { value: 12, label: 'Tháng 12' },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary-600 mx-auto mb-4" />
          <p className="text-gray-600">Đang tải dữ liệu...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-red-600">
          <p className="text-lg font-medium">❌ {error}</p>
          <button onClick={fetchStats} className="btn btn-primary mt-4">
            Thử lại
          </button>
        </div>
      </div>
    );
  }

  const chartData = stats?.diseaseDistribution?.map((disease) => ({
    name: disease.diseaseName,
    value: parseFloat(disease.percentage),
    count: disease.count,
  })) || [];

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-primary-600" />
            Thống Kê Bệnh Cây Trồng
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Phân tích xu hướng bệnh theo tháng
          </p>
        </div>

        {/* Period Selector */}
        <div className="flex items-center gap-3">
          <Calendar className="w-5 h-5 text-gray-500" />
          <select
            value={selectedMonth}
            onChange={(e) => setSelectedMonth(parseInt(e.target.value))}
            className="input"
          >
            {months.map((month) => (
              <option key={month.value} value={month.value}>
                {month.label}
              </option>
            ))}
          </select>
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(parseInt(e.target.value))}
            className="input"
          >
            {years.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Total Detections */}
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Tổng phát hiện</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">
                {stats?.summary?.totalDetections || 0}
              </p>
            </div>
            <div className="bg-primary-100 p-3 rounded-full">
              <BarChart3 className="w-6 h-6 text-primary-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {stats?.period?.monthName} {stats?.period?.year}
          </p>
        </div>

        {/* Trend */}
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Xu hướng</p>
              <p className={`text-3xl font-bold mt-1 ${
                stats?.summary?.trendValue >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {stats?.summary?.trend || 'N/A'}
              </p>
            </div>
            <div className={`p-3 rounded-full ${
              stats?.summary?.trendValue >= 0 ? 'bg-green-100' : 'bg-red-100'
            }`}>
              {stats?.summary?.trendValue >= 0 ? (
                <TrendingUp className="w-6 h-6 text-green-600" />
              ) : (
                <TrendingDown className="w-6 h-6 text-red-600" />
              )}
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            So với tháng trước: {stats?.summary?.previousMonthTotal || 0} ca
          </p>
        </div>

        {/* Top Disease */}
        <div className="card">
          <div>
            <p className="text-sm text-gray-600">Bệnh phổ biến nhất</p>
            <p className="text-lg font-bold text-gray-900 mt-1 line-clamp-2">
              {stats?.topDiseases?.[0]?.diseaseName || 'Chưa có dữ liệu'}
            </p>
          </div>
          {stats?.topDiseases?.[0] && (
            <div className="mt-3 flex items-center gap-2">
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-primary-600 h-2 rounded-full"
                  style={{ width: `${stats.topDiseases[0].percentage}%` }}
                />
              </div>
              <span className="text-sm font-medium text-gray-700">
                {stats.topDiseases[0].percentage}%
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Charts & Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Phân bố bệnh (%)
          </h2>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              Không có dữ liệu
            </div>
          )}
        </div>

        {/* Top 5 Diseases Table */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Top 5 Bệnh Phổ Biến
          </h2>
          {stats?.topDiseases?.length > 0 ? (
            <div className="space-y-3">
              {stats.topDiseases.map((disease, index) => (
                <div
                  key={disease.diseaseName}
                  className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg"
                >
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-white`}
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  >
                    {index + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 truncate">
                      {disease.diseaseName}
                    </p>
                    <p className="text-sm text-gray-600">
                      {disease.count} ca · Độ tin cậy: {disease.avgConfidence}%
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-primary-600">
                      {disease.percentage}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              Không có dữ liệu
            </div>
          )}
        </div>
      </div>

      {/* Message */}
      {stats?.message && (
        <div className="card bg-primary-50 border-primary-200">
          <p className="text-sm text-primary-900">
            💡 <strong>Gợi ý:</strong> {stats.message}
          </p>
        </div>
      )}
    </div>
  );
}

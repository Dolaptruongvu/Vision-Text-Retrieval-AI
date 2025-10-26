# ThesisFE - Plant Disease Detection Frontend

## 🎨 **UI Design - Giống ChatGPT/Gemini**

React app với Vite + Tailwind CSS, giao diện thân thiện, đơn giản và sạch.

## 📁 **Project Structure**

```
src/
├── services/
│   └── api.js ✅ (Already created - API service layer)
├── context/
│   └── AuthContext.jsx (Auth provider với localStorage)
├── pages/
│   ├── Login.jsx (Login form)
│   ├── Register.jsx (Register form)
│   ├── Chat.jsx (ChatGPT-style chat interface)
│   └── Dashboard.jsx (Monthly statistics with charts)
├── components/
│   ├── Layout.jsx (Sidebar + main content)
│   ├── Sidebar.jsx (Navigation: Chat, Dashboard, Logout)
│   ├── ChatMessage.jsx (Message bubble component)
│   ├── ChatInput.jsx (Input box with image upload)
│   └── ProtectedRoute.jsx (Auth guard)
├── App.jsx (Router setup)
└── main.jsx (Entry point)
```

## 🚀 **Next Steps - Tạo các file components**

### **1. AuthContext.jsx**
```jsx
import { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../services/api';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (userData) {
      setUser(JSON.parse(userData));
    }
    setLoading(false);
  }, []);

  const login = async (credentials) => {
    const response = await authAPI.login(credentials);
    setUser(response.data.user);
    return response;
  };

  const register = async (userData) => {
    return await authAPI.register(userData);
  };

  const logout = () => {
    authAPI.logout();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
```

### **2. App.jsx - Router Setup**
```jsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Login from './pages/Login';
import Register from './pages/Register';
import Chat from './pages/Chat';
import Dashboard from './pages/Dashboard';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route index element={<Navigate to="/chat" replace />} />
            <Route path="chat" element={<Chat />} />
            <Route path="dashboard" element={<Dashboard />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
```

### **3. Login.jsx - Login Page**
```jsx
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Leaf } from 'lucide-react';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login({ email, password });
      navigate('/chat');
    } catch (err) {
      setError(err.response?.data?.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-50 to-blue-50">
      <div className="card w-full max-w-md">
        <div className="flex items-center justify-center mb-8">
          <Leaf className="w-12 h-12 text-green-600 mr-2" />
          <h1 className="text-3xl font-bold text-gray-900">Plant AI</h1>
        </div>

        <h2 className="text-2xl font-semibold mb-6 text-center">Đăng nhập</h2>

        {error && (
          <div className="bg-red-50 text-red-600 p-3 rounded-lg mb-4">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="input"
              placeholder="your@email.com"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Mật khẩu</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="input"
              placeholder="••••••••"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary w-full"
          >
            {loading ? 'Đang đăng nhập...' : 'Đăng nhập'}
          </button>
        </form>

        <p className="text-center mt-6 text-gray-600">
          Chưa có tài khoản?{' '}
          <Link to="/register" className="text-primary-600 hover:underline">
            Đăng ký ngay
          </Link>
        </p>
      </div>
    </div>
  );
}

export default Login;
```

### **4. Chat.jsx - Main Chat Interface (ChatGPT-style)**
```jsx
import { useState, useRef, useEffect } from 'react';
import { Send, Image, Loader } from 'lucide-react';
import { chatAPI } from '../services/api';
import ChatMessage from '../components/ChatMessage';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !image) return;

    // Add user message
    const userMessage = {
      role: 'user',
      content: input,
      image: imagePreview,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Prepare form data
    const formData = new FormData();
    if (input.trim()) formData.append('query', input);
    if (image) formData.append('image', image);

    // Clear input
    setInput('');
    setImage(null);
    setImagePreview(null);
    setLoading(true);

    try {
      const response = await chatAPI.predict(formData);

      // Add AI response
      const aiMessage = {
        role: 'assistant',
        content: response.ai_response,
        visionResult: response.vision_result,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: 'Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.',
        error: true,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-20">
            <Leaf className="w-16 h-16 mx-auto mb-4 text-green-500" />
            <p className="text-xl font-medium">Xin chào! Tôi là AI trợ lý nông nghiệp</p>
            <p className="mt-2">Upload ảnh cây trồng hoặc đặt câu hỏi về bệnh cây</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <ChatMessage key={idx} message={msg} />
        ))}

        {loading && (
          <div className="flex items-center justify-center p-4">
            <Loader className="w-6 h-6 animate-spin text-primary-600" />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        {imagePreview && (
          <div className="mb-3 relative inline-block">
            <img src={imagePreview} alt="Preview" className="h-20 rounded-lg" />
            <button
              onClick={() => {
                setImage(null);
                setImagePreview(null);
              }}
              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6"
            >
              ×
            </button>
          </div>
        )}

        <div className="flex gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            className="btn btn-secondary"
            disabled={loading}
          >
            <Image className="w-5 h-5" />
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Đặt câu hỏi về bệnh cây trồng..."
            className="input flex-1"
            disabled={loading}
          />

          <button
            onClick={handleSend}
            disabled={loading || (!input.trim() && !image)}
            className="btn btn-primary"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

export default Chat;
```

### **5. Dashboard.jsx - Monthly Statistics**
```jsx
import { useEffect, useState } from 'react';
import { diseaseAPI } from '../services/api';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { TrendingUp, Calendar } from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [year, setYear] = useState(2025);
  const [month, setMonth] = useState(10);

  useEffect(() => {
    fetchStats();
  }, [year, month]);

  const fetchStats = async () => {
    setLoading(true);
    try {
      const response = await diseaseAPI.getMonthlyStats(year, month);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="p-6">Loading...</div>;
  if (!stats) return <div className="p-6">No data</div>;

  const chartData = stats.diseaseDistribution.map((item) => ({
    name: item.diseaseName,
    value: parseFloat(item.percentage),
  }));

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">Dashboard - Thống kê bệnh</h1>

      {/* Period Selector */}
      <div className="card flex gap-4 items-center">
        <Calendar className="w-5 h-5" />
        <select
          value={year}
          onChange={(e) => setYear(e.target.value)}
          className="input w-32"
        >
          <option value={2025}>2025</option>
          <option value={2024}>2024</option>
        </select>
        <select
          value={month}
          onChange={(e) => setMonth(e.target.value)}
          className="input w-32"
        >
          {Array.from({ length: 12 }, (_, i) => (
            <option key={i + 1} value={i + 1}>
              Tháng {i + 1}
            </option>
          ))}
        </select>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="text-gray-600 mb-2">Tổng phát hiện</h3>
          <p className="text-3xl font-bold">{stats.summary.totalDetections}</p>
        </div>

        <div className="card">
          <h3 className="text-gray-600 mb-2">Xu hướng</h3>
          <div className="flex items-center gap-2">
            <TrendingUp className={stats.summary.trendValue > 0 ? 'text-green-600' : 'text-red-600'} />
            <p className="text-3xl font-bold">{stats.summary.trend}</p>
          </div>
        </div>

        <div className="card">
          <h3 className="text-gray-600 mb-2">Bệnh phổ biến nhất</h3>
          <p className="text-lg font-semibold">
            {stats.topDiseases[0]?.diseaseName || 'N/A'}
          </p>
          <p className="text-gray-600">{stats.topDiseases[0]?.percentage}%</p>
        </div>
      </div>

      {/* Pie Chart */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Phân bố bệnh theo %</h2>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={chartData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={120}
                label
              >
                {chartData.map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500 text-center py-20">Không có dữ liệu</p>
        )}
      </div>

      {/* Top Diseases Table */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Top 5 Bệnh</h2>
        <table className="w-full">
          <thead className="border-b">
            <tr>
              <th className="text-left py-2">Bệnh</th>
              <th className="text-right py-2">Số lượng</th>
              <th className="text-right py-2">Tỷ lệ %</th>
              <th className="text-right py-2">Độ tin cậy TB</th>
            </tr>
          </thead>
          <tbody>
            {stats.topDiseases.map((disease, idx) => (
              <tr key={idx} className="border-b last:border-0">
                <td className="py-3">{disease.diseaseName}</td>
                <td className="text-right">{disease.count}</td>
                <td className="text-right">{disease.percentage}%</td>
                <td className="text-right">{disease.avgConfidence}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Dashboard;
```

## 🎯 **Để chạy app:**

```bash
cd c:\wd\ThesisLLM\ThesisApp\ThesisFE
npm run dev
```

App sẽ chạy tại: http://localhost:5173

## ✅ **Checklist:**
- [x] API service với auth interceptor
- [ ] AuthContext for global auth state
- [ ] Login/Register pages
- [ ] Chat interface (ChatGPT-style)
- [ ] Dashboard with charts
- [ ] Layout với Sidebar
- [ ] Protected routes

Bạn cần tôi tạo tiếp file nào không? 🚀

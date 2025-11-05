import { useState, useRef, useEffect } from 'react';
import { chatAPI, sessionAPI } from '../services/api';
import { Send, Image as ImageIcon, X, Loader2, MessageSquare, Plus, Trash2 } from 'lucide-react';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Chat session management
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  // Load existing sessions
  const loadSessions = async () => {
    try {
      const response = await sessionAPI.getSessions();
      if (response.success) {
        setSessions(response.data);
        // Auto-load most recent session or create new one
        if (response.data.length > 0) {
          loadSession(response.data[0].id);
        } else {
          createNewSession();
        }
      }
    } catch (error) {
      console.error('Load sessions error:', error);
    }
  };

  // Load specific session
  const loadSession = async (sessionId) => {
    try {
      const response = await sessionAPI.getSession(sessionId);
      if (response.success && response.data) {
        setCurrentSessionId(sessionId);
        // Convert messages to frontend format (safe guard for empty array)
        const messages = response.data.messages || [];
        const formattedMessages = messages.map((msg, index) => ({
          id: index,
          type: msg.role === 'user' ? 'user' : 'ai',
          text: msg.content,
          timestamp: new Date(msg.timestamp) // ← Parse string to Date object
        }));
        setMessages(formattedMessages);
      }
    } catch (error) {
      console.error('Load session error:', error);
      // If load fails, create new session
      createNewSession();
    }
  };

  // Create new session
  const createNewSession = async () => {
    try {
      const response = await sessionAPI.createSession();
      if (response.success) {
        setCurrentSessionId(response.data.id);
        setMessages([]);
        setInput('');
        setImage(null);
        setImagePreview(null);
        loadSessions(); // Refresh session list
      }
    } catch (error) {
      console.error('Create session error:', error);
    }
  };

  // Delete session
  const deleteSession = async (sessionId) => {
    if (!confirm('Xóa cuộc trò chuyện này?')) return;
    
    try {
      await sessionAPI.deleteSession(sessionId);
      loadSessions();
      if (sessionId === currentSessionId) {
        createNewSession();
      }
    } catch (error) {
      console.error('Delete session error:', error);
    }
  };

  // Save message to current session
  const saveMessage = async (role, content) => {
    if (!currentSessionId) return;
    
    try {
      await sessionAPI.addMessage(currentSessionId, { role, content });
      loadSessions(); // Refresh to update lastMessageAt
    } catch (error) {
      console.error('Save message error:', error);
    }
  };

  // Auto scroll to bottom when new message arrives
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !image) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: input,
      image: imagePreview,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    
    // Save user message to session
    await saveMessage('user', input || 'Uploaded image');
    
    const currentInput = input;
    setInput('');
    setLoading(true);

    try {
      const formData = new FormData();
      if (image) {
        formData.append('image', image);
      }
      if (currentInput.trim()) {
        formData.append('query', input);
      }

      const response = await chatAPI.predict(formData);

      if (response.success) {
        const aiMessage = {
          id: Date.now() + 1,
          type: 'ai',
          text: response.data.ai_response,
          visionResult: response.data.vision_result,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, aiMessage]);
        
        // Save AI response to session
        await saveMessage('assistant', response.data.ai_response);
      } else {
        const errorMessage = {
          id: Date.now() + 1,
          type: 'error',
          text: response.error || 'Đã xảy ra lỗi khi xử lý yêu cầu',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        text: error.response?.data?.message || 'Không thể kết nối đến server',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      handleRemoveImage();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-full">
      {/* Session Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* New Chat Button */}
        <div className="p-4 border-b">
          <button
            onClick={createNewSession}
            className="btn btn-primary w-full flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>

        {/* Session List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => loadSession(session.id)}
              className={`p-3 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors group ${
                session.id === currentSessionId ? 'bg-primary-50 border border-primary-200' : ''
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium truncate ${
                    session.id === currentSessionId ? 'text-primary-700' : 'text-gray-900'
                  }`}>
                    {session.title}
                  </p>
                  <p className="text-xs text-gray-500 mt-1 truncate">
                    {session.preview}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    {new Date(session.lastMessageAt).toLocaleString('vi-VN', {
                      day: '2-digit',
                      month: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(session.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded transition-opacity"
                  title="Xóa"
                >
                  <Trash2 className="w-4 h-4 text-red-600" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-primary-600" />
            Chẩn Đoán Bệnh Cây Trồng
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Upload ảnh và đặt câu hỏi về bệnh cây trồng
          </p>
        </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <MessageSquare className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <p className="text-lg font-medium">Chưa có tin nhắn</p>
              <p className="text-sm mt-2">Upload ảnh hoặc đặt câu hỏi để bắt đầu</p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl rounded-lg px-4 py-3 ${
                  message.type === 'user'
                    ? 'bg-primary-600 text-white'
                    : message.type === 'error'
                    ? 'bg-red-50 text-red-800 border border-red-200'
                    : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                }`}
              >
                {/* User message with image */}
                {message.type === 'user' && message.image && (
                  <img
                    src={message.image}
                    alt="Uploaded"
                    className="w-48 h-48 object-cover rounded-lg mb-2"
                  />
                )}
                
                {/* Message text */}
                {message.text && (
                  <p className="whitespace-pre-wrap">{message.text}</p>
                )}

                {/* AI response with vision result */}
                {message.type === 'ai' && message.visionResult && (
                  <div className="bg-primary-50 border border-primary-200 rounded-lg p-3 mb-3">
                    <p className="text-sm font-medium text-primary-900 mb-1">
                      🔍 Kết quả phân tích:
                    </p>
                    <p className="text-primary-800">
                      <span className="font-semibold">{message.visionResult.disease}</span>
                      {' '}
                      ({(message.visionResult.confidence * 100).toFixed(1)}% tin cậy)
                    </p>
                  </div>
                )}

                {/* Timestamp */}
                <p className={`text-xs mt-2 ${
                  message.type === 'user' ? 'text-primary-100' : 'text-gray-500'
                }`}>
                  {message.timestamp && typeof message.timestamp.toLocaleTimeString === 'function' 
                    ? message.timestamp.toLocaleTimeString('vi-VN', { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })
                    : ''
                  }
                </p>
              </div>
            </div>
          ))
        )}

        {/* Loading indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-lg px-4 py-3 shadow-sm border border-gray-200">
              <div className="flex items-center gap-2 text-gray-600">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Đang phân tích...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 p-4">
        {/* Image Preview */}
        {imagePreview && (
          <div className="mb-3 relative inline-block">
            <img
              src={imagePreview}
              alt="Preview"
              className="w-32 h-32 object-cover rounded-lg border-2 border-primary-300"
            />
            <button
              onClick={handleRemoveImage}
              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Input Box */}
        <div className="flex items-end gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImageSelect}
            accept="image/*"
            className="hidden"
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="btn btn-secondary p-3"
            title="Upload ảnh"
          >
            <ImageIcon className="w-5 h-5" />
          </button>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Nhập câu hỏi của bạn... (Enter để gửi, Shift+Enter để xuống dòng)"
            disabled={loading}
            className="input flex-1 resize-none"
            rows={1}
            style={{
              minHeight: '44px',
              maxHeight: '120px',
            }}
          />

          <button
            onClick={handleSend}
            disabled={loading || (!input.trim() && !image)}
            className="btn btn-primary p-3"
            title="Gửi tin nhắn"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>

        <p className="text-xs text-gray-500 mt-2 text-center">
          AI có thể mắc lỗi. Hãy kiểm tra thông tin quan trọng.
        </p>
      </div>
    </div>
    </div>
  );
}

const API_BASE = window.__API_BASE__ || 'http://127.0.0.1:8000';
let currentSessionId = null; // 全局会话ID
let previousSessionId = null; // 用于检测会话切换
const auth = {
  get token() { return localStorage.getItem('fauna_token') || sessionStorage.getItem('fauna_token'); },
  get user() { try { return JSON.parse(localStorage.getItem('fauna_user') || sessionStorage.getItem('fauna_user') || 'null'); } catch { return null; } },
  save(data, remember = true) {
    const store = remember ? localStorage : sessionStorage;
    store.setItem('fauna_token', data.token);
    store.setItem('fauna_user', JSON.stringify(data.user));
  },
  clear() {
    localStorage.removeItem('fauna_token'); localStorage.removeItem('fauna_user');
    sessionStorage.removeItem('fauna_token'); sessionStorage.removeItem('fauna_user');
  }
};

async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || '请求失败');
  return data;
}

function showError(id, msg) { const el = document.getElementById(id); if (el) { el.textContent = msg; el.style.display = msg ? 'block' : 'none'; } }
function showSuccess(id, msg) { const el = document.getElementById(id); if (el) { el.textContent = msg; el.style.display = msg ? 'block' : 'none'; } }

function initLoginPage() {
  const loginBox = document.getElementById('login-box');
  const registerBox = document.getElementById('register-box');
  if (!loginBox || !registerBox) return;
  if (auth.token) location.href = 'index.html';

  document.getElementById('show-register')?.addEventListener('click', e => { e.preventDefault(); loginBox.style.display = 'none'; registerBox.style.display = 'block'; });
  document.getElementById('show-login')?.addEventListener('click', e => { e.preventDefault(); registerBox.style.display = 'none'; loginBox.style.display = 'block'; });

  document.getElementById('login-form')?.addEventListener('submit', async e => {
    e.preventDefault(); showError('login-error', '');
    try {
      const data = await api('/auth/login', { method: 'POST', body: JSON.stringify({ username: document.getElementById('login-username').value, password: document.getElementById('login-password').value }) });
      auth.save(data, document.getElementById('remember-me').checked);
      // 根据用户角色跳转到不同页面
      if (data.user?.is_admin) {
        location.href = 'admin.html';
      } else {
        location.href = 'index.html';
      }
    } catch (err) { showError('login-error', err.message); }
  });

  document.getElementById('register-form')?.addEventListener('submit', async e => {
    e.preventDefault(); showError('register-error', ''); showSuccess('register-success', '');
    const password = document.getElementById('register-password').value;
    if (password !== document.getElementById('register-confirm-password').value) return showError('register-error', '两次输入的密码不一致');
    if (!document.getElementById('agree-terms').checked) return showError('register-error', '请先同意用户协议和隐私政策');
    try {
      const data = await api('/auth/register', { method: 'POST', body: JSON.stringify({ username: document.getElementById('register-username').value, email: document.getElementById('register-email').value, password }) });
      auth.save(data, true); showSuccess('register-success', '注册成功，正在进入主页面...');
      setTimeout(() => location.href = 'index.html', 600);
    } catch (err) { showError('register-error', err.message); }
  });
}

function appendMessage(role, content, meta = '') {
  const list = document.getElementById('messages-container'); if (!list) return null;
  const item = document.createElement('div'); item.className = `message ${role}`;
  item.innerHTML = `<div class="bubble"><div class="content"></div>${meta ? `<div class="meta">${meta}</div>` : ''}</div>`;
  item.querySelector('.content').textContent = content;
  list.appendChild(item); list.scrollTop = list.scrollHeight;
  return item.querySelector('.content');
}

async function loadHistory() {
  const container = document.getElementById('history-container');
  if (!container) return;
  
  try {
    const data = await api('/history?limit=80');
    
    if (!data.history || data.history.length === 0) {
      container.innerHTML = '<div class="empty-history">暂无历史记录</div>';
      return;
    }
    
    // 按会话分组
    const sessions = {};
    data.history.forEach(msg => {
      if (!sessions[msg.session_id]) {
        sessions[msg.session_id] = [];
      }
      sessions[msg.session_id].push(msg);
    });
    
    let html = '';
    
    // 显示每个会话的第一条用户消息
    for (const [sessionId, messages] of Object.entries(sessions)) {
      const userMessages = messages.filter(m => m.role === 'user');
      if (userMessages.length === 0) continue;
      
      // 使用第一条用户消息作为会话标题
      const firstUserMsg = userMessages[0];
      const lastMsg = messages[messages.length - 1];
      
      html += `<div class="history-item" data-session="${sessionId}">
        <div class="history-icon"></div>
        <div class="history-content">
          <div class="history-title">${escapeHtml(firstUserMsg.content.slice(0, 50))}${firstUserMsg.content.length > 50 ? '...' : ''}</div>
          <div class="history-preview">${lastMsg.role === 'assistant' ? escapeHtml(lastMsg.content.slice(0, 30)) + (lastMsg.content.length > 30 ? '...' : '') : '等待回复...'}</div>
        </div>
        <div class="history-time">${formatTime(firstUserMsg.created_at)}</div>
        <button class="history-delete-btn" data-session="${sessionId}" title="删除此对话">✕</button>
      </div>`;
    }
    
    container.innerHTML = html || '<div class="empty-history">暂无历史记录</div>';
  } catch (err) {
    container.innerHTML = `<div class="empty-history">历史加载失败：${err.message}</div>`;
  }
}

// 在 initMainPage 中使用事件委托处理历史记录点击

async function loadSessionMessages(sessionId) {
  const messagesContainer = document.getElementById('messages-container');
  if (!messagesContainer) return;
  
  // 清空聊天区域并显示选中会话的消息（只读查看）
  messagesContainer.innerHTML = '';
  
  try {
    const data = await api(`/history?session_id=${sessionId}`);
    if (!data.history || data.history.length === 0) return;
    
    // 显示会话消息
    data.history.forEach(msg => {
      appendMessage(msg.role, msg.content);
    });
    
    // 隐藏欢迎界面
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.style.display = 'none';
    
    // 设置 currentSessionId，这样用户发送新消息会追加到当前查看的会话中
    currentSessionId = sessionId;
    previousSessionId = sessionId;
  } catch (err) {
    console.error('加载会话失败:', err);
  }
}

function groupByDate(messages) {
  const groups = {};
  messages.forEach(msg => {
    const date = formatDate(msg.created_at);
    if (!groups[date]) {
      groups[date] = [];
    }
    groups[date].push(msg);
  });
  return groups;
}

function formatDate(dateStr) {
  if (!dateStr) return '未知';
  const date = new Date(dateStr);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  if (date.toDateString() === today.toDateString()) {
    return '今天';
  } else if (date.toDateString() === yesterday.toDateString()) {
    return '昨天';
  } else {
    return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
  }
}

function formatTime(dateStr) {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function getPreview(messages, userMsg) {
  const idx = messages.findIndex(m => m.id === userMsg.id || (m.created_at === userMsg.created_at && m.role === 'user'));
  if (idx < messages.length - 1 && messages[idx + 1].role === 'assistant') {
    return messages[idx + 1].content.slice(0, 30) + (messages[idx + 1].content.length > 30 ? '...' : '');
  }
  return '等待回复...';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

async function initMainPage() {
  const app = document.getElementById('app-container'); if (!app) return;
  if (!auth.token) return location.href = 'login.html';
  const user = auth.user;
  
  // 更新用户信息显示
  const userNameEl = document.getElementById('current-user-name');
  if (userNameEl) userNameEl.textContent = user?.username || '用户';
  
  const userRoleEl = document.getElementById('current-user-role');
  if (userRoleEl) userRoleEl.textContent = user?.is_admin ? '管理员' : '普通用户';
  
  // 显示/隐藏管理员面板按钮
  const adminBtn = document.getElementById('admin-panel-btn');
  if (adminBtn) adminBtn.style.display = user?.is_admin ? 'flex' : 'none';
  
  // 登出按钮
  document.getElementById('btn-logout')?.addEventListener('click', async () => { 
    try { await api('/auth/logout', { method: 'POST' }); } catch {} 
    auth.clear(); 
    location.href = 'login.html'; 
  });
  
  document.getElementById('refresh-history')?.addEventListener('click', loadHistory);

  // 管理员功能
  if (user?.is_admin) {
    initAdminPanel();
    try { 
      const s = await api('/admin/summary'); 
      const adminSummaryEl = document.getElementById('admin-summary');
      if (adminSummaryEl) adminSummaryEl.textContent = `用户 ${s.users}｜在线会话 ${s.active_sessions}｜消息 ${s.messages}`; 
    } catch {}
  }
  
  await loadHistory();

  // 使用事件委托处理历史记录点击和删除
  document.getElementById('history-container')?.addEventListener('click', async (e) => {
    // 处理删除按钮点击
    const deleteBtn = e.target.closest('.history-delete-btn');
    if (deleteBtn) {
      e.stopPropagation();
      const sessionId = deleteBtn.dataset.session;
      if (sessionId && confirm('确定要删除这个对话吗？')) {
        try {
          await api(`/history/${sessionId}`, { method: 'DELETE' });
          // 如果删除的是当前查看的会话，清空聊天区域
          if (sessionId === currentSessionId) {
            currentSessionId = null;
            const messagesContainer = document.getElementById('messages-container');
            if (messagesContainer) {
              messagesContainer.innerHTML = `
                <div class="welcome-message">
                  <div class="welcome-icon"></div>
                  <h2>欢迎来到 Fauna AI</h2>
                  <p>我是您的智能宠物助手，请问有什么可以帮助您的？</p>
                  <div class="welcome-suggestions">
                    <button class="suggestion-btn">狗狗训练指南</button>
                    <button class="suggestion-btn">猫咪品种介绍</button>
                    <button class="suggestion-btn">宠物健康咨询</button>
                  </div>
                </div>
              `;
            }
          }
          await loadHistory();
        } catch (err) {
          alert('删除失败：' + err.message);
        }
      }
      return;
    }
    
    // 处理历史记录项点击
    const item = e.target.closest('.history-item');
    if (item) {
      const sessionId = item.dataset.session;
      if (sessionId) loadSessionMessages(sessionId);
    }
  });

  // 新对话按钮
  document.getElementById('new-chat-btn')?.addEventListener('click', () => {
    currentSessionId = null;
    const messagesContainer = document.getElementById('messages-container');
    if (messagesContainer) {
      messagesContainer.innerHTML = `
        <div class="welcome-message">
          <div class="welcome-icon"></div>
          <h2>欢迎来到 Fauna AI</h2>
          <p>我是您的智能宠物助手，请问有什么可以帮助您的？</p>
          <div class="welcome-suggestions">
            <button class="suggestion-btn">狗狗训练指南</button>
            <button class="suggestion-btn">猫咪品种介绍</button>
            <button class="suggestion-btn">宠物健康咨询</button>
          </div>
        </div>
      `;
    }
  });

  // 发送消息函数
  const sendMessage = async () => {
    const input = document.getElementById('input-field');
    if (!input) return;
    const message = input.value.trim();
    if (!message) return;
    
    // 隐藏欢迎界面
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.style.display = 'none';
    
    // 如果是新会话（currentSessionId 为 null）或会话切换，清空聊天区域
    if (!currentSessionId || previousSessionId !== currentSessionId) {
      const messagesContainer = document.getElementById('messages-container');
      if (messagesContainer) messagesContainer.innerHTML = '';
      previousSessionId = currentSessionId;
    }
    
    input.value = '';
    appendMessage('user', message);
    const target = appendMessage('assistant', '');
    let receivedContent = '';
    let currentAgentLabel = '';
    try {
      const res = await fetch(`${API_BASE}/chat`, { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${auth.token}` }, 
        body: JSON.stringify({ 
          message,
          session_id: currentSessionId 
        }) 
      });
      if (!res.ok) throw new Error((await res.json()).detail || '聊天请求失败');
      if (!res.body) throw new Error('服务端未返回可读流');
      const reader = res.body.getReader(); 
      const decoder = new TextDecoder(); 
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read(); 
        if (value) buffer += decoder.decode(value, { stream: true });
        if (done) break;
        const parts = buffer.split('\n\n'); 
        buffer = parts.pop();
        for (const part of parts) {
          if (!part.startsWith('data: ')) continue;
          const raw = part.slice(6).trim();
          if (!raw) continue;
          let evt;
          try {
            evt = JSON.parse(raw);
          } catch (parseErr) {
            continue;
          }
          if (evt.type === 'agent') {
            currentAgentLabel = evt.agent_type || '';
          }
          if (evt.type === 'content') {
            receivedContent += evt.content || '';
            if (target) target.textContent = receivedContent;
          }
          if (evt.type === 'error') {
            const errorText = evt.content || '请求出错';
            if (target) target.textContent = errorText;
            receivedContent = errorText;
          }
          if (evt.type === 'done' && evt.session_id) {
            currentSessionId = evt.session_id;
            previousSessionId = evt.session_id;
          }
        }
      }
      if (buffer.trim().startsWith('data: ')) {
        const raw = buffer.trim().slice(6).trim();
        try {
          const evt = JSON.parse(raw);
          if (evt.type === 'content') {
            receivedContent += evt.content || '';
            if (target) target.textContent = receivedContent;
          }
          if (evt.type === 'agent') currentAgentLabel = evt.agent_type || currentAgentLabel;
          if (evt.type === 'done' && evt.session_id) { currentSessionId = evt.session_id; previousSessionId = evt.session_id; }
          if (evt.type === 'error') {
            const errorText = evt.content || '请求出错';
            if (target) target.textContent = errorText;
            receivedContent = errorText;
          }
        } catch {}
      }
      if (target && !receivedContent.trim()) {
        target.textContent = '暂无回答';
      }
      if (currentAgentLabel && target && !target.dataset.agent) {
        target.dataset.agent = currentAgentLabel;
      }
      await loadHistory();
    } catch (err) { 
      if (target) target.textContent = err.message; 
    }
  };

  // 发送按钮点击事件
  document.getElementById('send-btn')?.addEventListener('click', async () => {
    await sendMessage();
  });

  // 键盘事件处理：Enter发送，Shift+Enter换行
  const messageInput = document.getElementById('input-field');
  if (messageInput) {
    messageInput.addEventListener('keydown', async e => {
      if (e.key === 'Enter') {
        if (e.shiftKey) {
          // Shift+Enter: 换行
          return;
        } else {
          // Enter: 发送消息
          e.preventDefault();
          await sendMessage();
        }
      }
    });
  }
}

document.addEventListener('DOMContentLoaded', () => { initLoginPage(); initMainPage(); });

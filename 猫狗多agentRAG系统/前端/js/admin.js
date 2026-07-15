const API_BASE = 'http://127.0.0.1:8000';
const auth = {
  get token() { return localStorage.getItem('fauna_token') || sessionStorage.getItem('fauna_token'); },
  get user() { try { return JSON.parse(localStorage.getItem('fauna_user') || sessionStorage.getItem('fauna_user') || 'null'); } catch { return null; } },
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

async function initAdminPage() {
  // 检查登录状态
  if (!auth.token) {
    location.href = 'login.html';
    return;
  }

  const user = auth.user;
  if (!user?.is_admin) {
    // 如果不是管理员，跳转到普通用户界面
    location.href = 'index.html';
    return;
  }

  // 更新用户信息
  const userNameEl = document.getElementById('current-user-name');
  if (userNameEl) userNameEl.textContent = user?.username || '管理员';

  // 绑定事件
  document.getElementById('btn-logout')?.addEventListener('click', async () => {
    try { await api('/auth/logout', { method: 'POST' }); } catch {}
    auth.clear();
    location.href = 'login.html';
  });

  document.getElementById('btn-back-to-main')?.addEventListener('click', () => {
    location.href = 'index.html';
  });

  // 用户搜索
  document.getElementById('user-search-input')?.addEventListener('input', debounce(async (e) => {
    await loadUserList(e.target.value);
  }, 300));

  // 加载数据
  await loadAdminStats();
  await loadUserList();
}

async function loadAdminStats() {
  try {
    const data = await api('/admin/summary');
    const statUsers = document.getElementById('stat-users');
    const statSessions = document.getElementById('stat-sessions');
    const statMessages = document.getElementById('stat-messages');
    if (statUsers) statUsers.textContent = data.users;
    if (statSessions) statSessions.textContent = data.active_sessions;
    if (statMessages) statMessages.textContent = data.messages;
  } catch (err) {
    console.error('加载统计失败:', err);
  }
}

async function loadUserList(search = '') {
  const listEl = document.getElementById('user-list');
  if (!listEl) return;

  try {
    const data = await api(`/admin/users?search=${encodeURIComponent(search)}`);
    if (!data.users?.length) {
      listEl.innerHTML = '<div class="muted empty-state">暂无用户</div>';
      return;
    }

    listEl.innerHTML = data.users.map(u => `
      <div class="user-item" data-user-id="${u.id}">
        <div class="user-info">
          <div class="user-avatar-sm">${u.username.charAt(0).toUpperCase()}</div>
          <div class="user-details">
            <span class="user-name">${u.username}</span>
            <span class="user-email">${u.email}</span>
          </div>
        </div>
        <div class="user-meta">
          <span class="user-role ${u.is_admin ? 'admin' : ''}">${u.is_admin ? '管理员' : '普通用户'}</span>
          <span class="user-date">${formatDate(u.created_at)}</span>
        </div>
        <div class="user-actions">
          <button class="action-btn" onclick="toggleAdmin('${u.id}', ${!u.is_admin})">
            ${u.is_admin ? '取消管理员' : '设为管理员'}
          </button>
          <button class="action-btn delete" onclick="deleteUser('${u.id}')">删除</button>
        </div>
      </div>
    `).join('');
  } catch (err) {
    listEl.innerHTML = `<div class="muted">加载失败: ${err.message}</div>`;
  }
}

async function toggleAdmin(userId, makeAdmin) {
  if (!confirm(makeAdmin ? '确定要将该用户设为管理员吗？' : '确定要取消该用户的管理员权限吗？')) return;

  try {
    await api(`/admin/users/${userId}`, {
      method: 'PATCH',
      body: JSON.stringify({ is_admin: makeAdmin })
    });
    await loadUserList();
  } catch (err) {
    alert('操作失败: ' + err.message);
  }
}

async function deleteUser(userId) {
  if (!confirm('确定要删除该用户吗？此操作不可撤销！')) return;

  try {
    await api(`/admin/users/${userId}`, { method: 'DELETE' });
    await loadUserList();
  } catch (err) {
    alert('删除失败: ' + err.message);
  }
}

function formatDate(dateStr) {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
}

function debounce(fn, delay) {
  let timer = null;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

document.addEventListener('DOMContentLoaded', () => {
  initAdminPage();
});
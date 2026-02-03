const axios = require('axios');

// API 기본 URL
const API_BASE_URL = 'http://localhost:5000/api';

// 토큰 저장
let authToken = null;
let currentUser = null;

// API 클라이언트
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json'
    }
});

// 요청 인터셉터 (토큰 추가)
api.interceptors.request.use(config => {
    if (authToken) {
        config.headers.Authorization = `Bearer ${authToken}`;
    }
    return config;
});

// 화면 전환
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.add('hidden');
    });
    document.getElementById(screenId).classList.remove('hidden');
}

// 페이지 전환
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageId + '-page').classList.add('active');
    
    document.querySelectorAll('.menu-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`[data-page="${pageId}"]`).classList.add('active');
}

// 로그인
async function login(username, password) {
    try {
        const response = await api.post('/auth/login', { username, password });
        authToken = response.data.token;
        currentUser = response.data.user;
        
        document.getElementById('user-info').textContent = 
            `${currentUser.full_name} (${currentUser.role})`;
        
        showScreen('main-screen');
        loadDashboard();
        
    } catch (error) {
        const errorMsg = error.response?.data?.error || '로그인 실패';
        document.getElementById('login-error').textContent = errorMsg;
    }
}

// 로그아웃
function logout() {
    authToken = null;
    currentUser = null;
    document.getElementById('login-form').reset();
    document.getElementById('login-error').textContent = '';
    showScreen('login-screen');
}

// 대시보드 로드
async function loadDashboard() {
    try {
        const [coursesRes, studentsRes] = await Promise.all([
            api.get('/courses'),
            api.get('/users?role=student')
        ]);
        
        document.getElementById('total-courses').textContent = coursesRes.data.length;
        document.getElementById('total-students').textContent = studentsRes.data.length;
        document.getElementById('today-attendance').textContent = '85%'; // 임시 데이터
        
    } catch (error) {
        console.error('대시보드 로드 오류:', error);
    }
}

// 과목 목록 로드
async function loadCourses() {
    try {
        const response = await api.get('/courses');
        const coursesList = document.getElementById('courses-list');
        
        if (response.data.length === 0) {
            coursesList.innerHTML = '<p>등록된 과목이 없습니다.</p>';
            return;
        }
        
        let html = '<table><thead><tr><th>과목코드</th><th>과목명</th><th>교수</th><th>학기</th><th>년도</th></tr></thead><tbody>';
        
        response.data.forEach(course => {
            html += `
                <tr>
                    <td>${course.course_code}</td>
                    <td>${course.course_name}</td>
                    <td>${course.professor_name || '-'}</td>
                    <td>${course.semester}</td>
                    <td>${course.year}</td>
                </tr>
            `;
        });
        
        html += '</tbody></table>';
        coursesList.innerHTML = html;
        
    } catch (error) {
        console.error('과목 로드 오류:', error);
    }
}

// 학생 목록 로드
async function loadStudents() {
    try {
        const response = await api.get('/users?role=student');
        const studentsList = document.getElementById('students-list');
        
        if (response.data.length === 0) {
            studentsList.innerHTML = '<p>등록된 학생이 없습니다.</p>';
            return;
        }
        
        let html = '<table><thead><tr><th>학번</th><th>이름</th><th>아이디</th><th>이메일</th></tr></thead><tbody>';
        
        response.data.forEach(student => {
            html += `
                <tr>
                    <td>${student.student_number || '-'}</td>
                    <td>${student.full_name}</td>
                    <td>${student.username}</td>
                    <td>${student.email}</td>
                </tr>
            `;
        });
        
        html += '</tbody></table>';
        studentsList.innerHTML = html;
        
    } catch (error) {
        console.error('학생 로드 오류:', error);
    }
}

// 이벤트 리스너
document.addEventListener('DOMContentLoaded', () => {
    // 로그인 폼
    document.getElementById('login-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        login(username, password);
    });
    
    // 로그아웃 버튼
    document.getElementById('logout-btn').addEventListener('click', logout);
    
    // 메뉴 아이템
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = e.target.dataset.page;
            showPage(page);
            
            // 페이지별 데이터 로드
            if (page === 'courses') {
                loadCourses();
            } else if (page === 'students') {
                loadStudents();
            } else if (page === 'dashboard') {
                loadDashboard();
            }
        });
    });
});

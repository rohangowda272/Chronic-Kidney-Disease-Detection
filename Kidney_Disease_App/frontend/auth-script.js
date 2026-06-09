// Authentication System
// ========================================

let loggedInUser = null;
let allUsers = JSON.parse(localStorage.getItem('appUsers')) || [];

// ========================================
// FORM VALIDATION
// ========================================

function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function validatePassword(password) {
    return password.length >= 8;
}

function getPasswordStrength(password) {
    let strength = 0;
    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[^a-zA-Z0-9]/.test(password)) strength++;
    return strength;
}

// ========================================
// LOGIN FUNCTIONALITY
// ========================================

const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
}

function handleLogin(e) {
    e.preventDefault();
    
    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value;
    const rememberMe = document.getElementById('rememberMe').checked;

    const emailError = document.getElementById('loginEmailError');
    const passwordError = document.getElementById('loginPasswordError');
    
    let hasError = false;

    // Clear previous errors
    emailError.textContent = '';
    passwordError.textContent = '';

    if (!validateEmail(email)) {
        emailError.textContent = 'Please enter a valid email address';
        hasError = true;
    }

    if (!password) {
        passwordError.textContent = 'Password is required';
        hasError = true;
    }

    if (hasError) return;

    const user = allUsers.find(u => u.email === email && u.password === password);

    if (user) {
        loggedInUser = user;
        sessionStorage.setItem('loggedInUser', JSON.stringify(user));
        
        if (rememberMe) {
            localStorage.setItem('rememberMe', 'true');
            localStorage.setItem('userEmail', email);
        } else {
            localStorage.removeItem('rememberMe');
            localStorage.removeItem('userEmail');
        }
        
        showToast(`Welcome back, ${user.fullName}!`, 'success');
        
        setTimeout(() => {
            window.location.href = 'prediction.html';
        }, 1500);
    } else {
        showToast('Invalid email or password', 'error');
    }
}

// ========================================
// SIGNUP FUNCTIONALITY
// ========================================

const signupForm = document.getElementById('signupForm');
if (signupForm) {
    signupForm.addEventListener('submit', handleSignup);
    const signupPasswordInput = document.getElementById('signupPassword');
    if (signupPasswordInput) {
        signupPasswordInput.addEventListener('input', updatePasswordStrength);
    }
}

function updatePasswordStrength(e) {
    const password = e.target.value;
    const strength = getPasswordStrength(password);
    const strengthBar = document.querySelector('.strength-bar');
    
    if (!strengthBar) return;
    
    const strengthColors = ['#e74c3c', '#f39c12', '#f39c12', '#27ae60', '#27ae60'];
    const strengthWidths = ['20%', '40%', '60%', '80%', '100%'];
    
    if (password.length === 0) {
        strengthBar.style.width = '0%';
        strengthBar.style.backgroundColor = '#ddd';
    } else {
        strengthBar.style.backgroundColor = strengthColors[strength - 1] || '#e74c3c';
        strengthBar.style.width = strengthWidths[strength - 1] || '20%';
    }
}

function handleSignup(e) {
    e.preventDefault();

    const fullName = document.getElementById('signupName').value.trim();
    const email = document.getElementById('signupEmail').value.trim();
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('signupConfirmPassword').value;

    // Clear previous errors
    document.getElementById('signupNameError').textContent = '';
    document.getElementById('signupEmailError').textContent = '';
    document.getElementById('signupPasswordError').textContent = '';
    document.getElementById('signupConfirmPasswordError').textContent = '';

    let hasError = false;

    if (!fullName || fullName.length < 2) {
        document.getElementById('signupNameError').textContent = 'Please enter a valid name (at least 2 characters)';
        hasError = true;
    }

    if (!validateEmail(email)) {
        document.getElementById('signupEmailError').textContent = 'Please enter a valid email address';
        hasError = true;
    }

    if (allUsers.some(u => u.email === email)) {
        document.getElementById('signupEmailError').textContent = 'This email is already registered';
        hasError = true;
    }

    if (!validatePassword(password)) {
        document.getElementById('signupPasswordError').textContent = 'Password must be at least 8 characters';
        hasError = true;
    }

    if (password !== confirmPassword) {
        document.getElementById('signupConfirmPasswordError').textContent = 'Passwords do not match';
        hasError = true;
    }

    if (hasError) return;

    const newUser = {
        id: Date.now(),
        fullName,
        email,
        password,
        createdAt: new Date().toISOString()
    };

    allUsers.push(newUser);
    localStorage.setItem('appUsers', JSON.stringify(allUsers));

    showToast('Account created successfully! Redirecting...', 'success');
    
    setTimeout(() => {
        loggedInUser = newUser;
        sessionStorage.setItem('loggedInUser', JSON.stringify(newUser));
        window.location.href = 'prediction.html';
    }, 1500);
}

// ========================================
// UI HELPERS
// ========================================

function togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    if (!input) return;
    
    if (input.type === 'password') {
        input.type = 'text';
    } else {
        input.type = 'password';
    }
}

function switchToSignup() {
    const loginPage = document.getElementById('loginPage');
    const signupPage = document.getElementById('signupPage');
    
    if (loginPage) loginPage.classList.remove('active');
    if (signupPage) signupPage.classList.add('active');
}

function switchToLogin() {
    const loginPage = document.getElementById('loginPage');
    const signupPage = document.getElementById('signupPage');
    
    if (signupPage) signupPage.classList.remove('active');
    if (loginPage) loginPage.classList.add('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) {
        console.error('Toast container not found!');
        return;
    }
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

function handleLogout() {
    if (confirm('Are you sure you want to logout?')) {
        sessionStorage.removeItem('loggedInUser');
        localStorage.removeItem('rememberMe');
        localStorage.removeItem('userEmail');
        showToast('Logged out successfully', 'success');
        
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1000);
    }
}

// ========================================
// INITIALIZATION
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('✓ Authentication system loaded');
    
    const rememberMe = localStorage.getItem('rememberMe');
    const userEmail = localStorage.getItem('userEmail');
    
    if (rememberMe && userEmail) {
        const loginEmailInput = document.getElementById('loginEmail');
        const rememberMeCheckbox = document.getElementById('rememberMe');
        
        if (loginEmailInput) {
            loginEmailInput.value = userEmail;
        }
        if (rememberMeCheckbox) {
            rememberMeCheckbox.checked = true;
        }
    }
});

// ========================================
// CHECK IF USER IS LOGGED IN (For protected pages)
// ========================================

function checkUserLoggedIn() {
    const loggedInUser = sessionStorage.getItem('loggedInUser');
    if (!loggedInUser) {
        showToast('Please login to access this page', 'warning');
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1500);
    }
    return loggedInUser ? JSON.parse(loggedInUser) : null;
}
# Quick Reference: Key Code Changes

## 1. app.py - Authentication Protection

### Added Helper Function
```python
def require_login():
    """Check if user is logged in via session"""
    return 'user_id' in session
```

### Protected Routes
```python
@app.route('/')
def landing():
    if not require_login():
        return redirect(url_for('login_page'))
    return render_template('landing.html')

@app.route('/login')
def login_page():
    if require_login():
        return redirect(url_for('landing'))
    return render_template('login.html')

@app.route('/upload')
def upload_page():
    if not require_login():
        return redirect(url_for('login_page'))
    return render_template('upload.html')
```

### Protected API Endpoints
```python
@app.route('/upload', methods=['POST'])
def upload_file():
    if not require_login():
        return jsonify({'error': 'Authentication required'}), 401
    # ... rest of upload logic

@app.route('/ask', methods=['POST'])
def ask_question():
    if not require_login():
        return jsonify({'error': 'Authentication required'}), 401
    # ... rest of Q&A logic
```

## 2. login.html - Authentication Page

### Form Submission
```javascript
loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const email = emailInput.value.trim();
    const password = passwordInput.value;
    
    const response = await fetch("/auth", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ email, password })
    });
    
    const data = await response.json();
    
    if (response.ok && data.success) {
        // Success - redirect to home
        window.location.href = "/";
    } else {
        // Show error
        showError(data.error);
    }
});
```

## 3. All Pages - Auth Check & Logout

### Authentication Check (on page load)
```javascript
async function checkAuth() {
    try {
        const response = await fetch("/me");
        const data = await response.json();
        
        if (!data.authenticated) {
            window.location.href = "/login";
        }
    } catch (error) {
        window.location.href = "/login";
    }
}

checkAuth();
```

### Logout Functionality
```javascript
logoutBtn.addEventListener("click", async () => {
    await fetch("/logout", {method: "POST"});
    window.location.href = "/login";
});
```

## 4. chat.html - Chat History Saving

### Save Message to Database
```javascript
async function saveChatMessage(userMessage, aiResponse) {
    try {
        await fetch("/chat/save", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                user_message: userMessage,
                ai_response: aiResponse
            })
        });
    } catch (error) {
        console.error("Error saving chat:", error);
    }
}
```

### Called After Q&A
```javascript
chatForm.addEventListener("submit", async (e) => {
    // ... get question and answer
    
    if (response.ok) {
        const answer = data.answer;
        addMessage(answer, false);
        
        // Save to database
        saveChatMessage(question, answer);
    }
});
```

## 5. Backend Routes (Already Exist)

### Authentication Routes (routes/auth_routes.py)
```python
@auth_bp.route('/auth', methods=['POST'])
def authenticate():
    """Login or signup - checks if email exists"""
    email = data.get('email')
    password = data.get('password')
    
    user = get_user_by_email(email)
    
    if user:
        # LOGIN
        if verify_password(user['password_hash'], password):
            set_session_user(user['id'])
            return jsonify({'success': True, 'action': 'login'})
    else:
        # SIGNUP
        new_user = create_user(email, password)
        set_session_user(new_user['id'])
        return jsonify({'success': True, 'action': 'signup'})

@auth_bp.route('/logout', methods=['POST'])
def logout():
    clear_session()
    return jsonify({'success': True})

@auth_bp.route('/me', methods=['GET'])
@login_required
def get_me():
    user = get_current_user()
    return jsonify({'authenticated': True, 'user': user})
```

### Chat History Routes (routes/chat_routes.py)
```python
@chat_bp.route('/history', methods=['GET'])
@login_required
def get_chat_history():
    user = get_current_user()
    # Fetch last 20 messages from database
    return jsonify({'messages': messages})

@chat_bp.route('/save', methods=['POST'])
@login_required
def save_chat_message():
    user = get_current_user()
    user_message = data.get('user_message')
    ai_response = data.get('ai_response')
    # Save to database
    return jsonify({'success': True})
```

## 6. Session Management

### How Session Works
```python
# utils/auth.py

def set_session_user(user_id):
    """Store user_id in Flask session after login"""
    session['user_id'] = user_id
    session.permanent = True

def get_current_user():
    """Get user from session"""
    user_id = session.get('user_id')
    if not user_id:
        return None
    # Fetch from database
    return user

def clear_session():
    """Clear session on logout"""
    session.clear()
```

## 7. Database Connection

### Connection Pooling (database/db.py)
```python
from psycopg2 import pool

# Initialize pool
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

# Context manager for queries
@contextmanager
def get_db_cursor():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
        finally:
            cursor.close()
```

## Testing Commands

### Test Authentication
```bash
# Signup/Login
curl -X POST http://localhost:5000/auth \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password123"}' \
  -c cookies.txt

# Check current user
curl http://localhost:5000/me -b cookies.txt

# Logout
curl -X POST http://localhost:5000/logout -b cookies.txt
```

### Test Chat History
```bash
# Save message
curl -X POST http://localhost:5000/chat/save \
  -H "Content-Type: application/json" \
  -d '{"user_message":"Question?","ai_response":"Answer."}' \
  -b cookies.txt

# Get history
curl http://localhost:5000/chat/history -b cookies.txt
```

## File Structure
```
research-paper-chatbot/
├── app.py                      # ✅ Modified - added auth protection
├── backend.py                  # No changes
├── requirements.txt            # ✅ Modified - added psycopg2-binary
├── .env                        # ✅ Created - environment variables
├── init_db.py                  # ✅ Created - database initialization
├── test_auth.py                # ✅ Created - testing script
│
├── database/
│   └── db.py                   # ✅ Created - PostgreSQL connection
│
├── routes/
│   ├── auth_routes.py          # ✅ Created - login/signup/logout
│   └── chat_routes.py          # ✅ Created - chat history
│
├── utils/
│   └── auth.py                 # ✅ Created - auth helpers
│
├── templates/
│   ├── login.html              # ✅ Created - login page
│   ├── landing.html            # ✅ Modified - added logout & auth check
│   ├── chat.html               # ✅ Modified - added logout & chat saving
│   ├── upload.html             # ✅ Modified - added logout & auth check
│   └── summary.html            # ✅ Modified - added logout & auth check
│
└── static/
    └── js/
        └── main.js             # ✅ Modified - added auth error handling
```

## Key Features Implemented

✅ Login/Signup with single endpoint
✅ Session-based authentication
✅ Protected routes (redirects to login)
✅ Logout button on all pages
✅ Authentication check on page load
✅ Automatic chat history saving
✅ Secure password hashing
✅ Database connection pooling
✅ Error handling with redirects
✅ Clean, modular code structure

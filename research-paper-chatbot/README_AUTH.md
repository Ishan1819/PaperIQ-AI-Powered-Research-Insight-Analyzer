# PaperIQ Authentication System

## Overview
Production-ready authentication system for Flask application with PostgreSQL database.

## Environment Variables
Create a `.env` file in the project root with the following variables:

```bash
# Flask Configuration
SECRET_KEY=your-super-secure-secret-key-here-change-in-production
FLASK_ENV=production  # or development

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=paperiq_db
DB_USER=postgres
DB_PASSWORD=your-database-password
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (see above)

3. Verify database tables exist (see Database Setup section)

## Database Setup

The following tables should already exist in your PostgreSQL database:

```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat history table
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    user_message TEXT,
    ai_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX idx_chat_history_created_at ON chat_history(created_at DESC);
```

## API Endpoints

### Authentication Endpoints

#### POST /auth
Single endpoint for both login and signup.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response (Login):**
```json
{
  "success": true,
  "action": "login",
  "message": "Login successful",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "created_at": "2026-03-12T10:00:00"
  }
}
```

**Response (Signup):**
```json
{
  "success": true,
  "action": "signup",
  "message": "Account created successfully",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "created_at": "2026-03-12T10:00:00"
  }
}
```

#### POST /logout
Logout the current user and clear session.

**Response:**
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

#### GET /me
Get current logged-in user information (requires authentication).

**Response:**
```json
{
  "authenticated": true,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "created_at": "2026-03-12T10:00:00"
  }
}
```

#### GET /check-auth
Check authentication status without requiring login.

**Response:**
```json
{
  "authenticated": true,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "created_at": "2026-03-12T10:00:00"
  }
}
```

### Chat History Endpoints

#### GET /chat/history
Retrieve last 20 chat messages for logged-in user (requires authentication).

**Response:**
```json
{
  "success": true,
  "count": 15,
  "messages": [
    {
      "id": 123,
      "user_id": 1,
      "user_message": "What is the conclusion?",
      "ai_response": "The conclusion states...",
      "created_at": "2026-03-12T10:30:00"
    }
  ]
}
```

#### POST /chat/save
Save a chat message to database (requires authentication).

**Request Body:**
```json
{
  "user_message": "What is the main finding?",
  "ai_response": "The main finding is..."
}
```

**Response:**
```json
{
  "success": true,
  "message": "Chat message saved",
  "id": 124,
  "created_at": "2026-03-12T10:35:00"
}
```

#### DELETE /chat/clear
Delete all chat history for current user (requires authentication).

**Response:**
```json
{
  "success": true,
  "message": "Deleted 15 messages",
  "deleted_count": 15
}
```

## Project Structure

```
research-paper-chatbot/
├── app.py                      # Main Flask application
├── backend.py                  # Document processing logic
├── requirements.txt            # Python dependencies
├── database/
│   └── db.py                  # PostgreSQL connection & pooling
├── routes/
│   ├── auth_routes.py         # Authentication endpoints
│   └── chat_routes.py         # Chat history endpoints
├── utils/
│   └── auth.py                # Authentication helpers
├── static/
│   ├── css/
│   └── js/
└── templates/
    ├── landing.html
    ├── upload.html
    ├── summary.html
    └── chat.html
```

## Security Features

✅ **Password Security**
- Passwords hashed using Werkzeug's `pbkdf2:sha256`
- Never stores plain text passwords
- Minimum 6 character password requirement

✅ **Session Security**
- HTTP-only cookies (prevents XSS attacks)
- Secure flag in production (HTTPS only)
- SameSite=Lax (CSRF protection)
- 1-hour session timeout

✅ **Database Security**
- Parameterized queries (prevents SQL injection)
- Connection pooling with automatic cleanup
- Environment variables for credentials

✅ **Authentication**
- `@login_required` decorator for protected routes
- Session-based authentication
- Automatic user creation on signup

## Usage in Frontend

### Example: Login/Signup
```javascript
async function authenticate(email, password) {
  const response = await fetch('/auth', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ email, password })
  });
  
  const data = await response.json();
  
  if (data.success) {
    console.log(`${data.action} successful:`, data.user);
    // Redirect to main app
  } else {
    console.error('Error:', data.error);
  }
}
```

### Example: Check Authentication
```javascript
async function checkAuth() {
  const response = await fetch('/check-auth');
  const data = await response.json();
  
  if (data.authenticated) {
    console.log('User is logged in:', data.user);
  } else {
    console.log('User is not logged in');
    // Redirect to login page
  }
}
```

### Example: Save Chat Message
```javascript
async function saveChatMessage(question, answer) {
  const response = await fetch('/chat/save', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      user_message: question,
      ai_response: answer
    })
  });
  
  const data = await response.json();
  console.log('Message saved:', data);
}
```

### Example: Get Chat History
```javascript
async function getChatHistory() {
  const response = await fetch('/chat/history');
  const data = await response.json();
  
  if (data.success) {
    console.log(`Found ${data.count} messages:`, data.messages);
  }
}
```

## Running the Application

### Development Mode
```bash
python app.py
```

### Production Mode (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using systemd (recommended for AWS EC2)
Create `/etc/systemd/system/paperiq.service`:
```ini
[Unit]
Description=PaperIQ Flask Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/path/to/research-paper-chatbot
Environment="PATH=/path/to/venv/bin"
EnvironmentFile=/path/to/.env
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable paperiq
sudo systemctl start paperiq
```

## Testing

### Test Database Connection
```python
from database.db import init_db_pool, test_connection

init_db_pool()
if test_connection():
    print("Database connected successfully!")
```

### Test Authentication
```bash
# Signup
curl -X POST http://localhost:5000/auth \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Login
curl -X POST http://localhost:5000/auth \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Get current user
curl http://localhost:5000/me --cookie "session=..."
```

## Troubleshooting

### Database Connection Issues
1. Verify PostgreSQL is running: `sudo systemctl status postgresql`
2. Check database credentials in `.env` file
3. Ensure database and tables exist
4. Check firewall rules if database is remote

### Session Issues
1. Ensure `SECRET_KEY` is set in environment
2. Check browser cookie settings
3. Verify session cookie is being sent with requests

### Import Errors
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python path includes project directory
3. Verify all files are in correct locations

## Next Steps

1. **Frontend Integration**: Update HTML templates to call authentication endpoints
2. **Password Reset**: Implement password reset functionality
3. **Email Verification**: Add email verification for new signups
4. **Rate Limiting**: Add rate limiting to prevent brute force attacks
5. **Logging**: Implement comprehensive logging for security events
6. **2FA**: Add two-factor authentication for enhanced security

## Support

For issues or questions, please refer to the project documentation or contact the development team.

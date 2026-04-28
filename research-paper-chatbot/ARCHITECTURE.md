# Authentication System Architecture

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                 │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                │ 1. Visit http://your-site.com/
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FLASK APP (app.py)                              │
│                                                                      │
│  @app.route('/')                                                     │
│  def landing():                                                      │
│      if not require_login():  ◄─── Checks session['user_id']       │
│          return redirect('/login')                                   │
│      return render_template('landing.html')                          │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                │ 2. Not authenticated → Redirect to /login
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LOGIN PAGE (login.html)                            │
│                                                                      │
│  [Email Input: user@example.com        ]                            │
│  [Password Input: ••••••••             ]                            │
│  [        Sign In / Sign Up Button     ]                            │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                │ 3. Submit email + password
                │ POST /auth {email, password}
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              AUTH ROUTES (routes/auth_routes.py)                     │
│                                                                      │
│  @auth_bp.route('/auth', methods=['POST'])                          │
│  def authenticate():                                                 │
│      email = request.json['email']                                  │
│      password = request.json['password']                            │
│                                                                      │
│      user = get_user_by_email(email)  ◄─┐                          │
│                                          │                           │
│      if user exists:                     │                           │
│          # LOGIN FLOW                    │                           │
│          if verify_password(...):        │                           │
│              set_session_user(user.id) ──┼─► session['user_id'] = 1│
│              return {"action": "login"}  │                           │
│      else:                               │                           │
│          # SIGNUP FLOW                   │                           │
│          new_user = create_user(...)   ──┼─► INSERT INTO users      │
│          set_session_user(user.id)     ──┘                          │
│          return {"action": "signup"}                                 │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                │ 4. Session cookie set → Redirect to /
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LANDING PAGE (/)                                  │
│                                                                      │
│  JavaScript on page load:                                            │
│  checkAuth() → fetch('/me')  ◄─── Reads session cookie              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  🎓 Research Paper Chatbot          [🚪 Logout]    │           │
│  │                                                      │           │
│  │         Upload • Summarize • Chat                   │           │
│  │                                                      │           │
│  │              [Get Started]                           │           │
│  └─────────────────────────────────────────────────────┘           │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                │ 5. Click "Get Started" → /upload
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   UPLOAD PAGE (/upload)                              │
│                                                                      │
│  Protection: require_login() checks session['user_id']              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  📁 Drag & drop PDF or DOCX here                    │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                      │
│  POST /upload → Processes file → Stores in memory                   │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                │ 6. Document uploaded → /chat
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CHAT PAGE (/chat)                                 │
│                                                                      │
│  Protection: require_login() + require_document()                   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  💬 Chat with Your Document                         │           │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │           │
│  │  User: What is the main finding?                    │           │
│  │  Bot: The main finding is...                        │           │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │           │
│  │  [Type your question...] [Send]                     │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                      │
│  On submit:                                                          │
│  1. POST /ask {question} → Get AI response                          │
│  2. saveChatMessage() → POST /chat/save {q, a}  ◄─┐                │
└────────────────────────────────────────────────────┼────────────────┘
                                                      │
                                                      │
                 ┌────────────────────────────────────┘
                 │ 7. Save to database
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  POSTGRESQL DATABASE                                 │
│                                                                      │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐│
│  │  users                  │  │  chat_history                     ││
│  ├─────────────────────────┤  ├──────────────────────────────────┤│
│  │ id: 1                   │  │ id: 1                             ││
│  │ email: user@example.com │◄─┤ user_id: 1 (FK)                  ││
│  │ password_hash: $pbkdf2..│  │ user_message: "What is..."       ││
│  │ created_at: 2026-03-12  │  │ ai_response: "The main..."       ││
│  └─────────────────────────┘  │ created_at: 2026-03-12 10:30:00  ││
│                                └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Authentication Flow

```
USER ACTION                    FRONTEND                    BACKEND                 DATABASE
                                                                                   
Visit site         ──────────►  landing.html   ───────────► require_login()
                                                             │
                                                             │ Check session
                                                             ▼
                                                          session.get('user_id')
                                                             │
Not logged in      ◄─────────  redirect        ◄───────────┘ None
                                                             
Visit /login       ──────────► login.html
                               │
Enter email/pass               │
Click submit       ──────────► POST /auth
                               {email, password}
                                                             
                                                ───────────► get_user_by_email() ──► SELECT * FROM users
                                                                                      WHERE email = ?
                                                                                      │
If email exists                                                                      │
  (LOGIN)                                      ◄──────────────────────────────────────┘
                                               verify_password()
                                               │
                                               ▼
                                               set_session_user(id)
                                               │
                                               ▼
                                               session['user_id'] = 1
                                               │
Success!           ◄─────────  200 OK         ◄┘
                               {success: true}
                               │
Redirect to /      ◄───────────┘

If email NOT exists
  (SIGNUP)                                    ───────────► create_user()      ──────► INSERT INTO users
                                                                                      (email, password_hash)
                                                                                      RETURNING id
                                                                                      │
Success!           ◄─────────  200 OK         ◄──────────────────────────────────────┘
                               {action: signup}
```

## Chat History Saving Flow

```
USER                          FRONTEND (chat.html)                BACKEND                      DATABASE

Type question     ────────►   Question input
                              │
Click send        ────────►   POST /ask
                              {question: "..."}
                                                      ────────────►  /ask endpoint
                                                                     │
                                                                     │ require_login()
                                                                     │ get AI response
                                                                     │
AI response       ◄──────────  200 OK                ◄──────────────┘
shown on screen               {answer: "..."}
                              │
                              │ Display answer
                              │
                              ▼
                              saveChatMessage()
                              POST /chat/save
                              {
                                user_message: "...",
                                ai_response: "..."
                              }
                                                      ────────────►  /chat/save endpoint
                                                                     │
                                                                     │ get_current_user()
                                                                     │ user_id from session
                                                                     ▼
                                                                     INSERT INTO     ──────────►  chat_history
                                                                     chat_history                 (user_id,
                                                                     (user_id, ...)                user_message,
                                                                                                   ai_response)
Silently saved    ◄──────────  200 OK                ◄──────────────┘
in background                 {success: true}
```

## Session Management

```
LOGIN SUCCESS                         SESSION COOKIE                         SUBSEQUENT REQUESTS
                                                                            
User logs in                          Set-Cookie:                           User visits /chat
  ↓                                   session=eyJ...;
set_session_user(user_id=1)           HttpOnly;                            require_login() checks:
  ↓                                   Secure;                               ↓
session['user_id'] = 1                SameSite=Lax                         session.get('user_id')
  ↓                                   Max-Age=3600                          ↓
Flask creates session                   ↓                                   Returns: 1
  ↓                                   Stored in:                            ↓
Response includes cookie              Browser cookies                       User is authenticated!
                                        ↓                                     ↓
                                      Sent with ALL                         Allow access to page
                                      future requests                         ↓
                                      to same domain                        Render chat.html
```

## Logout Flow

```
USER                    FRONTEND              BACKEND                 SESSION

Click logout  ────────► POST /logout  ──────► clear_session()  ─────► session.clear()
                                               │                       │
                                               │                       │ Delete user_id
                                               ▼                       │ Delete all session data
Success       ◄───────  200 OK         ◄──────┘                       │
                        {success: true}                                ▼
Redirect to            │                                              Session empty
/login        ◄────────┘
```

## Database Schema Relationships

```
┌──────────────────────────┐
│        users             │
├──────────────────────────┤
│ id (PK)                  │───┐
│ email (UNIQUE)           │   │
│ password_hash            │   │ One-to-Many
│ created_at               │   │ Relationship
└──────────────────────────┘   │
                                │
                                │
                                ▼
                        ┌──────────────────────────┐
                        │    chat_history          │
                        ├──────────────────────────┤
                        │ id (PK)                  │
                        │ user_id (FK) ────────────┼─► References users(id)
                        │ user_message             │
                        │ ai_response              │
                        │ created_at               │
                        └──────────────────────────┘
```

## Security Layers

```
LAYER 1: Route Protection (app.py)
┌─────────────────────────────────────────┐
│ @app.route('/')                         │
│ def landing():                          │
│     if not require_login():             │  ◄─── Checks session['user_id']
│         return redirect('/login')       │
└─────────────────────────────────────────┘

LAYER 2: Frontend Auth Check (JavaScript)
┌─────────────────────────────────────────┐
│ async function checkAuth() {            │
│     response = await fetch('/me')       │  ◄─── Verifies session is valid
│     if (!response.authenticated)        │
│         redirect('/login')              │
│ }                                       │
└─────────────────────────────────────────┘

LAYER 3: API Endpoint Protection
┌─────────────────────────────────────────┐
│ @app.route('/ask', methods=['POST'])    │
│ def ask_question():                     │
│     if not require_login():             │  ◄─── Validates on each request
│         return 401 Unauthorized         │
└─────────────────────────────────────────┘

LAYER 4: Database Security
┌─────────────────────────────────────────┐
│ - Parameterized queries (no SQL inject) │
│ - Password hashing (pbkdf2:sha256)      │  ◄─── Never stores plain passwords
│ - Connection pooling                    │
│ - Environment variables for credentials │
└─────────────────────────────────────────┘
```

## Complete Request Lifecycle

```
1. Browser → GET / 
2. Flask app.py → require_login() → Check session['user_id'] → None
3. Flask → redirect('/login')
4. Browser → Render login.html
5. User enters credentials
6. Browser → POST /auth {email, password}
7. Flask routes/auth_routes.py → authenticate()
8. Query DB → SELECT * FROM users WHERE email = ?
9. If found → verify_password() → bcrypt check
10. If valid → set_session_user(user_id)
11. Flask session['user_id'] = 1
12. Flask → Set-Cookie: session=encrypted_data
13. Browser receives cookie
14. Browser → redirect to /
15. Browser → GET / (with cookie)
16. Flask → require_login() → session['user_id'] → Returns 1 ✓
17. Flask → render_template('landing.html')
18. Browser displays landing page
19. User clicks "Get Started"
20. Browser → GET /upload (with cookie)
21. All subsequent requests include cookie automatically
```

This architecture ensures secure, session-based authentication with automatic chat history saving!

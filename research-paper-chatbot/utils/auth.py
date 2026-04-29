"""
Authentication Utilities
Provides helper functions for user authentication and session management.
DEMO MODE: Session-only auth - no database required
"""

from functools import wraps
from flask import session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash


def hash_password(password):
    """
    Hash a plain text password using Werkzeug's secure hash function.
    
    Args:
        password (str): Plain text password
    
    Returns:
        str: Hashed password
    """
    return generate_password_hash(password, method='pbkdf2:sha256')


def verify_password(password_hash, password):
    """
    Verify a plain text password against a hashed password.
    
    Args:
        password_hash (str): Stored hashed password
        password (str): Plain text password to verify
    
    Returns:
        bool: True if password matches, False otherwise
    """
    return check_password_hash(password_hash, password)


def get_current_user():
    """
    Get the currently logged-in user from the session.
    Demo mode: Uses session storage only, no database.
    
    Returns:
        dict: User object with id, email or None if not logged in
    """
    user_id = session.get('user_id')
    
    if not user_id:
        return None
    
    # For demo mode, store minimal user info in session
    user_data = session.get('user_data')
    if user_data:
        return user_data
    
    return None


def get_user_by_email(email):
    """
    Check if a user exists by email.
    Demo mode: Uses session storage only, no database.
    Stores a simple in-memory user registry in the app.
    
    Args:
        email (str): User's email address
    
    Returns:
        dict: User object with password_hash, or None if not found
    """
    # In demo mode, we store all users in the session's app context
    # For simplicity, we'll use a simple in-memory store per session
    users_registry = session.get('_users_registry', {})
    
    email_lower = email.lower()
    if email_lower in users_registry:
        return users_registry[email_lower]
    
    return None


def create_user(email, password):
    """
    Create a new user in session storage.
    Demo mode: No database required.
    
    Args:
        email (str): User's email address
        password (str): Plain text password
    
    Returns:
        dict: Newly created user object or None if error
    """
    try:
        email_lower = email.lower()
        password_hash = hash_password(password)
        
        # Create user object
        new_user = {
            'id': hash(email_lower),  # Simple ID generation from email
            'email': email_lower,
            'password_hash': password_hash,
            'created_at': __import__('datetime').datetime.now()
        }
        
        # Store in session's user registry
        users_registry = session.get('_users_registry', {})
        users_registry[email_lower] = new_user
        session['_users_registry'] = users_registry
        
        return new_user
    except Exception as e:
        print(f"Error creating user: {e}")
        return None


def login_required(f):
    """
    Decorator to protect routes that require authentication.
    Returns 401 Unauthorized if user is not logged in.
    
    Usage:
        @app.route('/protected')
        @login_required
        def protected_route():
            return "This is protected"
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return jsonify({'error': 'Authentication required', 'authenticated': False}), 401
        return f(*args, **kwargs)
    return decorated_function


def set_session_user(user_id, user_data=None):
    """
    Set the user session after successful login/signup.
    Demo mode: Stores user data directly in session.
    
    Args:
        user_id (int): User's ID
        user_data (dict): User object with email, id, created_at
    """
    session['user_id'] = user_id
    if user_data:
        session['user_data'] = user_data
    session.permanent = True  # Makes session use PERMANENT_SESSION_LIFETIME


def clear_session():
    """
    Clear all session data on logout.
    """
    session.clear()


def validate_email(email):
    """
    Basic email validation.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email format is valid
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """
    Validate password strength.
    Requires at least 6 characters for basic security.
    
    Args:
        password (str): Password to validate
    
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    # Add more validation rules as needed:
    # - At least one uppercase letter
    # - At least one number
    # - At least one special character
    
    return True, ""

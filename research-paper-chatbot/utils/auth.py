"""
Authentication Utilities
Provides helper functions for user authentication and session management.
"""

from functools import wraps
from flask import session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import get_db_cursor


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
    Reads user_id from Flask session and fetches user data from PostgreSQL.
    
    Returns:
        dict: User object with id, email, created_at or None if not logged in
    """
    user_id = session.get('user_id')
    
    if not user_id:
        return None
    
    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                "SELECT id, email, created_at FROM users WHERE id = %s",
                (user_id,)
            )
            user = cursor.fetchone()
            return user
    except Exception as e:
        print(f"Error fetching current user: {e}")
        return None


def get_user_by_email(email):
    """
    Fetch a user from the database by email address.
    
    Args:
        email (str): User's email address
    
    Returns:
        dict: User object including password_hash, or None if not found
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                "SELECT id, email, password_hash, created_at FROM users WHERE email = %s",
                (email,)
            )
            user = cursor.fetchone()
            return user
    except Exception as e:
        print(f"Error fetching user by email: {e}")
        return None


def create_user(email, password):
    """
    Create a new user in the database.
    Hashes the password before storing.
    
    Args:
        email (str): User's email address
        password (str): Plain text password
    
    Returns:
        dict: Newly created user object or None if error
    """
    try:
        password_hash = hash_password(password)
        
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (email, password_hash)
                VALUES (%s, %s)
                RETURNING id, email, created_at
                """,
                (email, password_hash)
            )
            new_user = cursor.fetchone()
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


def set_session_user(user_id):
    """
    Set the user_id in the Flask session after successful login/signup.
    
    Args:
        user_id (int): User's database ID
    """
    session['user_id'] = user_id
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

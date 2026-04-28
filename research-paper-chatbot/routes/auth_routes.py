"""
Authentication Routes
Handles user login, signup, logout, and user info retrieval.
"""

from flask import Blueprint, request, jsonify, session
from utils.auth import (
    set_session_user,
    clear_session,
    validate_email,
    validate_password,
    login_required
)

# Create a Blueprint for auth routes
auth_bp = Blueprint('auth', __name__, url_prefix='')

# ✅ DEMO MODE: In-memory user storage (no database needed)
demo_users = {}


@auth_bp.route('/auth', methods=['POST'])
def authenticate():
    """
    ✅ DEMO MODE: Accept any email/password without database.
    
    Request Body:
        {
            "email": "user@example.com",
            "password": "password123"
        }
    
    Logic:
        - Accept any email and password
        - Store in session
        - Return success
    
    Returns:
        200: Success with user data
        400: Validation error
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validate input
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Validate email format
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password strength
        is_valid_password, password_error = validate_password(password)
        if not is_valid_password:
            return jsonify({'error': password_error}), 400
        
        # ✅ DEMO MODE: Just store email in session
        # Generate a simple user_id based on email hash
        user_id = abs(hash(email)) % 1000000
        
        # Store in in-memory user list
        demo_users[user_id] = {'id': user_id, 'email': email}
        
        # Create session
        set_session_user(user_id)
        
        return jsonify({
            'success': True,
            'action': 'login',
            'message': 'Logged in successfully!',
            'user': {
                'id': user_id,
                'email': email,
                'created_at': '2026-04-28'
            }
        }), 200
    
    except Exception as e:
        print(f"Authentication error: {e}")
        return jsonify({'error': 'An error occurred during authentication'}), 500


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    Logout the current user by clearing the session.
    
    Returns:
        200: Success message
    """
    clear_session()
    return jsonify({
        'success': True,
        'message': 'Logged out successfully'
    }), 200


@auth_bp.route('/me', methods=['GET'])
def get_me():
    """
    ✅ DEMO MODE: Get current logged-in user from session.
    No database access needed.
    
    Returns:
        200: User data if authenticated
        401: Not authenticated
    """
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({
            'authenticated': False,
            'message': 'Not logged in'
        }), 401
    
    # Get user from in-memory storage
    user = demo_users.get(user_id)
    
    if not user:
        return jsonify({
            'authenticated': False,
            'message': 'User not found'
        }), 401
    
    return jsonify({
        'authenticated': True,
        'user': {
            'id': user['id'],
            'email': user['email'],
            'created_at': '2026-04-28'
        }
    }), 200


@auth_bp.route('/check-auth', methods=['GET'])
def check_auth():
    """
    ✅ DEMO MODE: Check if user is authenticated (from session).
    
    Returns:
        200: Authentication status
    """
    user_id = session.get('user_id')
    
    if user_id:
        user = demo_users.get(user_id)
        if user:
            return jsonify({
                'authenticated': True,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'created_at': '2026-04-28'
                }
            }), 200
    
    return jsonify({
        'authenticated': False,
        'user': None
    }), 200

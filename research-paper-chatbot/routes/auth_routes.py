"""
Authentication Routes
Handles user login, signup, logout, and user info retrieval.
"""

from flask import Blueprint, request, jsonify, session
from utils.auth import (
    get_current_user,
    get_user_by_email,
    create_user,
    verify_password,
    set_session_user,
    clear_session,
    validate_email,
    validate_password,
    login_required
)

# Create a Blueprint for auth routes
auth_bp = Blueprint('auth', __name__, url_prefix='')


@auth_bp.route('/auth', methods=['POST'])
def authenticate():
    """
    Single endpoint for both login and signup.
    
    Request Body:
        {
            "email": "user@example.com",
            "password": "password123"
        }
    
    Logic:
        1. Check if email exists in database
        2. If exists → verify password (login)
        3. If not exists → create new user (signup)
    
    Returns:
        200: Success with user data
        400: Validation error
        401: Invalid credentials
        500: Server error
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
        
        # Check if user exists
        existing_user = get_user_by_email(email)
        
        if existing_user:
            # LOGIN flow - user exists, verify password
            if verify_password(existing_user['password_hash'], password):
                # Password is correct - create session
                set_session_user(existing_user['id'])
                
                return jsonify({
                    'success': True,
                    'action': 'login',
                    'message': 'Login successful',
                    'user': {
                        'id': existing_user['id'],
                        'email': existing_user['email'],
                        'created_at': existing_user['created_at'].isoformat()
                    }
                }), 200
            else:
                # Password is incorrect
                return jsonify({'error': 'Invalid email or password'}), 401
        
        else:
            # SIGNUP flow - user doesn't exist, create new user
            new_user = create_user(email, password)
            
            if new_user:
                # User created successfully - create session
                set_session_user(new_user['id'])
                
                return jsonify({
                    'success': True,
                    'action': 'signup',
                    'message': 'Account created successfully',
                    'user': {
                        'id': new_user['id'],
                        'email': new_user['email'],
                        'created_at': new_user['created_at'].isoformat()
                    }
                }), 200
            else:
                return jsonify({'error': 'Failed to create account. Please try again.'}), 500
    
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
@login_required
def get_me():
    """
    Get the currently logged-in user's information.
    Protected route - requires authentication.
    
    Returns:
        200: User data
        401: Not authenticated (handled by @login_required decorator)
    """
    user = get_current_user()
    
    return jsonify({
        'authenticated': True,
        'user': {
            'id': user['id'],
            'email': user['email'],
            'created_at': user['created_at'].isoformat()
        }
    }), 200


@auth_bp.route('/check-auth', methods=['GET'])
def check_auth():
    """
    Check if a user is currently authenticated.
    Non-protected version of /me for frontend to check auth state.
    
    Returns:
        200: Authentication status
    """
    user = get_current_user()
    
    if user:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'created_at': user['created_at'].isoformat()
            }
        }), 200
    else:
        return jsonify({
            'authenticated': False,
            'user': None
        }), 200

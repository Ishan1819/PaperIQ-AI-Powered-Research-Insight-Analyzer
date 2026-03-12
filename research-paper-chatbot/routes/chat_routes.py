"""
Chat History Routes
Handles retrieval and storage of user chat history.
"""

from flask import Blueprint, jsonify
from utils.auth import get_current_user, login_required
from database.db import get_db_cursor

# Create a Blueprint for chat routes
chat_bp = Blueprint('chat', __name__, url_prefix='/chat')


@chat_bp.route('/history', methods=['GET'])
@login_required
def get_chat_history():
    """
    Retrieve the last 20 chat messages for the logged-in user.
    Protected route - requires authentication.
    
    Returns:
        200: List of chat messages ordered by created_at (newest first)
        401: Not authenticated (handled by @login_required decorator)
        500: Server error
    
    Response format:
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
                },
                ...
            ]
        }
    """
    try:
        # Get current logged-in user
        user = get_current_user()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        user_id = user['id']
        
        # Fetch last 20 messages from chat_history table
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, user_id, user_message, ai_response, created_at
                FROM chat_history
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (user_id,)
            )
            messages = cursor.fetchall()
        
        # Convert datetime to ISO format for JSON serialization
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'id': msg['id'],
                'user_id': msg['user_id'],
                'user_message': msg['user_message'],
                'ai_response': msg['ai_response'],
                'created_at': msg['created_at'].isoformat()
            })
        
        return jsonify({
            'success': True,
            'count': len(formatted_messages),
            'messages': formatted_messages
        }), 200
    
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500


@chat_bp.route('/save', methods=['POST'])
@login_required
def save_chat_message():
    """
    Save a chat message to the database.
    This endpoint can be called after each Q&A interaction.
    
    Request Body:
        {
            "user_message": "What is the main finding?",
            "ai_response": "The main finding is..."
        }
    
    Returns:
        201: Message saved successfully
        400: Invalid request
        401: Not authenticated
        500: Server error
    """
    from flask import request
    
    try:
        user = get_current_user()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        user_message = data.get('user_message', '').strip()
        ai_response = data.get('ai_response', '').strip()
        
        if not user_message or not ai_response:
            return jsonify({'error': 'Both user_message and ai_response are required'}), 400
        
        # Insert into chat_history table
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_history (user_id, user_message, ai_response)
                VALUES (%s, %s, %s)
                RETURNING id, created_at
                """,
                (user['id'], user_message, ai_response)
            )
            result = cursor.fetchone()
        
        return jsonify({
            'success': True,
            'message': 'Chat message saved',
            'id': result['id'],
            'created_at': result['created_at'].isoformat()
        }), 201
    
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return jsonify({'error': 'Failed to save chat message'}), 500


@chat_bp.route('/clear', methods=['DELETE'])
@login_required
def clear_chat_history():
    """
    Delete all chat history for the current user.
    
    Returns:
        200: Success message
        401: Not authenticated
        500: Server error
    """
    try:
        user = get_current_user()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        # Delete all messages for this user
        with get_db_cursor() as cursor:
            cursor.execute(
                "DELETE FROM chat_history WHERE user_id = %s",
                (user['id'],)
            )
            deleted_count = cursor.rowcount
        
        return jsonify({
            'success': True,
            'message': f'Deleted {deleted_count} messages',
            'deleted_count': deleted_count
        }), 200
    
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        return jsonify({'error': 'Failed to clear chat history'}), 500

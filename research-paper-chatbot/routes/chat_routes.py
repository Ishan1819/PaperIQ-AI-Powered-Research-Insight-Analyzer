"""
Chat History Routes
Handles retrieval and storage of user chat history.
DEMO MODE: Session-only - no database required.
"""

from flask import Blueprint, jsonify, session
from utils.auth import get_current_user, login_required

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
        
        # Demo mode: Get messages from session storage
        user_id = user['id']
        chat_history = session.get(f'chat_history_{user_id}', [])
        
        # Return last 20 messages
        messages = sorted(chat_history, key=lambda x: x.get('created_at', ''), reverse=True)[:20]
        
        return jsonify({
            'success': True,
            'count': len(messages),
            'messages': messages
        }), 200
    
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500


@chat_bp.route('/save', methods=['POST'])
@login_required
def save_chat_message():
    """
    Save a chat message to session storage.
    Demo mode - no database required.
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
    from datetime import datetime
    
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
        
        # Demo mode: Store in session storage
        user_id = user['id']
        chat_history_key = f'chat_history_{user_id}'
        chat_history = session.get(chat_history_key, [])
        
        # Create message object
        message_id = len(chat_history) + 1
        new_message = {
            'id': message_id,
            'user_id': user_id,
            'user_message': user_message,
            'ai_response': ai_response,
            'created_at': datetime.now().isoformat()
        }
        
        # Add to history
        chat_history.append(new_message)
        session[chat_history_key] = chat_history
        
        return jsonify({
            'success': True,
            'message': 'Chat message saved',
            'id': message_id,
            'created_at': new_message['created_at']
        }), 201
    
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return jsonify({'error': 'Failed to save chat message'}), 500


@chat_bp.route('/clear', methods=['DELETE'])
@login_required
def clear_chat_history():
    """
    Delete all chat history for the current user.
    Demo mode - using session storage.
    
    Returns:
        200: Success message
        401: Not authenticated
        500: Server error
    """
    try:
        user = get_current_user()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        # Demo mode: Clear from session storage
        user_id = user['id']
        chat_history_key = f'chat_history_{user_id}'
        deleted_count = len(session.get(chat_history_key, []))
        session[chat_history_key] = []
        
        return jsonify({
            'success': True,
            'message': f'Deleted {deleted_count} messages',
            'deleted_count': deleted_count
        }), 200
    
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        return jsonify({'error': 'Failed to clear chat history'}), 500

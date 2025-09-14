# Add these new imports and endpoints to your existing app.py file

from chat_assistant import ABTestChatAssistant
import uuid
from datetime import datetime

# Store chat sessions (in production, use Redis or database)
chat_sessions = {}

@app.route('/api/chat/start', methods=['POST'])
def start_chat_session():
    """Start a new chat session"""
    try:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = ABTestChatAssistant()
        
        # Get initial greeting
        response = chat_sessions[session_id].process_message("Hello")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': response['message'],
            'suggestions': response.get('suggestions', []),
            'state': response.get('state', 'greeting')
        })
        
    except Exception as e:
        logger.error(f"Error starting chat session: {e}")
        return jsonify({'error': 'Failed to start chat session'}), 500

@app.route('/api/chat/<session_id>/message', methods=['POST'])
def send_chat_message(session_id):
    """Send a message to existing chat session"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
            
        if session_id not in chat_sessions:
            return jsonify({'error': 'Chat session not found'}), 404
            
        user_message = str(data['message']).strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Process message through chat assistant
        chat_assistant = chat_sessions[session_id]
        response = chat_assistant.process_message(user_message)
        
        # Check if analysis is ready
        analysis_ready = response.get('analysis_ready', False)
        parameters = None
        
        if analysis_ready:
            parameters = chat_assistant.export_parameters_for_api()
        
        return jsonify({
            'success': True,
            'message': response['message'],
            'suggestions': response.get('suggestions', []),
            'state': response.get('state', 'active'),
            'analysis_ready': analysis_ready,
            'parameters': parameters,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return jsonify({'error': 'Failed to process message'}), 500

@app.route('/api/chat/<session_id>/parameters', methods=['GET'])
def get_chat_parameters(session_id):
    """Get collected parameters from chat session"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Chat session not found'}), 404
            
        chat_assistant = chat_sessions[session_id]
        parameters = chat_assistant.export_parameters_for_api()
        summary = chat_assistant.get_conversation_summary()
        
        return jsonify({
            'success': True,
            'parameters': parameters,
            'summary': summary,
            'ready_for_analysis': summary['ready_for_analysis']
        })
        
    except Exception as e:
        logger.error(f"Error getting chat parameters: {e}")
        return jsonify({'error': 'Failed to get parameters'}), 500

@app.route('/api/chat/<session_id>/reset', methods=['POST'])
def reset_chat_session(session_id):
    """Reset chat session to start over"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Chat session not found'}), 404
            
        chat_sessions[session_id].reset_conversation()
        
        # Get fresh greeting
        response = chat_sessions[session_id].process_message("Hello")
        
        return jsonify({
            'success': True,
            'message': response['message'],
            'suggestions': response.get('suggestions', []),
            'state': 'greeting'
        })
        
    except Exception as e:
        logger.error(f"Error resetting chat session: {e}")
        return jsonify({'error': 'Failed to reset session'}), 500

@app.route('/api/chat/<session_id>/history', methods=['GET'])
def get_chat_history(session_id):
    """Get conversation history"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Chat session not found'}), 404
            
        chat_assistant = chat_sessions[session_id]
        history = chat_assistant.context.conversation_history
        
        return jsonify({
            'success': True,
            'history': history,
            'message_count': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({'error': 'Failed to get history'}), 500

# Cleanup old chat sessions (run periodically)
@app.route('/api/chat/cleanup', methods=['POST'])
def cleanup_chat_sessions():
    """Clean up old chat sessions (admin endpoint)"""
    try:
        # In production, implement proper session cleanup based on timestamp
        # For now, keep only the last 100 sessions
        if len(chat_sessions) > 100:
            oldest_sessions = list(chat_sessions.keys())[:-100]
            for session_id in oldest_sessions:
                del chat_sessions[session_id]
        
        return jsonify({
            'success': True,
            'active_sessions': len(chat_sessions),
            'message': 'Cleanup completed'
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up chat sessions: {e}")
        return jsonify({'error': 'Failed to cleanup sessions'}), 500
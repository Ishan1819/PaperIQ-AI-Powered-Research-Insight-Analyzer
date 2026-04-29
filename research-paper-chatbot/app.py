from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from werkzeug.utils import secure_filename
from backend import (
    extract_text_from_pdf, 
    extract_text_from_docx, 
    extract_sections_precise,
    simple_summarize,
    retrieve_paragraph_answer
)

# ✅ Re-enabled Auth imports - needed for frontend endpoints
from routes.auth_routes import auth_bp
from routes.chat_routes import chat_bp

app = Flask(__name__)

app.secret_key = os.getenv('SECRET_KEY', 'dev-key')

# Session config
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600

# Upload config
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Re-enabled blueprint registration - /me and /check-auth endpoints
app.register_blueprint(auth_bp)
app.register_blueprint(chat_bp)

# ❌ Removed DB initialization
# init_db_pool(minconn=2, maxconn=10)

ALLOWED_EXTENSIONS = {'pdf', 'docx'}

document_storage = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ✅ Bypass login (always True)
def require_login():
    return True


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if filename.endswith('.pdf'):
                full_text, pages_data = extract_text_from_pdf(filepath)
            elif filename.endswith('.docx'):
                full_text, pages_data = extract_text_from_docx(filepath)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
            
            sections = extract_sections_precise(pages_data)
            
            doc_id = 'current_document'
            document_storage[doc_id] = {
                'full_text': full_text,
                'pages_data': pages_data,
                'sections': sections,
                'filename': filename
            }
            
            session['doc_id'] = doc_id
            session['has_document'] = True
            
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/summary')
def summary_page():
    return render_template('summary.html')

# ✅ LOGIN REDIRECT REMOVED - prevents infinite redirect loop
# Removed: return redirect(url_for('landing'))
# Frontend login.html handles its own checkAuth() to redirect if needed
# This endpoint now serves the login page directly
@app.route('/login')
def login_page():
    try:
        return render_template('login.html')
    except Exception as e:
        return jsonify({'error': 'Failed to load login page'}), 500


@app.route('/generate-summary', methods=['POST'])
def generate_summary():

    if not session.get('has_document'):
        return jsonify({'error': 'No document uploaded'}), 400
    
    try:
        doc_id = session.get('doc_id')
        
        data = request.get_json()
        ratio = data.get('ratio', 0.3)
        
        full_text = document_storage[doc_id]['full_text']
        summary = simple_summarize(full_text, summary_ratio=ratio)
        
        return jsonify({'summary': summary})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat')
def chat_page():
    doc_id = session.get('doc_id')
    filename = document_storage.get(doc_id, {}).get('filename', 'Document')
    return render_template('chat.html', filename=filename)


@app.route('/ask', methods=['POST'])
def ask_question():

    if not session.get('has_document'):
        return jsonify({'error': 'No document uploaded'}), 400
    
    try:
        doc_id = session.get('doc_id')
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        doc_data = document_storage[doc_id]
        
        answer = retrieve_paragraph_answer(
            question,
            doc_data['sections'],
            doc_data['full_text']
        )
        
        return jsonify({'answer': answer})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear-document', methods=['POST'])
def clear_document():
    session.clear()
    return jsonify({'success': True})


# ❌ Removed DB test
# @app.before_request
# def before_first_request():
#     if not hasattr(app, 'db_tested'):
#         test_connection()
#         app.db_tested = True


if __name__ == '__main__':
    # ✅ FIXED: debug=False for production on EB
    # debug=True causes auto-reload and infinite refresh loops in production
    # Use environment variable to control debug mode safely
    debug_mode = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)

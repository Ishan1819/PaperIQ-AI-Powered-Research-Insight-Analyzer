from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from werkzeug.utils import secure_filename
from backend import (
    extract_text_from_pdf, 
    extract_text_from_docx, 
    extract_sections,
    simple_summarize,
    retrieve_paragraph_answer
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Global storage for document data (alternative to session for large data)
document_storage = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
                text = extract_text_from_pdf(filepath)
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(filepath)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
            
            sections = extract_sections(text)
            
            # Store in global dictionary instead of session
            doc_id = 'current_document'
            document_storage[doc_id] = {
                'full_text': text,
                'sections': sections,
                'filename': filename
            }
            
            # Store only the document ID in session
            session['doc_id'] = doc_id
            session['has_document'] = True
            session.modified = True
            
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'message': 'Document processed successfully',
                'filename': filename
            })
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Only PDF and DOCX are allowed.'}), 400


@app.route('/summary')
def summary_page():
    if not session.get('has_document'):
        return redirect(url_for('upload_page'))
    return render_template('summary.html')


@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    if not session.get('has_document'):
        return jsonify({'error': 'No document uploaded'}), 400
    
    try:
        doc_id = session.get('doc_id')
        if doc_id not in document_storage:
            return jsonify({'error': 'Document not found. Please upload again.'}), 400
        
        data = request.get_json()
        ratio = data.get('ratio', 0.3)
        
        full_text = document_storage[doc_id]['full_text']
        summary = simple_summarize(full_text, summary_ratio=ratio)
        
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': f'Error generating summary: {str(e)}'}), 500


@app.route('/chat')
def chat_page():
    if not session.get('has_document'):
        return redirect(url_for('upload_page'))
    
    doc_id = session.get('doc_id')
    filename = document_storage.get(doc_id, {}).get('filename', 'Document')
    
    return render_template('chat.html', filename=filename)


@app.route('/ask', methods=['POST'])
def ask_question():
    if not session.get('has_document'):
        return jsonify({'error': 'No document uploaded'}), 400
    
    try:
        doc_id = session.get('doc_id')
        if doc_id not in document_storage:
            return jsonify({'error': 'Document not found. Please upload again.'}), 400
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        doc_data = document_storage[doc_id]
        answer = retrieve_paragraph_answer(
            question, 
            doc_data['sections'], 
            doc_data['full_text']
        )
        
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500


@app.route('/clear-document', methods=['POST'])
def clear_document():
    """Clear current document and allow new upload"""
    doc_id = session.get('doc_id')
    if doc_id and doc_id in document_storage:
        del document_storage[doc_id]
    
    session.clear()
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(debug=True)
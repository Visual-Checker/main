from flask import Flask, redirect, url_for
from dotenv import load_dotenv
import os
import logfire

load_dotenv()

logfire.configure(send_to_logfire=False)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# Register blueprints
from routes.admin import admin_bp
app.register_blueprint(admin_bp)

# Redirect root to admin UI
@app.route('/')
def root():
    return redirect(url_for('admin.index'))

# Health endpoint
@app.route('/health')
def health():
    return {'status': 'ok'}

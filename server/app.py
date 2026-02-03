from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# Register blueprints
from routes.admin import admin_bp
app.register_blueprint(admin_bp)

# Health endpoint
@app.route('/health')
def health():
    return {'status': 'ok'}

from flask import Blueprint

bp = Blueprint('courses', __name__, url_prefix='/api/courses')

from routes.courses import routes

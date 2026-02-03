from flask import Blueprint

bp = Blueprint('attendance', __name__, url_prefix='/api/attendance')

from routes.attendance import routes

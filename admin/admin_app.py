"""
Admin Desktop App (PyQt5)
- Login (admin/admin123)
- Dashboard: Redis + Postgres stats (total students, attendance events, table counts)
- Config via env vars: POSTGRES_HOST/POSTGRES_PORT/POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD, REDIS_URL
"""
import os
import sys
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QTimer

load_dotenv()

# Use structured services (SQLAlchemy + redis wrapper)
from structure import services


class LoginWindow(QWidget):
    def __init__(self, on_success):
        super().__init__()
        self.on_success = on_success
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Admin Login')
        layout = QVBoxLayout()

        self.user_in = QLineEdit()
        self.user_in.setPlaceholderText('Username')
        layout.addWidget(self.user_in)

        self.pw_in = QLineEdit()
        self.pw_in.setPlaceholderText('Password')
        self.pw_in.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.pw_in)

        btn = QPushButton('Login')
        btn.clicked.connect(self.try_login)
        layout.addWidget(btn)

        self.setLayout(layout)

    def try_login(self):
        u = self.user_in.text()
        p = self.pw_in.text()
        if u == 'admin' and p == 'admin123':
            self.on_success(u)
            self.close()
        else:
            QMessageBox.warning(self, 'Login failed', 'Invalid credentials')


class DashboardWindow(QMainWindow):
    def __init__(self, user):
        super().__init__()
        self.user = user
        self.redis = get_redis_client()
        self.init_ui()
        # Periodic refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_stats)
        self.timer.start(5000)

    def init_ui(self):
        self.setWindowTitle('Admin Dashboard')
        main = QWidget()
        layout = QVBoxLayout()

        header = QLabel(f'Logged in as: {self.user} (Admin)')
        header.setAlignment(Qt.AlignLeft)
        layout.addWidget(header)

        # Stats frame
        self.table_count_lbl = QLabel('Tables: -')
        self.students_count_lbl = QLabel('Students table rows: -')
        self.attendance_lbl = QLabel('Attendance events (Redis): -')

        layout.addWidget(self.table_count_lbl)
        layout.addWidget(self.students_count_lbl)
        layout.addWidget(self.attendance_lbl)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton('Refresh')
        refresh_btn.clicked.connect(self.refresh_stats)
        btn_layout.addWidget(refresh_btn)

        manage_btn = QPushButton('Manage Students')
        manage_btn.clicked.connect(self.open_students)
        btn_layout.addWidget(manage_btn)

        logout_btn = QPushButton('Logout')
        logout_btn.clicked.connect(self.logout)
        btn_layout.addWidget(logout_btn)

        layout.addLayout(btn_layout)

        main.setLayout(layout)
        self.setCentralWidget(main)

        self.resize(480, 240)
        self.refresh_stats()

    def refresh_stats(self):
        # Use service layer to fetch stats
        tables = services.get_table_count()
        students = services.get_students_count()
        attendance = services.get_attendance_events_count()

        # Fallbacks for display
        tables_display = tables if tables is not None else 'err'
        students_display = students if students is not None else 'err'
        attendance_display = attendance if attendance is not None else 'err'

        self.table_count_lbl.setText(f'Tables: {tables_display}')
        self.students_count_lbl.setText(f'Students table rows: {students_display}')
        self.attendance_lbl.setText(f'Attendance events (Redis): {attendance_display}')

    def logout(self):
        QMessageBox.information(self, 'Logout', 'You have been logged out')
        self.close()
        # Relaunch login
        self.login = LoginWindow(self.launch_dashboard)
        self.login.show()

    def open_students(self):
        # Lazy import to avoid overhead
        from ui_students import StudentsWindow
        self.students_win = StudentsWindow(self)
        self.students_win.show()

    def launch_dashboard(self, user):
        self.__init__(user)
        self.show()


def main():
    app = QApplication(sys.argv)
    # Start with login
    def on_success(user):
        dash = DashboardWindow(user)
        dash.show()
        app.exec_()

    login = LoginWindow(on_success)
    login.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

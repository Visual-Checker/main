from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt
from structure import services


class StudentsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Students Management')
        self.init_ui()
        self.refresh()

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls
        ctrl_layout = QHBoxLayout()
        self.sid_in = QLineEdit()
        self.sid_in.setPlaceholderText('Student ID')
        self.name_in = QLineEdit()
        self.name_in.setPlaceholderText('Name')
        add_btn = QPushButton('Add')
        add_btn.clicked.connect(self.add_student)
        refresh_btn = QPushButton('Refresh')
        refresh_btn.clicked.connect(self.refresh)

        ctrl_layout.addWidget(self.sid_in)
        ctrl_layout.addWidget(self.name_in)
        ctrl_layout.addWidget(add_btn)
        ctrl_layout.addWidget(refresh_btn)

        layout.addLayout(ctrl_layout)

        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['DB id', 'Student ID', 'Name'])
        self.table.setSelectionBehavior(self.table.SelectRows)
        layout.addWidget(self.table)

        # Buttons: Delete / Enroll
        btn_row = QHBoxLayout()

        del_btn = QPushButton('Delete Selected')
        del_btn.clicked.connect(self.delete_selected)
        btn_row.addWidget(del_btn)

        enroll_face_btn = QPushButton('Enroll Face (from file)')
        enroll_face_btn.clicked.connect(self.enroll_face_file)
        btn_row.addWidget(enroll_face_btn)

        enroll_gesture_btn = QPushButton('Enroll Gesture (from file)')
        enroll_gesture_btn.clicked.connect(self.enroll_gesture_file)
        btn_row.addWidget(enroll_gesture_btn)

        enroll_voice_btn = QPushButton('Enroll Voice (WAV)')
        enroll_voice_btn.clicked.connect(self.enroll_voice_file)
        btn_row.addWidget(enroll_voice_btn)

        layout.addLayout(btn_row)

        self.setLayout(layout)
        self.resize(600, 400)

    def refresh(self):
        students = services.list_students(limit=500)
        self.table.setRowCount(0)
        for s in students:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(s['id'])))
            self.table.setItem(r, 1, QTableWidgetItem(s['student_id']))
            self.table.setItem(r, 2, QTableWidgetItem(s['name']))

    def add_student(self):
        sid = self.sid_in.text().strip()
        name = self.name_in.text().strip()
        if not sid or not name:
            QMessageBox.warning(self, 'Input error', 'student id and name required')
            return
        ok = services.add_student(sid, name)
        if ok:
            QMessageBox.information(self, 'Added', 'Student added')
            self.sid_in.clear(); self.name_in.clear(); self.refresh()
        else:
            QMessageBox.warning(self, 'Failed', 'Could not add student (maybe duplicate)')

    def delete_selected(self):
        sel = self.table.selectedItems()
        if not sel:
            QMessageBox.warning(self, 'Select', 'No row selected')
            return
        row = sel[0].row()
        student_id = self.table.item(row, 1).text()
        ok = services.delete_student_by_student_id(student_id)
        if ok:
            QMessageBox.information(self, 'Deleted', f'{student_id} deleted')
            self.refresh()
        else:
            QMessageBox.warning(self, 'Failed', 'Delete failed')

    def enroll_face_file(self):
        sel = self.table.selectedItems()
        if not sel:
            QMessageBox.warning(self, 'Select', 'No row selected')
            return
        student_id = sel[0].text().split(' - ')[0] if ' - ' in sel[0].text() else self.table.item(sel[0].row(),1).text()
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select face image', '', 'Images (*.png *.jpg *.jpeg)')
        if not file_path:
            return
        ok, err = services.enroll_face_from_file(student_id, file_path)
        if ok:
            QMessageBox.information(self, 'Enroll', 'Face enrolled successfully')
            self.refresh()
        else:
            QMessageBox.warning(self, 'Enroll failed', f'Failed: {err}')

    def enroll_gesture_file(self):
        sel = self.table.selectedItems()
        if not sel:
            QMessageBox.warning(self, 'Select', 'No row selected')
            return
        student_id = sel[0].text().split(' - ')[0] if ' - ' in sel[0].text() else self.table.item(sel[0].row(),1).text()
        from PyQt5.QtWidgets import QFileDialog, QInputDialog
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select gesture image', '', 'Images (*.png *.jpg *.jpeg)')
        if not file_path:
            return
        gesture_type, ok_g = QInputDialog.getText(self, 'Gesture Type', 'Enter gesture label (optional)')
        if not ok_g:
            gesture_type = 'unknown'
        ok, err = services.enroll_gesture_from_file(student_id, file_path, gesture_type)
        if ok:
            QMessageBox.information(self, 'Enroll', 'Gesture enrolled successfully')
            self.refresh()
        else:
            QMessageBox.warning(self, 'Enroll failed', f'Failed: {err}')

    def enroll_voice_file(self):
        sel = self.table.selectedItems()
        if not sel:
            QMessageBox.warning(self, 'Select', 'No row selected')
            return
        student_id = sel[0].text().split(' - ')[0] if ' - ' in sel[0].text() else self.table.item(sel[0].row(),1).text()
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select WAV file', '', 'WAV files (*.wav)')
        if not file_path:
            return
        ok, err = services.enroll_voice_from_file(student_id, file_path)
        if ok:
            QMessageBox.information(self, 'Enroll', 'Voice enrolled successfully')
            self.refresh()
        else:
            QMessageBox.warning(self, 'Enroll failed', f'Failed: {err}')

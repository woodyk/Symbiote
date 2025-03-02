#!/usr/bin/env python3
#
# gui.py

import sys
from PyQt5.QtWidgets import QApplication, QPlainTextEdit
from PyQt5.QtCore import QProcess

class Terminal(QPlainTextEdit):
    def __init__(self, parent=None):
        super(Terminal, self).__init__(parent)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)

    def execute(self, command):
        self.process.start(command)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(data)
        self.setTextCursor(cursor)

    def handle_stderr(self):
        error = self.process.readAllStandardError().data().decode()
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(error)
        self.setTextCursor(cursor)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    terminal = Terminal()
    terminal.show()
    terminal.execute('/opt/homebrew/bin/symbiote')
    sys.exit(app.exec_())


#!/usr/bin/env python3
"""Entry point for PaddleOCR Label Studio.

Run with:
    python main.py [optional/path/to/image/folder]
"""
import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PaddleOCR Label Studio")

    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        window.load_folder(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

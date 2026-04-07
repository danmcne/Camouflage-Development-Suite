"""
Camouflage Development Tool – entry point.

Run with:
    python main.py

Requires:  PyQt6, numpy, opencv-python, Pillow, scikit-learn, noise, pyswarms
Install:   pip install -r requirements.txt
           (or with venv:  .venv/bin/pip install -r requirements.txt)
"""
import sys
import os

# Ensure the project root is on sys.path so sub-packages import cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from ui.main_window import MainWindow


def main():
    # High-DPI support (Qt6 handles this automatically, but explicit is safer)
    app = QApplication(sys.argv)
    app.setApplicationName("Camouflage Dev Tool")
    app.setApplicationVersion("0.1")
    app.setOrganizationName("CamoDev")

    # Dark palette (optional – comment out for system theme)
    _apply_dark_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


def _apply_dark_theme(app: QApplication):
    """Apply a simple dark colour palette to the whole app."""
    from PyQt6.QtGui import QPalette, QColor
    from PyQt6.QtCore import Qt

    dark = QPalette()
    base   = QColor(30, 30, 30)
    alt    = QColor(42, 42, 42)
    text   = QColor(220, 220, 220)
    bright = QColor(255, 255, 255)
    mid    = QColor(80, 80, 80)
    hl     = QColor(46, 107, 46)    # green highlight matching the buttons
    hl_txt = QColor(255, 255, 255)
    btn    = QColor(55, 55, 55)
    btn_txt= QColor(220, 220, 220)

    dark.setColor(QPalette.ColorRole.Window,          base)
    dark.setColor(QPalette.ColorRole.WindowText,      text)
    dark.setColor(QPalette.ColorRole.Base,            alt)
    dark.setColor(QPalette.ColorRole.AlternateBase,   base)
    dark.setColor(QPalette.ColorRole.ToolTipBase,     bright)
    dark.setColor(QPalette.ColorRole.ToolTipText,     base)
    dark.setColor(QPalette.ColorRole.Text,            text)
    dark.setColor(QPalette.ColorRole.Button,          btn)
    dark.setColor(QPalette.ColorRole.ButtonText,      btn_txt)
    dark.setColor(QPalette.ColorRole.BrightText,      bright)
    dark.setColor(QPalette.ColorRole.Highlight,       hl)
    dark.setColor(QPalette.ColorRole.HighlightedText, hl_txt)
    dark.setColor(QPalette.ColorRole.Link,            QColor(80, 160, 255))
    dark.setColor(QPalette.ColorRole.Mid,             mid)

    app.setPalette(dark)
    app.setStyle("Fusion")


if __name__ == "__main__":
    main()

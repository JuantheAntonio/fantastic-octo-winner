from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract
from tkinter import filedialog, Tk
from docx import Document
import json
import os

pytesseract.pytesseract.tesseract_cmd = r"D:\\tesseract-OCR\\tesseract.exe"

class ImagePanel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)  # Fixed image panel size
        self.setAlignment(Qt.AlignCenter)

    def set_image(self, pixmap):
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tab_count = 1  # Start counting from 1
        
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBarDoubleClicked.connect(self.tab_open_doubleclick)
        self.tabs.currentChanged.connect(self.current_tab_changed)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_current_tab)

        self.setCentralWidget(self.tabs)
        self.add_new_tab()
        self.show()

    def add_new_tab(self):
        tab_name = f"Tab {self.tab_count}"
        new_tab = QWidget()
        self.tab_count += 1
        layout = QHBoxLayout()

        # Left side: Image display and buttons
        left_layout = QVBoxLayout()
        self.image_label = ImagePanel()
        left_layout.addWidget(self.image_label, 1)

        # Right side: Text area and convert button
        text_layout = QVBoxLayout()
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        text_layout.addWidget(self.text_area)

        layout.addLayout(left_layout, 1)
        layout.addLayout(text_layout, 1)
        new_tab.setLayout(layout)
        i = self.tabs.addTab(new_tab, tab_name)
        self.tabs.setCurrentIndex(i)

        # Add toolbar actions
        self.left_toolbar = QToolBar("Tools", self)
        self.addToolBar(Qt.LeftToolBarArea, self.left_toolbar)

        action_upload = QAction(QIcon(r"C:\Users\HP\OneDrive\Desktop\visualstudio\python series\IMAGE TO TEXT REE\FINAL\icons\upload.png"), "Upload Image", self)
        action_upload.setToolTip("Upload Image")
        action_upload.triggered.connect(lambda: self.extract_text(new_tab))
        self.left_toolbar.addAction(action_upload)

        action_paste = QAction(QIcon(r"C:\Users\HP\OneDrive\Desktop\visualstudio\python series\IMAGE TO TEXT REE\FINAL\icons\clipboard.png"), "Paste Image", self)
        action_paste.setToolTip("Paste Image from Clipboard")
        action_paste.triggered.connect(lambda: self.paste_image(new_tab))
        self.left_toolbar.addAction(action_paste)

        action_save_text = QAction(QIcon(r"C:\Users\HP\OneDrive\Desktop\visualstudio\python series\IMAGE TO TEXT REE\FINAL\icons\save text.png"), "Save Text", self)
        action_save_text.setToolTip("Save Text")
        action_save_text.triggered.connect(lambda: self.save_file(new_tab))
        self.left_toolbar.addAction(action_save_text)

        action_export_ai = QAction(QIcon(r"C:\Users\HP\OneDrive\Desktop\visualstudio\python series\IMAGE TO TEXT REE\FINAL\icons\export.png"), "Export for AI", self)
        action_export_ai.setToolTip("Export for AI")
        action_export_ai.triggered.connect(lambda: self.save_as_json(new_tab))
        self.left_toolbar.addAction(action_export_ai)

    def save_as_json(self, tab):
        for widget in tab.children():
            if isinstance(widget, QTextEdit):
                text = widget.toPlainText()
                if text:
                    folder_path = "output_json"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    
                    index = 1
                    while True:
                        file_path = os.path.join(folder_path, f"json{index}.json")
                        if not os.path.exists(file_path):
                            break
                        index += 1
                    
                    data = {"text": text}
                    with open(file_path, "w", encoding="utf-8") as json_file:
                        json.dump(data, json_file, ensure_ascii=False, indent=4)
                    
                    QMessageBox.information(self, "Success", f"Text saved to {file_path}")

    def extract_text(self, tab, image=None):
        if image is None:
            root = Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
            if not file_path:
                return

            image = cv2.imread(file_path)
            if image is None:
                return

        image = self.crop_and_deskew_image(image)
        if image is None:
            return

        self.display_image(image, tab)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(Image.fromarray(gray), lang="eng")

        for widget in tab.children():
            if isinstance(widget, QTextEdit):
                widget.setPlainText(text)

    def save_file(self, tab):
        for widget in tab.children():
            if isinstance(widget, QTextEdit):
                text = widget.toPlainText()
                if text:
                    options = QFileDialog.Options()
                    file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;Word Documents (*.docx);;All Files (*)", options=options)
                    if file_path:
                        if file_path.endswith(".docx"):
                            doc = Document()
                            doc.add_paragraph(text)
                            doc.save(file_path)
                        else:
                            with open(file_path, "w", encoding="utf-8") as file:
                                file.write(text)

    def paste_image(self, tab):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        if mime_data.hasImage():
            image = clipboard.image()
            qpixmap = QPixmap.fromImage(image)
            for widget in tab.children():
                if isinstance(widget, ImagePanel):
                    widget.set_image(qpixmap)
            
            buffer = QBuffer()
            buffer.open(QIODevice.ReadWrite)
            qpixmap.save(buffer, "PNG")
            np_image = np.array(Image.open(buffer))
            self.extract_text(tab, np_image)

    def qimage_to_cv2(self, qimage):
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width, height = qimage.width(), qimage.height()
        bytes_per_line = 3 * width
        ptr = qimage.bits()
        ptr.setsize(height * bytes_per_line)
        return np.array(ptr, dtype=np.uint8).reshape((height, width, 3))

    def display_image(self, image, tab):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        for widget in tab.children():
            if isinstance(widget, ImagePanel):
                widget.set_image(pixmap)
                
    def crop_and_deskew_image(self, image):
        try:
            height, width = image.shape[:2]
            max_dim = 800
            scale = min(max_dim / width, max_dim / height, 1.0)
            resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))

            roi = cv2.selectROI("Crop Image", resized_image, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            if roi == (0, 0, 0, 0):
                return None

            x, y, w, h = map(int, [roi[0] / scale, roi[1] / scale, roi[2] / scale, roi[3] / scale])
            cropped_image = image[y:y+h, x:x+w]

            angle = 0
            while True:
                rotated_image = self.rotate_bound(cropped_image, angle)
                cv2.imshow("Deskew Image - Press Q/E to Rotate, Enter to Confirm", rotated_image)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    angle -= 1
                elif key == ord('e'):
                    angle += 1
                elif key == 13:
                    cv2.destroyAllWindows()
                    return rotated_image
                elif key == 27:
                    cv2.destroyAllWindows()
                    return None
        except Exception as e:
            return None

    def rotate_bound(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    def tab_open_doubleclick(self, i):
        if i == -1:
            self.add_new_tab()

    def current_tab_changed(self, i):
        pass

    def close_current_tab(self, i):
        if self.tabs.count() > 1:
            self.tabs.removeTab(i)
    
class MainWindow(QMainWindow):
    # existing code...

    def closeEvent(self, event):
        event.accept()
        sys.exit(0)

app = QApplication(sys.argv)
window = MainWindow()
result = app.exec_()
sys.exit(result)
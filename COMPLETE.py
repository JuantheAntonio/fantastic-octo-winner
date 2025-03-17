import sys
import json
import re
from transformers import BartForConditionalGeneration, BartTokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog, QMessageBox, QMainWindow, QTabWidget, QHBoxLayout, QLabel, QToolBar, QAction
from PyQt5.QtCore import Qt, QBuffer, QIODevice
from PyQt5.QtGui import QImage, QPixmap, QIcon
import cv2
import numpy as np
from PIL import Image
import pytesseract
from tkinter import filedialog, Tk
from docx import Document
import os
from autocorrect import Speller  # Ensure this library is installed

pytesseract.pytesseract.tesseract_cmd = r"D:\\tesseract-OCR\\tesseract.exe"

class ImagePanel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)  # Fixed image panel size
        self.setAlignment(Qt.AlignCenter)

    def set_image(self, pixmap):
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class TextProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.input_text = ""  # Store the text from the JSON file
        self.model = None
        self.tokenizer = None
        self.initUI()

    def initUI(self):
        # Set up the layout
        layout = QVBoxLayout()

        # Create a button to load the JSON file
        self.load_button = QPushButton("Load JSON File", self)
        self.load_button.clicked.connect(self.load_json)
        layout.addWidget(self.load_button)

        # Create buttons for proofreading and summarization
        self.proofread_button = QPushButton("Proofread", self)
        self.proofread_button.clicked.connect(self.proofread_text)
        self.proofread_button.setEnabled(False)  # Disable until JSON is loaded
        layout.addWidget(self.proofread_button)

        self.summarize_button = QPushButton("Summarize", self)
        self.summarize_button.clicked.connect(self.summarize_text)
        self.summarize_button.setEnabled(False)  # Disable until JSON is loaded
        layout.addWidget(self.summarize_button)

        # Create a text editor to display the results
        self.text_editor = QTextEdit(self)
        self.text_editor.setReadOnly(False)  # Allow editing
        layout.addWidget(self.text_editor)

        # Set the layout for the main window
        self.setLayout(layout)
        self.setWindowTitle("BART-POWERED AI")
        self.setGeometry(300, 300, 600, 500)

    def load_json(self):
        # Open a file dialog to select a JSON file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json)")

        if file_path:
            try:
                # Read the JSON file
                with open(file_path, "r", encoding="utf-8") as file:
                    self.data = json.load(file)

                # Extract the text from the JSON file
                self.input_text = self.data.get("text", "")

                if not self.input_text:
                    QMessageBox.warning(self, "Error", "No text found in the JSON file under the key 'text'.")
                else:
                    # Display the text in the text editor
                    self.text_editor.setPlainText(self.input_text)

                    # Enable the function buttons after JSON is loaded
                    self.proofread_button.setEnabled(True)
                    self.summarize_button.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def proofread_text(self):
        try:
            # Fix spacing and spelling
            proofread_text = self.fix_spacing(self.input_text)
            proofread_text = self.correct_spelling(proofread_text)
            self.text_editor.setPlainText(proofread_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during proofreading: {str(e)}")

    def summarize_text(self):
        try:
            # Load the model and tokenizer if not already loaded
            if self.model is None or self.tokenizer is None:
                self.load_model()

            # Summarize the text using DistilBART
            summary = self.generate_summary(self.input_text)
            filtered_summary = self.filter_summary(summary, self.input_text)
            self.text_editor.setPlainText(filtered_summary)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during summarization: {str(e)}")

    def fix_spacing(self, text):
        # Fix extra spaces between words and after punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)  # Fix spaces after punctuation
        return text.strip()

    def correct_spelling(self, text):
        # Use autocorrect to fix spelling errors
        spell = Speller()
        return spell(text)

    def load_model(self):
        # Load the DistilBART model and tokenizer
        model_name = "sshleifer/distilbart-cnn-12-6"
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def generate_summary(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=100,  # Limit summary length to avoid extra text
            num_beams=4,     # Use beam search for better quality
            early_stopping=True,
            no_repeat_ngram_size=2,  # Avoid repetition
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def filter_summary(self, summary, original_text):
        original_words = set(original_text.split())
        summary_words = summary.split()
        filtered_words = [word for word in summary_words if word in original_words]
        return " ".join(filtered_words)

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

        action_open_ai = QAction(QIcon(r"C:\Users\HP\OneDrive\Desktop\visualstudio\python series\IMAGE TO TEXT REE\FINAL\icons\AI.png"), "Open AI", self)
        action_open_ai.setToolTip("Open AI")
        action_open_ai.triggered.connect(self.open_ai_window)
        self.left_toolbar.addAction(action_open_ai)

    def open_ai_window(self):
        self.ai_window = TextProcessorApp()
        self.ai_window.show()

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
    
    def closeEvent(self, event):
        event.accept()
        sys.exit(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    result = app.exec_()
    sys.exit(result)
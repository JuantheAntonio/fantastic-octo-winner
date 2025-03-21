import sys
import json
import re
from transformers import BartForConditionalGeneration, BartTokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog, QMessageBox
from autocorrect import Speller  # Ensure this library is installed

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextProcessorApp()
    window.show()
    sys.exit(app.exec_())

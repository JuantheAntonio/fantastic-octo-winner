import sys
import json
from transformers import BartForConditionalGeneration, BartTokenizer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog, QMessageBox

class TextProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.input_text = ""  # Store the text from the JSON file
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

        self.summarize_button = QPushButton("Summarize (Coming soon)", self)
        self.summarize_button.clicked.connect(self.summarize_text)
        self.summarize_button.setEnabled(False)  # Disable until JSON is loaded
        layout.addWidget(self.summarize_button)

        # Create a text editor to display the results
        self.text_editor = QTextEdit(self)
        self.text_editor.setReadOnly(False)  # Allow editing
        layout.addWidget(self.text_editor)

        # Set the layout for the main window
        self.setLayout(layout)
        self.setWindowTitle("Text Processor")
        self.setGeometry(300, 300, 600, 500)

    def load_json(self):
        # Open a file dialog to select a JSON file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json)")

        if file_path:
            try:
                # Read the JSON file
                with open(file_path, "r", encoding="utf-8") as file:
                    self.data = json.load(file)
                    # print("JSON Data:", self.data)  # Debugging: Print JSON data

                # Extract the text from the JSON file
                self.input_text = self.data.get("text", "")
                # print("Extracted Text:", self.input_text)  # Debugging: Print extracted text

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
        # Fix spacing and spelling
        proofread_text = self.fix_spacing(self.input_text)
        proofread_text = self.correct_spelling(proofread_text)
        self.text_editor.setPlainText(proofread_text)

    def summarize_text(self):
        # Summarize the text using DistilBART
        summary = self.generate_summary(self.input_text)
        self.text_editor.setPlainText(summary)

    def fix_spacing(self, text):
        # Fix extra spaces between words and after punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)  # Fix spaces after punctuation
        return text.strip()

    def correct_spelling(self, text):
        # Use autocorrect to fix spelling errors
        spell = Speller()
        return spell(text)

    def generate_summary(self, text):
        # Load the DistilBART model and tokenizer
        model_name = "sshleifer/distilbart-cnn-12-6"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=100,  # Limit summary length to avoid extra text
            num_beams=4,     # Use beam search for better quality
            early_stopping=True,
            no_repeat_ngram_size=2,  # Avoid repetition
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
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
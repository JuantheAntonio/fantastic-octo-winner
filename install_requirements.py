import subprocess
import sys

# List of required modules
modules = [
    "pytesseract",
    "opencv-python",
    "numpy",
    "pillow",
    "PyQt5",
    "python-docx",
    "fastapi",
    "python-multipart",
    "transformers",
    "datasets",
    "torch",
    "sentencepiece"
]

# Install each module
for module in modules:
    subprocess.call([sys.executable, "-m", "pip", "install", module])

print("All required modules have been installed.")

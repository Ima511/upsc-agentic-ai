import fitz  # PyMuPDF
import re

class UPSCPDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_raw_text(self):
        """Extracts text from PDF and handles basic cleanup."""
        doc = fitz.open(self.file_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text

    def segment_questions(self, raw_text):
        """
        Splits text based on UPSC question numbering pattern.
        Pattern: Looking for '1.', '2.', '100.' at the start of lines.
        """
        # Regex to identify question starts (e.g., "\n1. ", "\n2. ")
        pattern = r'\n(?=\d+\.)' 
        questions = re.split(pattern, raw_text)
        
        # Filter out noise (shorter fragments that aren't real questions)
        cleaned_questions = [q.strip() for q in questions if len(q) > 60]
        return cleaned_questions
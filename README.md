# Technical Resume Classification

This project provides a robust pipeline for classifying technical resumes using state-of-the-art natural language processing techniques. It is designed to handle resumes in PDF format, clean and process text data, and classify resumes as either accepted or rejected based on their content.

Project Structure
The project consists of the following main blocks:

Resume Importer

Function: Converts resumes from PDF format to plain text.
Features:
Handles various PDF structures and formats.
Extracts clean and readable text from resumes.
Text Cleaning

Function: Preprocesses and cleans the extracted text data.
Features:
Removes unwanted characters, whitespace, and formatting issues.
Normalizes text for consistent processing.
Feature Processor

Function: Extracts features from cleaned text using a custom-trained BERT model.
Features:
Utilizes BERT (Bidirectional Encoder Representations from Transformers) for deep contextual understanding.
Generates embeddings that represent the semantic meaning of the text.
Feature Processing and Classification

Function: Processes the extracted features and classifies resumes using an XLNet model.
Features:
Uses XLNet (Extra Long Network) for classification tasks.
Classifies resumes as either accepted or rejected based on learned features.

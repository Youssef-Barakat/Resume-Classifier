"""
This module contains the ResumeClassificationPipeline class, which is used to run the entire resume classification pipeline from start to finish.

The pipeline consists of the following steps:
1. Resume Processing: Extracts raw text from resumes. Supports batch processing.
2. Text Cleaning: Cleans raw text.
3. Feature Extraction: Extracts features from cleaned text.
4. Preprocessing: Computes the final feature vectors for the classification model.
5. Classification: Classifies resumes based on their feature vectors.

The pipeline can be run in its entirety or in parts. For example, if you already have the raw text extracted from resumes, you can skip the resume processing step and start with the text cleaning step.
"""
import typing
from pathlib import Path
import pandas as pd

# Abstract Classes
from resume_processing.resume_import import ResumeImporter, load_builtin_resume_importer
from resume_processing.text_cleaning import TextCleaner, load_builtin_text_cleaner
from resume_processing.feature_extraction import ResumeFeatureExtractor, load_builtin_feature_extractor
from resume_processing.feature_processing import FeatureProcessor, load_builtin_feature_processor
from classification.resume_classifiers import ResumeClassificationModel, load_builtin_classification_model

from utils import *


class ResumeClassificationPipeline:
    """
    This class is responsible for running the entire resume classification pipeline from start to finish.
    The pipeline consists of the following steps:
    1. Resume Processing: Extracts raw text from resumes. Supports batch processing.
    2. Text Cleaning: Cleans raw text.
    3. Feature Extraction: Extracts features from cleaned text.
    4. Preprocessing: Computes the final feature vectors for the classification model.
    5. Classification: Classifies resumes based on their feature vectors.

    The pipeline can be run in its entirety or in parts. For example, if you already have the raw text extracted from resumes, you can skip the resume processing step and start with the text cleaning step.
    
    Example:
    --------
    >>> from resume_classifier import ResumeClassificationPipeline
    >>> pipeline = ResumeClassificationPipeline()
    >>> pipeline.classify_resumes_by_paths(resume_paths=['path/to/resume1.pdf', 'path/to/resume2.pdf'])
    """
    def __init__(
            self, resume_importer: typing.Union[str, ResumeImporter] = 'fitz', text_cleaner: typing.Union[str, TextCleaner] = 'nltk',
            feature_extractor: typing.Union[str, ResumeFeatureExtractor] = 'hybrid', preprocessor: typing.Union[str, FeatureProcessor] = 'xlnet',
            classification_model: typing.Union[str, ResumeClassificationModel] = 'xlnet_modified'
            ):
        self.resume_importer = resume_importer if type(resume_importer) is ResumeImporter else load_builtin_resume_importer(resume_importer)
        self.text_cleaner = text_cleaner if type(text_cleaner) is TextCleaner else load_builtin_text_cleaner(text_cleaner)
        self.feature_extractor = feature_extractor if type(feature_extractor) is ResumeFeatureExtractor else load_builtin_feature_extractor(feature_extractor)
        self.preprocessor = preprocessor if type(preprocessor) is FeatureProcessor else load_builtin_feature_processor(preprocessor)
        self.classification_model = classification_model if type(classification_model) is ResumeClassificationModel else load_builtin_classification_model(classification_model)


    def classify_resume_batch_by_paths(
            self, resume_paths: typing.List[Path]
            ) -> pd.DataFrame:
        """
        Classifies resumes by their file paths.

        Parameters
        ----------
        resume_paths (List[str]): List of resume file paths.

        Returns
        -------
        predictions (pd.DataFrame): DataFrame containing the resume file paths and their predictions.
        """
        # 1. Resume Processing
        resume_batch = self.resume_importer.import_resume_batch_by_paths(resume_paths)
        # 2. Text Cleaning
        cleaned_resume_batch = self.text_cleaner.clean_batch(resume_batch)    
        # 3. Feature Extraction
        feature_df = self.feature_extractor.extract_features_from_batch(cleaned_resume_batch)        
        # 4. Preprocessing
        model_input = self.preprocessor.generate_model_inputs(feature_df)        
        # 5. Classification
        predictions = self.classification_model.classify(model_input)

        return predictions
    

    def classify_resume_batch_by_texts(
            self, resume_texts: typing.List[str]
            ) -> pd.DataFrame:
        """
        Classifies resumes by their texts.

        Parameters
        ----------
        resume_texts (List[str]): List of resume texts.

        Returns
        -------
        predictions (pd.DataFrame): DataFrame containing the resume texts and their predictions.
        """
        # 1. Resume Processing (SKIPPED because we already have the resume texts)
        resume_batch = resume_texts
        # 2. Text Cleaning
        cleaned_resume_batch = self.text_cleaner.clean_batch(resume_batch)    
        # 3. Feature Extraction
        feature_df = self.feature_extractor.extract_features_from_batch(cleaned_resume_batch)
        # 4. Preprocessing
        model_input = self.preprocessor.generate_model_inputs(feature_df)        
        # 5. Classification
        predictions = self.classification_model.classify(model_input)

        return predictions


    def classify_resumes_by_reference_file(
            self, reference_csv_path: str
            ) -> pd.DataFrame:
        """
        Classifies resumes by their file paths.

        Parameters
        ----------
        reference_csv_path (str): Path to the reference csv file.
        output_csv_path (str): Path to the output csv file.
        """
        # 1. Resume Processing
        resume_batch = self.resume_importer.import_resume_batch_by_reference_file(reference_csv_path)
        # 2. Text Cleaning
        cleaned_resume_batch = self.text_cleaner.clean_batch(resume_batch)    
        # 3. Feature Extraction
        feature_df = self.feature_extractor.extract_features_from_batch(cleaned_resume_batch)        
        # 4. Preprocessing
        model_input = self.preprocessor.generate_model_inputs(feature_df)        
        # 5. Classification
        predictions = self.classification_model.classify(model_input)
        
        return predictions

        
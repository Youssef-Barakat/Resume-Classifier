from abc import ABC, abstractmethod
from pathlib import Path
import typing
import os

from constants import feature_extraction as cs

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline



class ResumeFeatureExtractor(ABC):
    """
    Abstract class for extracting resume features from text.
    """
    @abstractmethod
    def extract_features_from_text(
        self, resume_text: str, output_format: str = cs.DATAFRAME
        ) -> typing.Union[dict, pd.DataFrame]:
        """
        Extracts features from the given text.

        Parameters:
        ----------
            resume_text (str): The text from which features are to be extracted.
            output_format (str): The format in which the extracted features are to be returned.
                                Supported formats: 'dict', 'df'.

        Returns:
        --------
            Extracted features in the specified format.
        """
        pass

    @abstractmethod
    def extract_features_from_batch(
        self, resume_batch: typing.List[str], output_format: str = cs.DATAFRAME
        ) -> typing.Union[pd.DataFrame, typing.List[dict]]:
        """
        Extracts features from a batch of texts.

        Parameters:
        ----------
            resume_batch (List[str]): A list of texts from which features are to be extracted.
            output_format (str): The format in which the extracted features are to be returned.
                                Supported formats: 'dict', 'df'.

        Returns:
        --------
            Extracted features in the specified format.
        """
        pass


##################################################################################################
############################# ResumeFeatureExtractor Implementations #############################
##################################################################################################
class HybridFeatureExtractor(ResumeFeatureExtractor):
    """
    This class is responsible for extracting resume features from cleaned text.
    The class uses a hybrid approach to extract features from the cleaned text. It uses a combination of rule-based and machine learning-based approaches.
    """
    def __init__(self, spacy_model: typing.Union[str, Path] = 'en_core_web_sm'):
        """
        Parameters:
        -----------
        spacy_model: The name or path of the Spacy model used in the backend.
        """
        self.model = pipeline("text-classification", model = cs.MODEL_FEATURE_EXT_PATH)

        if type(spacy_model) is str:
            if spacy_model not in spacy.util.get_installed_models():
                raise ValueError("SpaCy model name invalid or not installed. [" + str(spacy_model) + "]")
        if type(spacy_model) is Path:
            if not os.path.exists(spacy_model):
                raise FileNotFoundError("SpaCy model path does not exist. [" + str(spacy_model) + "]")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print("SpaCy model not found. Downloading small spaCy model [en_core_web_sm]...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        

    def _extract_skills(
            self, text, noun_chunks, skills_file=None
            ) -> typing.List[str]:
        """
        Extracts skills from natural language text using a list of known skills.

        Parameters:
        -----------
            text (spacy.Span): The Spacy Doc object representing the input text.
            noun_chunks (spacy.Span): A list of Spacy Span objects representing noun chunks in the text.
            skills_file (str, optional): The path to a CSV file containing a list of known skills.
                                        Each skill should be listed in a separate column.
                                        Defaults to None.

        Returns:
        --------
            list: A list of unique skills extracted from the input text, with each skill capitalized.

        Note:
        -----
            The function searches for skills in the provided text by matching one-grams, bi-grams, and tri-grams
            with the skills listed in the CSV file. It removes common stopwords before matching.
            
            If 'skills_file' is not provided, the function attempts to read skills from a default location.
            If the CSV file is missing, a 'FileNotFoundError' is caught, and a message is printed.
        """
        if not skills_file:
            data = pd.read_csv(skills_file)
        else:
            data = pd.read_csv(cs.SKILLS_MATCHER_PATH)

        tokens = [token.text for token in text if not token.is_stop]
        skills = list(data.columns.values)
        skillset = []
        
        # Checking for monograms
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)

        # Checking for bi-grams and tri-grams
        for token in noun_chunks:
            token = token.text.lower().strip()
            if token in skills:
                skillset.append(token)

        return [i.capitalize() for i in set([i.lower() for i in skillset])]


    def _extract_langs(
            self, text, lang_file: typing.Optional[str] = cs.LANGS_MATCHER_PATH
            ) -> typing.List[str]:
        """
        Extracts languages from natural language text using a list of known languages.

        Parameters:
        ----------
            nlp_text (spacy.Doc): The Spacy Doc object representing the input text.
            lang_file (str, optional): The path to a CSV file containing a list of known languages.
                                    Each language should be listed in a separate column.
                                    Defaults to None.

        Returns:
        --------
            list: A list of unique languages extracted from the input text, with each language capitalized.

        Note:
        -----
            The function searches for languages in the provided text by matching individual tokens
            with the languages listed in the CSV file. It removes common stopwords before matching.
            
            If 'lang_file' is not provided, the function attempts to read languages from a default location.
            If the CSV file is missing, a 'FileNotFoundError' is caught, and a message is printed.
        """
        data = pd.read_csv(lang_file)

        tokens = [token.text for token in text if not token.is_stop]
        langs = list(data.columns.values)
        langs = [i.lower() for i in langs]
        langset = []
        for token in tokens:
            if token.lower() in langs:
                langset.append(token)
                
        return [i.capitalize() for i in set([i.lower() for i in langset])]
    

    def _parse_text_to_dict(
            self, resume_text: str
            ) -> dict:
        """
        Parses the given text and extracts information related to skills, education, experience, and projects.

        Parameters:
        -----------
            resume_text (str): The text from which information is to be extracted.

        Returns:
        --------
            dict: A dictionary containing the extracted information organized by categories.
                - 'CPT' (str): A formatted string summarizing the extracted data.
                - 'skills' (list): A list of extracted skills.
                - 'edu' (list): A list of extracted educational information.
                - 'exp' (list): A list of extracted experience-related information.
                - 'projects' (list): A list of extracted project-related information.
                - 'language' (list): A list of extracted language-related information.
        """
        doc = self.nlp(resume_text)

        skills_list = []
        language_list = []
        education_list = []
        experience_list = []
        projects_list = []

        for sentence in doc.sents:
            label = self.model(sentence.text, truncation=True)[0]['label']

            if label == cs.LABEL_0:
                projects_list.append(sentence)

            elif label == cs.LABEL_1:
                skill_list = self._extract_skills(sentence, sentence.noun_chunks, skills_file=cs.SKILLS_MATCHER_PATH)
                if skill_list:
                    skills_list.append(skill_list)

            elif label == cs.LABEL_2:
                education_list.append(sentence)


            elif label == cs.LABEL_3:
                lang_list = self._extract_langs(sentence, lang_file=cs.LANGS_MATCHER_PATH)
                if lang_list:
                    language_list.append(lang_list)

            elif label == cs.LABEL_4:
                experience_list.append(sentence)

        skills_list = np.array([item for sublist in skills_list for item in sublist])
        skills_list= np.unique(skills_list)

        language_list = np.array([item for sublist in language_list for item in sublist])
        language_list = np.unique(language_list)

        extract_dict = {
            cs.DICT_SKILLS: skills_list
            , cs.DICT_EDU: education_list
            , cs.DICT_EXP: experience_list
            , cs.DICT_PROJECTS: projects_list
            , cs.DICT_LANG: language_list
            }
        
        skills_list = list(skills_list)
        language_list = list(language_list)
        
        return extract_dict
    

    def _parse_text_to_df(
            self, resume_text: str
            ) -> pd.DataFrame:
        """
        Converts the information extracted from a text into a Pandas DataFrame.

        Parameters:
        -----------
            extract_dict (dict): A dictionary containing extracted information, typically generated by the 'parser' function.

        Returns:
        --------
            pd.DataFrame: A Pandas DataFrame containing the extracted information organized by columns.
                - 'text': The original text from which information was extracted.
                - 'CPT': A formatted string summarizing the extracted data.
                - 'skills': A list of extracted skills.
                - 'exp': A list of extracted experience-related information.
                - 'projects': A list of extracted project-related information.
                - 'education': A list of extracted educational information.
        """
        extract_dict = self._parse_text_to_dict(resume_text)
        extract_df = pd.DataFrame({
            cs.DICT_TEXT: [resume_text],
            cs.DICT_SKILLS: [extract_dict[cs.DICT_SKILLS]],
            cs.DICT_EXP: [extract_dict[cs.DICT_EXP]],
            cs.DICT_PROJECTS: [extract_dict[cs.DICT_PROJECTS]],
            cs.DICT_EDU: [extract_dict[cs.DICT_EDU]],
            cs.DICT_LANG: [extract_dict[cs.DICT_LANG]]
        })
        
        return extract_df
    

    def extract_features_from_text(
            self, resume_text: str, output_format: str = cs.DATAFRAME
            ) -> typing.Union[dict, pd.DataFrame]:
        """
        Extracts features from the given text.

        Parameters:
        ----------
            resume_text (str): The text from which features are to be extracted.
            output_format (str): The format in which the extracted features are to be returned.
                                Supported formats: 'dict', 'df'.

        Returns:
        --------
            Extracted features in the specified format.
        """
        # Checking if text is valid
        if not isinstance(resume_text, str):
            raise TypeError("resume_text must be a string. Detected type: " + str(type(resume_text)))

        if output_format == cs.DICT:
            return self._parse_text_to_dict(resume_text)
        elif output_format == cs.DATAFRAME:
            return self._parse_text_to_df(resume_text)
        else:
            raise ValueError("output_format must be one of the following: " + str(cs.SUPPORTED_OUTPUT_FORMATS) + ". [" + output_format + "]")
        

    def extract_features_from_batch(
            self, resume_batch: typing.List[str], output_format: str = cs.DATAFRAME
            ) -> typing.Union[pd.DataFrame, typing.List[dict]]:
        """
        Extracts features from a batch of texts.

        Parameters:
        ----------
            resume_batch (List[str]): A list of texts from which features are to be extracted.
            output_format (str): The format in which the extracted features are to be returned.
                                Supported formats: 'dict', 'df'.

        Returns:
        --------
            Extracted features in the specified format.
        """
        # Checking if text list is valid
        if not isinstance(resume_batch, list):
            raise TypeError("resume_batch must be a list of strings. Detected type: " + str(type(resume_batch)))
        if len(resume_batch) == 0:
            raise ValueError("resume_batch must not be empty.") 

        if output_format == cs.DICT:
            return [self.extract_features_from_text(resume_text, output_format=cs.DICT) for resume_text in resume_batch]
        elif output_format == cs.DATAFRAME:
            return pd.concat([self.extract_features_from_text(resume_text, output_format=cs.DATAFRAME) for resume_text in resume_batch], ignore_index=True)
        else:
            raise ValueError("output_format must be one of the following: " + str(cs.SUPPORTED_OUTPUT_FORMATS) + ". [" + output_format + "]")


####################################################################################################
####################################### Utility Functions ##########################################
####################################################################################################
def load_builtin_feature_extractor(extractor_name: str) -> ResumeFeatureExtractor:
    """
    Loads a concrete implementation of the _ResumeFeatureExtractor abstract class.

    Parameters:
    -----------
    extractor_name (str): The name of the concrete implementation of the _ResumeFeatureExtractor abstract class.

    Returns:
    --------
    A concrete implementation of the _ResumeFeatureExtractor abstract class.
    """
    if extractor_name == cs.HYBRID:
        return HybridFeatureExtractor()
    else:
        raise ValueError("extractor_name must be one of the following: " + str(cs.SUPPORTED_FEATURE_EXTRACTORS) + ". [" + extractor_name + "]")

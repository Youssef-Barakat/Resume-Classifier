from abc import ABC, abstractmethod
import typing

from constants import text_cleaning as cs

import nltk
import re

class TextCleaner(ABC):
    """
    Abstract class for cleaning raw text.
    """
    @abstractmethod
    def remove_stopwords_from_text(
        self, resume_text: str
        ) -> str:
        """
        This method is responsible for removing stop words from the raw extracted text.

        Parameters:
        -----------
        resume_text (str): The raw extracted text from the .pdf file.

        Returns:
        --------
        A string containing the raw extracted text with stop words removed.
        """
        pass

    @abstractmethod
    def lemmatize_text(
        self, resume_text: str
        ) -> str:
        """
        This method is responsible for lemmatizing the given text.

        Parameters:
        -----------
        resume_text (str): The extracted text from the .pdf file.

        Returns:
        --------
        A string containing the lemmatized text.
        """
        pass

    @abstractmethod
    def clean_text(
        self, resume_text: str, remove_stopwords: bool = True, lemmatize: bool = False
        ) -> str:
        """
        This method is responsible for cleaning the raw extracted.

        Parameters:
        -----------
        raw_text (str): The raw extracted text from the .pdf file.

        Returns:
        --------
        A string containing the cleaned text.
        """
        pass

    @abstractmethod
    def clean_batch(
        self, resume_batch: typing.List[str], remove_stopwords: bool = True, lemmatize: bool = False
        ) -> typing.List[str]:
        """
        This method is responsible for cleaning a batch of raw extracted texts.

        Parameters:
        -----------
        resume_batch (List[str]): A list of raw extracted texts from the .pdf files.

        Returns:
        --------
        A list of strings containing the cleaned texts.
        """
        pass


#####################################################################################################
#################################### TextCleaner Implementations ####################################
#####################################################################################################
class nltkTextCleaner(TextCleaner):
    """
    This class is responsible for cleaning the raw extracted text from the .pdf file.
    """
    def __init__(
            self, custom_stopwords: typing.Optional[list] = []
            ):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stop_words = self.stop_words.union(cs.EXTRA_STOP_WORDS)
        self.stop_words = self.stop_words.union(custom_stopwords)


    def remove_stopwords_from_text(
            self, resume_text: str
            ) -> str:
        """
        This method is responsible for removing nltk stop words from the raw extracted text.

        Parameters:
        -----------
        resume_text (str): The raw extracted text from the .pdf file.

        Returns:
        --------
        A string containing the raw extracted text with stop words removed.
                
        """
        if not isinstance(resume_text, str):
            raise TypeError("resume_text must be a string. Detected type: " + str(type(resume_text)))
        
        resume_text =  " ".join([word for word in str(resume_text).split() if word not in self.stop_words])
        return resume_text


    def lemmatize_text(
            self, resume_text: str
            ) -> str:
        """
        This method uses nltk's WordNetLemmatizer to lemmatize the given text.

        Parameters:
        -----------
        resume_text (str): The extracted text from the .pdf file.

        Returns:
        --------
        A string containing the lemmatized text.
        """
        if not isinstance(resume_text, str):
            raise TypeError("resume_text must be a string. Detected type: " + str(type(resume_text)))

        lemmatizer = nltk.stem.WordNetLemmatizer()
        resume_text = " ".join([lemmatizer.lemmatize(word) for word in resume_text.split()])
        return resume_text


    def clean_text(
            self, resume_text: str, remove_stopwords: bool = True, lemmatize: bool = False
            ) -> str:
        """
        This method is responsible for cleaning the raw extracted text using nltk and regular expressions.

        Parameters:
        -----------
        raw_text (str): The raw extracted text from the .pdf file.

        Returns:
        --------
        A string containing the cleaned text.
        """
        resume_text = resume_text['text']
        if not isinstance(resume_text, str):
            raise TypeError("resume_text must be a string. Detected type: " + str(type(resume_text)))
        
        # Text cleaning using regular expressions
        resume_text = re.sub('http\S+\s*', ' ', resume_text)  # Removing urls
        resume_text = re.sub('RT|cc', ' ', resume_text)  # Removing RT and cc
        resume_text = re.sub('#\S+', '', resume_text)  # Removing hashtags
        resume_text = re.sub('@\S+', '  ', resume_text)  # Removing mentions
        resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # Removing punctuations
        resume_text = re.sub(r'[^\x00-\x7f]',r' ', resume_text) # Removing non-ASCII characters
        resume_text = re.sub('\s+', ' ', resume_text)  # Removing extra whitespaces
        resume_text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', ' ', resume_text) # Removing phone numbers

        # Removing stop words
        if remove_stopwords:
            resume_text = self.remove_stopwords_from_text(resume_text)

        # Lemmatizing text
        if lemmatize:
            resume_text = self.lemmatize_text(resume_text)

        return resume_text


    def clean_batch(
            self, resume_batch: typing.List[str], remove_stopwords: bool = True, lemmatize: bool = False
            ) -> typing.List[str]:
        """
        This method is responsible for cleaning a batch of raw extracted texts using nltk and regular expressions.

        Parameters:
        -----------
        resume_batch (List[str]): A list of raw extracted texts from the .pdf files.

        Returns:
        --------
        A list of strings containing the cleaned texts.
        """
        # Checking if text list is valid
        if not isinstance(resume_batch, list):
            raise TypeError("file_paths must be a list of .pdf file paths. \"" + str(resume_batch) + "\"")
        if len(resume_batch) == 0:
            raise ValueError("file_paths must not be empty.")

        cleaned_resume_batch = []
        for resume_text in resume_batch:
            cleaned_resume_batch.append(self.clean_text(resume_text, remove_stopwords, lemmatize))
        return cleaned_resume_batch
    

#####################################################################################################
#################################### TextCleaner Implementations ####################################
#####################################################################################################
def load_builtin_text_cleaner(cleaner_name: str) -> TextCleaner:
    """
    Loads a concrete implementation of the _TextCleaner abstract class.

    Parameters:
    -----------
    cleaner_name (str): The name of the concrete implementation of the _TextCleaner abstract class.

    Returns:
    --------
    A concrete implementation of the _TextCleaner abstract class.
    """
    if cleaner_name == cs.NLTK:
        return nltkTextCleaner()
    else:
        raise ValueError("cleaner_name must be one of the following: " + str(cs.SUPPORTED_TEXT_CLEANERS) + ". [" + cleaner_name + "]")

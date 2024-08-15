from abc import ABC, abstractmethod
from pathlib import Path
import warnings
import typing
import os

from constants import resume_import as cs

import fitz


class ResumeImporter(ABC):
    """
    Abstract class for processing resumes (e.g. .pdf files) and extracting raw text. The class supports batch processing.
    """
    @ abstractmethod
    def import_resume_batch_by_paths(
        self, file_paths: typing.List[Path], return_value: str = cs.ALL_IMPORTED,
        append_to_imported: bool = True, suppress_warnings: bool = False
        ) -> dict:
        """
        Imports resumes from a list of .pdf file paths and returns a list of dictionaries describing the imported resumes.
        Parameters:
        -----------
        file_paths (list[str]): A list of file paths to .pdf files.
        return_value (str, optional): The output the user wants.
                            Supported outputs: 'all_imported', 'passed'

        Returns:
        --------
        A dictionary describing the imported resumes.
        """
        pass

    @ abstractmethod
    def import_resume_batch_by_dir(
        self, dir_path: Path, return_value: str = cs.ALL_IMPORTED,
        append_to_imported: bool = True, suppress_warnings: bool = False
        ) -> dict:
        """
        Imports resumes from a directory containing .pdf files and returns a list of dictionaries describing the imported resumes.
        
        Parameters:
        -----------
        dir_path (str): A directory containing .pdf files.
        return_value (str, optional): The output the user wants.
                            Supported outputs: 'all_imported', 'passed'
                            
        Returns:
        --------
        A dictionary describing the imported resumes.
        """
        pass

    @ abstractmethod
    def import_resume_batch_by_reference_file(
        self, file_path: Path, return_value: str = cs.ALL_IMPORTED,
        append_to_imported: bool = True, suppress_warnings: bool = False
        ) -> dict:
        """
        Imports resumes from a locations specified in a reference csv file and returns a list of dictionaries describing the imported resumes.
        
        Parameters:
        -----------
        file_path (Path): A file path to a reference file.
        return_value (str, optional): The output the user wants.
                            Supported outputs: 'all_imported', 'passed'
        append_to_imported (bool, optional): Whether to append the imported resumes to the list of imported resumes.
        suppress_warnings (bool, optional): Whether to suppress warnings.
                            
        Returns:
        --------
        A dictionary describing the imported resumes.
        """
        pass

    @ abstractmethod
    def import_resume_by_path(
        self, file_path: Path, return_value: str = cs.ALL_IMPORTED,
        append_to_imported: bool = True, suppress_warnings: bool = False
        ) -> dict:
        """
        Imports a single resume from a file path and returns a list of dictionaries describing the imported resumes.

        This method is useful for importing a single resume from a file path. If you want to import multiple resumes, use import_resume_batch_by_paths() instead.

        Parameters:
        -----------
        file_path (str): A file path to a .pdf file.
        return_value (str, optional): The output the user wants.
                            Supported outputs: 'all_imported', 'passed'

        Returns:
        --------
        A dictionary describing the imported resumes.
        """
        pass

    @ abstractmethod
    def get_imported_resumes(
        self, return_dicts: bool=True, return_texts: bool=False, return_docs: bool=False, return_paths: bool=False
        ) -> dict:
        """
        Returns the imported resumes in the specified format.
        
        Parameters:
        -----------
        return_dicts (bool, optional): Whether to return the dictionaries describing the imported resumes.
        return_texts (bool, optional): Whether to return the raw extracted texts from the imported resumes.
        return_docs (bool, optional): Whether to return the document objects of the imported resumes.
        return_paths (bool, optional): Whether to return the file paths of the imported resumes.
        
        Returns:
        --------
        Dictionary containing the imported resumes in the specified format.
        """
        pass


####################################################################################################
################################## ResumeImporter Implementations ##################################
####################################################################################################
class FitzImporter(ResumeImporter):
    """
    This class is responsible for importing resumes using the PyMuPDF package.

    The class supports batch processing. The user can import resumes from a list of file paths or from a directory containing .pdf files.

    Example:
    --------
    >>> from resume_classifier import FitzImporter
    >>> importer = FitzImporter()
    >>> res = importer.import_resume_batch_by_dir(dir_path='path/to/resumes')
    """
    def __init__(self):
        self.resume_paths = []
        self.resume_docs = []
        self.resume_texts = []
        self.resume_dicts = []

    
    def get_imported_resumes(
            self, return_dicts: bool=True, return_texts: bool=False, return_docs: bool=False, return_paths: bool=False
    ) -> dict:
        """
        Returns the imported resumes in the specified format.

        Parameters:
        -----------
        return_dicts (bool, optional): Whether to return the dictionaries describing the imported resumes.
        return_texts (bool, optional): Whether to return the raw extracted texts from the imported resumes.
        return_docs (bool, optional): Whether to return the PyMuPDF fitz documents of the imported resumes.
        return_paths (bool, optional): Whether to return the file paths of the imported resumes.

        Returns:
        --------
        Dictionary containing the imported resumes in the specified format.
        """
        d = {}
        if return_dicts:
            d[cs.DICTS] = self.resume_dicts
        if return_texts:
            d[cs.TEXTS] = self.resume_texts
        if return_docs:
            d[cs.DOCS] = self.resume_docs
        if return_paths:
            d[cs.PATHS] = self.resume_paths
        return d


    def import_resume_by_path(
            self, file_path: str, return_value: str = cs.ALL_IMPORTED, append_to_imported: bool = True, suppress_warnings: bool = False
            ) -> dict:
        """
        Imports a single resume from a file path and returns a dictionary describing the imported resume.

        This method is useful for importing a single resume from a file path. If you want to import multiple resumes, use import_resume_batch_by_paths() instead.

        Parameters:
        -----------
        file_path (str): A file path to a .pdf file.
        return_value (str, optional): The output the user wants.
                            Supported outputs: 'all_imported', 'passed'

        Returns:
        --------
        A dictionary containing the following keys:
        "text" (str): The raw extracted text from the .pdf file.
        "doc" (fitz.Document): A PyMuPDF fitz document.
        """
        # Checking if file path is valid
        # if not os.path.exists(file_path):
        #     raise FileNotFoundError("File path does not exist. [" + file_path + "]")
        if file_path[-4:] != ".pdf":
            raise ValueError("File path must be a .pdf file. [" + file_path + "]")

        # Opening the .pdf with PyMuPDF fitz
        try:    
            doc = fitz.open(file_path)
        except:
            if not suppress_warnings:
                warnings.warn("Failed to open .pdf file. [" + file_path + "]")
            return None

        # Extracting text from the .pdf
        resume_text = ""
        for page in doc:
            resume_text += page.get_text()

        # The final dictionary to be returned
        d = {cs.TEXT: resume_text, cs.DOC: doc}

        # Appending the dictionary to the list of imported resumes if the user permits it
        if append_to_imported:
            if d[cs.TEXT] not in self.resume_texts:
                self.resume_dicts.append(d)
                self.resume_paths.append(file_path)
                self.resume_texts.append(resume_text)
                self.resume_docs.append(doc)
            elif not suppress_warnings:
                warnings.warn(f'[{file_path}] already imported. Skipping...')

        # Returning the desired output
        if return_value == cs.ALL_IMPORTED:
            return self.resume_dicts if append_to_imported else [d] + self.resume_dicts
        elif return_value == cs.PASSED:
            return d
        raise ValueError("Unrecognized return_value. [" + return_value + "]")
    

    def import_resume_batch_by_paths(
            self, file_paths: typing.List[Path], return_value: str = cs.ALL_IMPORTED, append_to_imported: bool = True, suppress_warnings: bool = False
            ) -> typing.List[dict]:
        """
        Imports resumes from a list of .pdf file paths and returns a list of PyMuPDF fitz documents.

        Parameters:
        -----------
        file_paths (list[str]): A list of file paths to .pdf files.
        return_value (str, optional): The output the user wants.
                            Supported outputs: 'all_imported', 'passed'

        Returns:
        --------
        A dictionary containing the following keys:
        - "texts" (list[str]): A list of the raw extracted texts from the .pdf files.
        - "docs" (list[fitz.Document]): A list of PyMuPDF fitz documents.
        """
        # Checking if file paths list is valid
        if not isinstance(file_paths, list):
            raise TypeError("file_paths must be a list of .pdf file paths. \"" + str(file_paths) + "\"")
        if len(file_paths) == 0:
            warnings.warn("file_paths is empty. No resumes will be imported.")

        # Importing resumes from file paths using our import_resume_by_path() method
        dicts = []
        for path in file_paths:
            res = self.import_resume_by_path(path, return_value=cs.PASSED, append_to_imported=append_to_imported, suppress_warnings=suppress_warnings)
            if res is not None:
                dicts.append(res)
        
        # Returning the desired output
        if return_value == cs.ALL_IMPORTED:
            return self.resume_dicts if append_to_imported else dicts + self.resume_dicts
        elif return_value == cs.PASSED:
            return dicts
        raise ValueError("Unrecognized return_value. [" + return_value + "]")
    

    def import_resume_batch_by_dir(
            self, dir_path: Path, return_value: str = cs.ALL_IMPORTED, append_to_imported: bool = True, suppress_warnings: bool = False
            ) -> typing.List[dict]:
        """
        Imports resumes from a directory containing .pdf files and returns a list of dictionaries describing the imported resumes.

        Parameters:
        -----------
        dir_path (str): A directory containing .pdf files.
        return_value (str, optional): The output the user wants.
                            Supported outputs: 'all_imported', 'passed'

        Returns:
        --------
        A dictionary containing the following keys:
        - "texts" (list[str]): A list of the raw extracted texts from the .pdf files.
        - "docs" (list[fitz.Document]): A list of PyMuPDF fitz documents.
        """
        # Checking if directory path is valid
        if not os.path.exists(dir_path):
            raise FileNotFoundError("Directory path does not exist. [" + dir_path + "]")
        if not os.path.isdir(dir_path):
            raise ValueError("Given path must point to a directory. [" + dir_path + "]")

        # Getting the file paths of all .pdf files in the directory
        file_paths = []
        for file in os.listdir(dir_path):
            if file[-4:] == ".pdf":
                file_paths.append(os.path.join(dir_path, file))

        # Importing resumes from file paths using our import_resume_batch_by_paths() method
        return self.import_resume_batch_by_paths(file_paths, return_value=return_value, append_to_imported=append_to_imported, suppress_warnings=suppress_warnings)

    def import_resume_batch_by_reference_file(self, ref_file_path: Path, ref_column: str = cs.APPLICANT_ID, label_column: str = cs.LABEL, col_contains_paths: bool = True, dir_path: Path = None,
                                              return_value: str = cs.ALL_IMPORTED, append_to_imported: bool = True, suppress_warnings: bool = False) -> dict:
        """
        Imports resumes from a locations specified in a reference csv file and returns a list of dictionaries describing the imported resumes.
        
        Parameters:
        -----------
        ref_file_path (Path): A file path to a reference file.
        return_value (str, optional): The output the user wants.
                            Supported outputs: 'all_imported', 'passed'
        ref_column (str, optional): The name of the column containing the file paths or the file names.
        col_contains_paths (bool, optional): Whether the specified column contains the file paths or the file names.
        dir_path (Path, optional): The directory containing the resumes. Only required if col_contains_paths is False.
        append_to_imported (bool, optional): Whether to append the imported resumes to the list of imported resumes.
        suppress_warnings (bool, optional): Whether to suppress warnings.
                            
        Returns:
        --------
        A dictionary describing the imported resumes.
        """
        # Checking if file path is valid
        # if not os.path.exists(ref_file_path):
        #     raise FileNotFoundError("File path does not exist. [" + ref_file_path + "]")
        if ref_file_path[-4:] != ".csv":
            raise ValueError("File path must be a .csv file. [" + ref_file_path + "]")
        if col_contains_paths is False and dir_path is None:
            raise ValueError("dir_path must be specified if col_contains_paths is False.")

        # Reading the reference file
        import pandas as pd
        df = pd.read_csv(ref_file_path)

        # Checking if the reference file has the required column
        if ref_column not in df.columns:
            raise ValueError("\" + paths_column + \" column not found in the reference file. [" + ref_file_path + "]")
        
        # Getting the file paths of all .pdf files in the directory
        file_paths = []
        for ref in df[ref_column]:
            if col_contains_paths:
                file_paths.append(ref)
            else:
                file_paths.append(os.path.join(dir_path, str(ref) + ".pdf"))

        labels = pd.Series(df[label_column]) if label_column in df.columns else pd.Series()

        # Importing resumes from file paths using our import_resume_batch_by_paths() method
        texts = self.import_resume_batch_by_paths(file_paths, return_value=return_value, append_to_imported=append_to_imported, suppress_warnings=suppress_warnings)
        return texts, labels
####################################################################################################
####################################### Utility Functions ##########################################
####################################################################################################
def load_builtin_resume_importer(importer_name: str) -> ResumeImporter:
    """
    Loads a concrete implementation of the _ResumeProcessor abstract class.

    Parameters:
    -----------
    processor_name (str): The name of the concrete implementation of the _ResumeProcessor abstract class.

    Returns:
    --------
    A concrete implementation of the _ResumeProcessor abstract class.
    """
    if importer_name == cs.FITZ:
        return FitzImporter()
    else:
        raise ValueError("processor_name must be one of the following: " + str(cs.SUPPORTED_RESUME_PROCESSORS) + ". [" + importer_name + "]")
    
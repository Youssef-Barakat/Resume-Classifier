from abc import ABC, abstractmethod
import typing
import os

from constants import feature_processing as cs
from constants import feature_extraction as rfe
import utils

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer


class FeatureProcessor(ABC):
    """
    Abstract class for preprocessing resume features prior to inference.
    """
    @abstractmethod
    def generate_model_inputs(
        self, resume_features: typing.Union[pd.DataFrame, typing.List[dict]], output_format: str = cs.HUGGINGFACE_DATASET, strategy=cs.STRATEGY_CPT_ONLY
        ) -> typing.Union[Dataset, pd.DataFrame, dict]:
        """
        Generates the model input for the XLNet classification model.

        Parameters:
        -----------
        resume_features (pd.DataFrame): The resume features to be preprocessed.
        output_format (str): The format of the output. Must be one of the following: ['huggingface_dataset', 'df', 'dict'].

        Returns:
        --------
        The preprocessed resume features in the specified output format.
        """
        pass



####################################################################################################
################################# FeatureProcessor Implementations #################################
####################################################################################################
class AutoFeatureProcessor(FeatureProcessor):
    def __init__(self, model_name: str = cs.XLNET_LARGE):    
        config = utils.load_config()
        self.path = os.path.join(config['models_dir'], model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.path, 'tokenizer'), cache_dir=self.path, local_files_only=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=self.path)
        self.tokenizer.save_pretrained(os.path.join(self.path, 'tokenizer'))
        print("Tokenizer saved to " + os.path.join(self.path, 'tokenizer'))

    def _generate_cpt(
        self, resume_features: typing.Union[pd.DataFrame, dict]
        ) -> str:
        """
        Generates string representation of the CPT (clean parsed text).

        Parameters:
        -----------
        resume_features (pd.DataFrame): The resume features to be preprocessed.

        Returns:
        --------
        cpt (str): The string representation of the CPT.
        """
        # Converting the resume features to a dictionary if it is a DataFrame
        if type(resume_features) is pd.DataFrame:
            resume_features = resume_features.to_dict(orient='records')

        # Generating the CPT
        cpt = f'{rfe.DICT_SKILLS}: {resume_features[rfe.DICT_SKILLS][0] if len(resume_features[rfe.DICT_SKILLS]) > 0 else ""}\n\n'
        cpt += f'{rfe.DICT_EXP}: {resume_features[rfe.DICT_EXP][0] if len(resume_features[rfe.DICT_EXP]) > 0 else ""}\n\n'
        cpt += f'{rfe.DICT_PROJECTS}: {resume_features[rfe.DICT_PROJECTS][0] if len(resume_features[rfe.DICT_PROJECTS]) > 0 else ""}\n\n'
        cpt += f'{rfe.DICT_EDU}: {resume_features[rfe.DICT_EDU][0] if len(resume_features[rfe.DICT_EDU]) > 0 else ""}\n\n'
        cpt += f'{rfe.DICT_LANG}: {resume_features[rfe.DICT_LANG][0] if len(resume_features[rfe.DICT_LANG]) > 0 else ""}'

        return cpt


    
    def generate_model_inputs(
        self, resume_features: typing.Union[pd.DataFrame, typing.List[dict]], output_format: str = cs.HUGGINGFACE_DATASET,
        strategy=cs.STRATEGY_CPT_ONLY, train: bool = False, **kwargs
        ) -> typing.Union[Dataset, pd.DataFrame, dict]:
        """
        Generates the model input for the XLNet classification model.

        Parameters:
        -----------
        resume_features (pd.DataFrame): The resume features to be preprocessed.
        output_format (str): The format of the output. Must be one of the following: ['huggingface_dataset', 'df', 'dict'].
        strategy (str): The preprocessing strategy to use. Must be one of the following: ['combined', 'cpt_only', 'text_only'].

        Returns:
        --------
        The preprocessed resume features in the specified output format.
        """

        def tokenize_fn(batch):
            return self.tokenizer(batch["sentence"], padding=True, truncation=True) 
        
        # Converting the resume features to a list of dicts
        if type(resume_features) is pd.DataFrame:
            resume_features = resume_features.to_dict(orient='records')

        cpts = [self._generate_cpt(rf) for rf in resume_features]
        clean_texts = [rf[rfe.DICT_TEXT] for rf in resume_features]

        if strategy == cs.STRATEGY_COMBINED:
            features_str = [f'{cpts[i]}\n\n{rfe.DICT_TEXT}: {clean_texts[i]}' for i in range(len(resume_features))]
        elif strategy == cs.STRATEGY_CPT_ONLY:
            features_str = cpts
        elif strategy == cs.STRATEGY_TEXT_ONLY:
            features_str = clean_texts
        else:
            raise ValueError("strategy must be one of the following: " + str(cs.SUPPORTED_PREPROCESSING_STRATEGIES) + ". [" + strategy + "]")
        
        df = pd.DataFrame(features_str, columns=['sentence'])
        if train and 'labels' in kwargs:
            df['label'] = kwargs['labels']

        raw_datasets = Dataset.from_pandas(df)
        tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
        tokenized_datasets.set_format(type = 'torch')
        tokenized_datasets = tokenized_datasets.remove_columns("sentence")

        if train:
            tokenized_datasets = tokenized_datasets.rename_column("label","labels")
        
        # Convert the tokenized resume features to the specified output format and return them
        if output_format == cs.HUGGINGFACE_DATASET:
            return tokenized_datasets
        elif output_format == cs.DATAFRAME:
            return pd.DataFrame(tokenized_datasets)
        elif output_format == cs.DICT:
            return tokenized_datasets.to_dict()
        else:
            raise ValueError("output_format must be one of the following: " + str(cs.SUPPORTED_PREPROCESSOR_OUTPUT_FORMATS) + ". [" + output_format + "]")
        
"""
This module encompasses classes responsible for resume classification.
"""
from abc import ABC, abstractmethod
import typing
from constants import classification as cs
import torch.nn as nn

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import  XLNetForSequenceClassification, XLNetConfig

class ResumeClassificationModel(ABC):

    @abstractmethod
    def classify(
        self, resume_features: typing.Union[Dataset, pd.DataFrame, dict]
        ) -> typing.Union[pd.DataFrame, typing.List[dict]]:
        pass


#####################################################################################################
############################# ResumeClassificationModel Implementations #############################
#####################################################################################################
class XLNetClassificationModel(ResumeClassificationModel):
    def __init__(self):
        self.path = cs.MODEL_PATHS[cs.XLNET_MODIFIED]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = XLNetConfig.from_pretrained(r"C:\Users\PC\Downloads\model\config.json")
        self.model = XLNetForSequenceClassification(self.config)


    def classify(
            self, resume_features: typing.Union[Dataset, pd.DataFrame, dict], output_format: str = cs.DATAFRAME
            ) -> typing.Union[pd.DataFrame, typing.List[dict]]:
        """
        Classifies resumes.

        Parameters:
        -----------
        resume_features (Union[Dataset, pd.DataFrame, dict]): Tokenized resume features ready for inference.
        output_format (str): The format of the output. Must be one of the following: ['df', 'dict'].

        Returns:
        --------
        predictions (Union[pd.DataFrame, List[dict]]): The predictions of the resumes. [Verdict, Score]
        """
        # Convert the resume features to a dictionary if it is a DataFrame

        classification_head = nn.Sequential(
            nn.Linear(self.model.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.model.logits_proj = classification_head
        self.model.load_state_dict(torch.load(r"C:\Users\PC\Downloads\xlnet_modified2",map_location=torch.device('cpu')))


        # Move the model to the device
        self.model.to(self.device)
        test_dataloader = DataLoader(
            resume_features,  batch_size = 1)
        
        df = {'status':[], 'score':[]}

        with torch.no_grad():
            for batch in test_dataloader:
                
                batch = {k: v.to('cpu') for k,v in batch.items()}
                logits = self.model(**batch).logits
                predicted_class_idx = logits.argmax(-1).item()
                prediction = 'ACCEPTED' if predicted_class_idx == 1 else 'REJECTED'
                score = logits[0][predicted_class_idx].item()
                df['status'].append(prediction)
                df['score'].append(score)

        df = pd.DataFrame(df)
        return df

        # Convert the predictions to the specified output format and return them
        if output_format == cs.DATAFRAME:
            return pd.DataFrame([{cs.VERDICT: prediction, cs.SCORE: score}])
        elif output_format == cs.DICT:
            return [{cs.VERDICT: prediction, cs.SCORE: score}]
        else:
            raise ValueError("output_format must be one of the following: " + str(cs.SUPPORTED_MODEL_OUTPUT_FORMATS) + ". [" + output_format + "]")
        

####################################################################################################
###################################### Utility Functions ###########################################
####################################################################################################
def load_builtin_classification_model(model_name: str) -> ResumeClassificationModel:
    """
    Loads a concrete implementation of the _ResumeClassificationModel abstract class.

    Parameters:
    -----------
    model_name (str): The name of the concrete implementation of the _ResumeClassificationModel abstract class.

    Returns:
    --------
    A concrete implementation of the _ResumeClassificationModel abstract class.
    """
    if model_name == cs.XLNET_MODIFIED:
        return XLNetClassificationModel()
    else:
        raise ValueError("model_name must be one of the following: " + str(cs.SUPPORTED_CLASSIFICATION_MODELS) + ". [" + model_name + "]")
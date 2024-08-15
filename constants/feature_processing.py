SUPPORTED_PREPROCESSORS = ["xlnet"]
SUPPORTED_PREPROCESSING_STRATEGIES = ["combined", "cpt_only", "text_only"]
SUPPORTED_PREPROCESSOR_OUTPUT_FORMATS = ["huggingface_dataset", "df", "dict"]

XLNET = "xlnet"
XLNET_BASE = "xlnet-base-cased"
XLNET_LARGE = "xlnet-large-cased"

STRATEGY_COMBINED = "combined"
STRATEGY_CPT_ONLY = "cpt_only"
STRATEGY_TEXT_ONLY = "text_only"

HUGGINGFACE_DATASET = "huggingface_dataset"
DATAFRAME = "df"
DICT = "dict"

TOKENIZER_PATHS = {
    XLNET_LARGE: "data/tokenizers/xlnet_large_cased/models--xlnet-large-cased",
    XLNET_BASE: "../data/tokenizers/xlnet_base_cased",
}

MAX_SEQ_LENGTHS = {
    XLNET_LARGE: 1024,
    XLNET_BASE: 1024,
}
"""
This is a script to run the entire pipeline from start to finish.
It is meant to be called from the command line with two arguments:
1. The path to the reference csv file.
2. The path to the output csv file.
"""
import sys
import pandas as pd

from pipeline import ResumeClassificationPipeline

if __name__ == '__main__':
    reference_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    pipeline = ResumeClassificationPipeline()
    predictions = pipeline.classify_resumes_by_reference_file(reference_csv_path)

    resume_paths = pipeline.resume_importer.get_imported_resumes(return_paths=True)['paths']
    resume_ids = [path.split('\\')[-1][:-4] for path in resume_paths]
    predictions.insert(0, 'ID', resume_ids)

    predictions.to_csv(output_csv_path, index=False)

import pandas as pd
import json
from pathlib import Path
import sys
import numpy as np
import argparse
sys.path.append('../../')
import config

'''
Sample notes to annotate & format notes into format required by annotation tool
'''

def format_as_txt(df, filepath, sample_type=''):
    # write each note to a separate txt file
    
    for index, row in df.iterrows():
        text = row['Report_Text']
        empi = row['EMPI']
        noteid = row['Report_Number']
        time = row['Report_Date_Time']

        with open(filepath / f"{sample_type}empi_{empi}_note_{noteid}_time_{time}.txt", 'w') as f:
            f.write(text)


def sample_notes(fname, n_to_sample, annotator_names, sample_name):

    df = pd.read_csv(config.FILTERED_DATA_DIR / fname)
    sampled_df = df.sample(n=n_to_sample, random_state=42)
    
    # split the notes df evenly by the number of annotators
    chunks = np.array_split(sampled_df, len(annotator_names))

    # save the sampled
    for annotator, annotator_notes in zip(annotator_names, chunks):
        annotator_notes.to_csv(config.ANNOTATED_DATA_DIR / 'sampled_notes' / f'{annotator}{sample_name}_sampled_notes.csv', index=False)
        format_as_txt(annotator_notes, config.ANNOTATED_DATA_DIR / 'formatted_for_annotator', sample_type = f'{annotator}_notes{sample_name}_' )



def main():
    parser = argparse.ArgumentParser(description='Sample notes and format for Prancer.')
    parser.add_argument('--input_filename', type=str, default='filtered_discharge_summary_cohort.csv', help='Input filename containing notes') 
    parser.add_argument('--n_sample', type=int, help='Number of notes to sample') 
    parser.add_argument('--annotators', type=str, help='Comma-separated list of annotators') 
    parser.add_argument('--sample_name', type=str, default='', help='Name to give this sample (should start with _ )') 

    args = parser.parse_args()
    
    
    annotator_names = [a.strip() for a in args.annotators.split(',')]

    sample_notes(args.input_filename, args.n_sample, annotator_names, args.sample_name)
    
    

if __name__ == "__main__":
    main()
import pickle as pkl
from pathlib import Path
import argparse
import json
import numpy as np
import re
import pandas as pd
import sys
sys.path.append('../')
sys.path.append('../..')

import config
from collections import Counter
from sklearn.metrics import cohen_kappa_score, f1_score
pd.options.mode.chained_assignment = None  
from evaluate import load
from annotation_utils import clean_str


'''
Process the outputs of PRANcER
'''

######################################
# functions for converting Prancer output into a dataframe of annotations

def clean_annotations(note_annotations):
    '''
    Remove parts of the annotation that aren't relevant for non-CUI annotation
    '''
    for annot in note_annotations:
        annot.pop('CUIMode')
        annot.pop('experimentMode')
        for lab in annot['labels']:
            lab.pop('labelId')
            lab['categories'] = [c['type'] for c in lab['categories']]

    return note_annotations


def read_annotations(verbose=False):
    '''
    Read in all annotations from json files in the specified dir
    Returns dict from filename to annotation
    '''
    all_annotations = {}
    for filename in config.PRANCER_DATA_DIR.iterdir():
        if filename.suffix == '.json':
            print(filename.name)
            if verbose:
                print(filename)
            with open(filename, "r") as f:
                note_annotations = json.load(f)
                note_annotations = clean_annotations(note_annotations)
                all_annotations[filename.name] = note_annotations

    return all_annotations


def filter_rejected_annotations(annotations):
    '''
    Filter out "rejected" annotations
    '''
    all_annotations = {}
    for filename, note in annotations.items():
        note_annotations = []
        for annot in note:
            if annot['decision'] != 'rejected' and annot['decision'] != 'undecided':
                note_annotations.append(annot)
        all_annotations[filename] = note_annotations
    return all_annotations

def annotations_to_df(annotations, annotator_names):
    '''
    Convert annotations output from annotator tool to a dataframe
    '''

    all_annots = []
    for fname, annots in annotations.items():
        annotator = ''
        for a in annotator_names:
            if a in fname:
                annotator = a

        empi = re.search(r'empi_([0-9]+)_', fname).group(1)
        report_id = re.search(r'note_(.+)_time', fname).group(1)
        note_time = re.search(r'time_(.+)\.txt', fname).group(1)

        
        with open(str(config.PRANCER_DATA_DIR / fname.replace('.json', '.txt'))) as f:
            report_text = f.read()
        for a in annots:
            for label, span in zip(a['labels'], a['spans']):
                assert  len(label['categories']) == 1
                text_span = report_text[int(span['start']):int(span['end'])]
                annot_dict = {'EMPI': empi, 'Report_Number': report_id, 'Report_Date_Time': note_time, 'label_type': label['title'], 'label_category': label['categories'][0], 'start': span['start'], 'end': span['end'], 'span':text_span, 'annotator': annotator}
                all_annots.append(annot_dict)
    
    annot_df = pd.DataFrame(all_annots)
    return annot_df


def clean_spans(df, label):
    df.loc[df['label_type'] == label ,f'clean_span'] = df.loc[df['label_type'] == label ,'span'].apply(lambda s: clean_str(s, label, clean_type='annotation'))
    df.loc[df['label_type'] == label,['EMPI', 'Report_Number', 'span', 'clean_span']].to_csv(config.ANNOTATED_DATA_DIR / 'debug_ebl_annotation_postprocessing.csv', index=False)
    return df



#############################################

def main():
    
    parser = argparse.ArgumentParser(description='Process annotator output')
    parser.add_argument('--annotators', type=str, default='', help='Comma-separated list of annotators') 
    args = parser.parse_args()
    annotator_names = [a.strip() for a in args.annotators.split(',')]

    annotations = read_annotations()
    annotations = filter_rejected_annotations(annotations)
    
    empty_annotations = { k:v for k,v in annotations.items() if len(v) == 0}
    if len(empty_annotations) > 0:
        print('There are some notes with zero annotations. Please confirm that these notes weren\'t accidentally skipped.')
        print(list(empty_annotations.keys()))

    annotations = { k:v for k,v in annotations.items() if len(v) > 0}
    print(f'\nNumber of files with annotations: {len(annotations)}')

    df = annotations_to_df(annotations, annotator_names)
    print(df.head())

    # remove extra punctuation
    df['span'] = df['span'].str.replace('[,:\.]$|^[,:\.]', '')

    
    # remove annotations for non-delivery notes
    nondelivery_reports = df.loc[df['label_type'] == 'not a delivery note']
    print(f'There are {len(set(nondelivery_reports["Report_Number"].tolist()))} unique Report Numbers that are nondelivery reports')
    df.loc[df['label_type'] == 'not a delivery note', ['EMPI', 'Report_Number']].to_csv(config.ANNOTATED_DATA_DIR / 'nondelivery_notes.csv', index=False)
    print(df.head())
    print('nondelivery', nondelivery_reports.head())
    print(f'# of annotations prior to filtering out non-delivery reports: {len(df.index)}')
    merged_df = pd.merge(df, nondelivery_reports, how='outer', on=["EMPI", "Report_Number"], indicator=True, suffixes=('', '_y'))
    delivery_annotations = merged_df[merged_df['_merge'] == 'left_only'].copy() 
    delivery_annotations = delivery_annotations[delivery_annotations.columns.drop(list(delivery_annotations.filter(regex='_y')))]
    delivery_annotations.drop(columns=['_merge'], inplace=True)  
    print(f'# of annotations prior to filtering out non-delivery reports: {len(delivery_annotations.index)}')

    # stats
    num_annotations_per_label = Counter(delivery_annotations['label_type'].tolist())
    print('Annotations for each label: ', num_annotations_per_label)
    num_annotations_per_annotator = Counter(delivery_annotations['annotator'].tolist())
    print('Annotations for each annotator: ', num_annotations_per_annotator)
    num_annotated_notes_per_annotator = df[['Report_Number', 'EMPI']].groupby(df['annotator']).nunique() 
    print('Annotated notes for each annotator: ', num_annotated_notes_per_annotator)

    # clean up annotations
    print('Cleaning annotations...')
    delivery_annotations['clean_span'] = delivery_annotations['span']
    delivery_annotations = clean_spans(delivery_annotations,'estimated blood loss')

    # save to file
    delivery_annotations.to_csv(config.ANNOTATED_DATA_DIR / 'processed_annotations.csv', index=False)



if __name__ == "__main__":
    main()
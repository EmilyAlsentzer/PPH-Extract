import pandas as pd
import sys
sys.path.append('../')
import config
import numpy as np
from collections import Counter

#########################################
# data utils


def prep_data(annotations_filepath = 'processed_annotations.csv', notes_filepath = 'filtered_discharge_summary_cohort.csv', verbose=False):
    '''
    Filter notes to only those with annotations & return annotations, filtered notes, and joined annotations+notes
    '''
    annotations = pd.read_csv(config.ANNOTATED_DATA_DIR / annotations_filepath)
    all_notes = pd.read_csv(config.FILTERED_DATA_DIR  / notes_filepath)

    # get notes with annotations
    notes = all_notes.loc[all_notes['Report_Number'].isin(annotations['Report_Number'].tolist())]
    notes = pd.merge(all_notes, annotations[['EMPI', 'Report_Number']], how='inner', on=["EMPI", "Report_Number"])

    # count unique EMPI & report numbers in the notes
    if verbose: print(f"# notes: {len(notes)}, unique EMPI: {len(notes['EMPI'].unique())}, unique report numbers: {len(notes['Report_Number'].unique())}")

    # join annotations & notes
    joined_notes_annot = annotations.set_index(['EMPI', 'Report_Number']).join(notes.set_index(['EMPI','Report_Number']), how='left')
    joined_notes_annot = joined_notes_annot.reset_index()
    assert len(joined_notes_annot) == len(annotations)

    return annotations, notes, joined_notes_annot

def prep_notes_for_ie_label(notes, orig_notes, label, clean_span=True, verbose=False):
    '''
    Create IE labels and add to the notes dataframe

    When clean_span = True, we're using a cleaned up version of the annotations. In this setting, the start & end locations are no longer correct.
    '''
    notes_one_lab = notes.loc[notes['label_type'] == label]

    span = 'clean_span' if clean_span else 'span'
    notes_one_lab = notes_one_lab[['EMPI', 'Report_Number', 'Report_Text', 'Report_Date_Time', span]].groupby(['EMPI', 'Report_Number',  'Report_Text', 'Report_Date_Time'])[span].apply(list).reset_index(name='spans')
    no_ie_lab = orig_notes.loc[(~(orig_notes['EMPI'].isin(notes_one_lab['EMPI'].tolist())) | (~orig_notes['Report_Number'].isin(notes_one_lab['Report_Number'].tolist())))]
    no_ie_lab['spans'] =  np.empty((len(no_ie_lab), 0)).tolist()
    if 'EPIC_PMRN' in no_ie_lab.columns:
        no_ie_lab = no_ie_lab.drop(columns=['Report_Status', 'Report_Type', 'Report_Description', 'EPIC_PMRN'])
    notes_one_lab = pd.concat([no_ie_lab, notes_one_lab])
    
    # filter notes to those in current dataset (this is important when we only run on train or test data)
    notes_one_lab = notes_one_lab.loc[(notes_one_lab['Report_Number'].isin(orig_notes['Report_Number'])) & (notes_one_lab['EMPI'].isin(orig_notes['EMPI']))]
    notes_one_lab = notes_one_lab.sample(frac=1) # shuffle
    if verbose: print(f'There are {len(notes_one_lab.index)} notes in the ie dataset')
    notes_one_lab['span_extracted'] = notes_one_lab['spans'].apply(lambda s: 1 if len(s) > 0 else 0)
    return notes_one_lab

def prep_binary_labels(annotations, notes, label):
    '''
    Create binary labels using the annotations dataframe and add to the notes dataframe
    '''
    
    # Need to convert the original IE label into a binary label
    if label == 'cryo' or label == 'platelets' or label == 'ffp' or label == 'rbc':
        filtered_annotations = annotations.loc[annotations['label_type'] == 'transfusion type']
        if label == 'cryo': filtered_annotations = filtered_annotations.loc[filtered_annotations['span'].str.contains('cryo', case=False)]
        elif label == 'platelets': filtered_annotations = filtered_annotations.loc[(filtered_annotations['span'].str.contains('platelet', case=False)) | (filtered_annotations['span'].str.contains('plt', case=False))]
        elif label == 'ffp': filtered_annotations = filtered_annotations.loc[(filtered_annotations['span'].str.contains('ffp', case=False)) | (filtered_annotations['span'].str.contains('plasma', case=False))]
        elif label == 'rbc': filtered_annotations = filtered_annotations.loc[(filtered_annotations['span'].str.contains('ffp', case=False)) | (filtered_annotations['span'].str.contains('prbc|packed|red|blood|rbc', case=False, regex=True))]
        notes[label] = 0
        notes.loc[(notes['EMPI'].isin(filtered_annotations['EMPI'].tolist())) & (notes['Report_Number'].isin(filtered_annotations['Report_Number'].tolist())), label] = 1
        print(f'Count of each label for {label}: ', Counter(notes[label]))

    else:
        filtered_annotations = annotations.loc[annotations['label_type'] == label]

        notes[label] = 0
        notes.loc[(notes['EMPI'].isin(filtered_annotations['EMPI'].tolist())) & (notes['Report_Number'].isin(filtered_annotations['Report_Number'].tolist())), label] = 1
        print(f'Count of each label for {label}: ', Counter(notes[label]))

    return notes
from sklearn.model_selection import train_test_split
from collections import Counter
import sys
import random
import argparse

sys.path.append('../')


# local
import config
from model.data_utils import *

'''
Sample small set of the annotated notes to use for determining the best prompt for Flan-T5

'''

SEED = 42

def get_binary_train_data(args, annotations, notes, labels):
    for lab in labels:
        print('\nLabel: ', lab)
        
        # get labels
        notes = prep_binary_labels(annotations, notes, lab)
        print(Counter(notes['MRN_Type']))

        # limit to BWH notes
        non_bwh_empi = notes.loc[notes['MRN_Type'] != 'BWH', 'EMPI'].unique().tolist()
        only_bwh_notes = notes.loc[(notes['MRN_Type'] == 'BWH') & (~notes['EMPI'].isin(non_bwh_empi))]# notes at BWH & where EMPI isn't also at another institution
        only_bwh_empi = notes.loc[(notes['MRN_Type'] == 'BWH') & (~notes['EMPI'].isin(non_bwh_empi)), 'EMPI'].unique().tolist() # notes at BWH & where EMPI isn't also at another institution
    
    
        # perform stratified splitting
        grouped_notes = only_bwh_notes.groupby('EMPI').sum(numeric_only=True).reset_index()
        n_pos_labs = grouped_notes[lab].sum()
        if n_pos_labs < 2:
            train_empi, test_empi, = train_test_split(grouped_notes['EMPI'].tolist(), train_size=args.n_train, random_state=SEED)
        else:
            train_empi, test_empi, = train_test_split(grouped_notes['EMPI'].tolist(), train_size=args.n_train, random_state=SEED, stratify=grouped_notes[lab].tolist())


        train_notes = notes.loc[notes['EMPI'].isin(train_empi)]
        test_notes = notes.loc[~notes['EMPI'].isin(train_empi)]
        print(f'There are {len(train_notes.index)} train notes and {len(test_notes.index)} test notes.')

        print('train count', train_notes.drop(columns=['EMPI', 'EPIC_PMRN', 'MRN_Type', 'Report_Number', 'Report_Date_Time',
        'Report_Description', 'Report_Status', 'Report_Type', 'Report_Text']).sum(axis=0, numeric_only=True)[lab])
        print('test count', test_notes.drop(columns=['EMPI', 'EPIC_PMRN', 'MRN_Type', 'Report_Number', 'Report_Date_Time',
        'Report_Description', 'Report_Status', 'Report_Type', 'Report_Text']).sum(axis=0, numeric_only=True)[lab])
    
        train_notes.to_csv(config.ANNOTATED_DATA_DIR / 'train_test_split' / f"train_set_seed={SEED}_label={lab.replace('/', '_')}.csv", index=False)
        test_notes.to_csv(config.ANNOTATED_DATA_DIR / 'train_test_split' / f"test_set_seed={SEED}_label={lab.replace('/', '_')}.csv", index=False)


def get_ie_train_data(args, joined_notes, notes):
    for lab in config.ie_labels:
        print('\nLabel: ', lab)
        
        # get labels
        notes_one_lab = prep_notes_for_ie_label(joined_notes, notes, lab)
        print(Counter(notes_one_lab['MRN_Type']))

        notes_one_lab['span_extracted'] = notes_one_lab['spans'].apply(lambda s: 1 if len(s) > 0 else 0)
        
        # limit to BWH notes
        non_bwh_non_null_empi = notes_one_lab.loc[(notes_one_lab['MRN_Type'] != 'BWH') & (~pd.isnull(notes_one_lab['MRN_Type'])), 'EMPI'].unique().tolist()
        only_bwh_notes = notes_one_lab.loc[(notes_one_lab['MRN_Type'] == 'BWH') & (~notes_one_lab['EMPI'].isin(non_bwh_non_null_empi))]# notes at BWH & where EMPI isn't also at another institution
        only_bwh_null_notes = notes_one_lab.loc[(pd.isnull(notes_one_lab['MRN_Type'])) & (notes_one_lab['Report_Number'].str.contains('BWH')) & (~notes_one_lab['EMPI'].isin(non_bwh_non_null_empi))]# MRN_Type is null, but BWH in Report_Number
        only_bwh_notes = pd.concat([only_bwh_notes, only_bwh_null_notes])
        
        # perform stratified splitting
        grouped_notes = only_bwh_notes.groupby('EMPI').sum(numeric_only=True).reset_index()
        print(grouped_notes.head())
        train_empi, test_empi, = train_test_split(grouped_notes['EMPI'].tolist(), train_size=args.n_train, random_state=SEED, stratify=grouped_notes['span_extracted'].tolist())

        train_notes = notes_one_lab.loc[notes_one_lab['EMPI'].isin(train_empi)]
        test_notes = notes_one_lab.loc[~notes_one_lab['EMPI'].isin(train_empi)]
        print(f'There are {len(train_notes.index)} train notes and {len(test_notes.index)} test notes.')
        print('Label split in train set: ', Counter(train_notes['span_extracted']))
    
        train_notes.to_csv(config.ANNOTATED_DATA_DIR / 'train_test_split' / f"train_set_seed={SEED}_label={lab.replace('/', '_')}.csv", index=False)
        test_notes.to_csv(config.ANNOTATED_DATA_DIR / 'train_test_split' /  f"test_set_seed={SEED}_label={lab.replace('/', '_')}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='Get notes for prompt evaluation.')
    parser.add_argument('--label', type=str) 
    parser.add_argument('--n_train',  type=int, default=50) 
    parser.add_argument('--annotations_filename', type=str, default=None, help='Annotations dataframe.')
    parser.add_argument('--all_notes_filename', type=str, default=None, help='All notes dataframe.')
    args = parser.parse_args()
    
    annotations, notes, joined_notes = prep_data(annotations_filepath=args.annotations_filename, notes_filepath=args.all_notes_filename)
    print(f'Length notes: {len(notes.index)}')

    get_binary_train_data(args, annotations, notes, config.binary_labels)
    get_ie_train_data(args, joined_notes, notes)
    
    

if __name__ == "__main__":
    main()
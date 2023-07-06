import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


import sys
sys.path.append('../..')
sys.path.append('..')
import config
from utils import *

def get_non_ed_notes(verbose=False):
    '''
    Get all discharge summary notes that are not ED notes
    '''
    df = get_notes(verbose=verbose)
    original_n_notes = len(df.index)
    original_n_patients = len(df["EMPI"].unique())
    if verbose: 
        print('STEP 1 Complete: Read in Notes from RPDR')
        print(f'There are {original_n_notes} total notes to format for {original_n_patients} unique empi IDs')
        print('--------------')


    # filter out ED notes
    df = df.loc[df['Report_Description'] != 'ED Observation Discharge Summary']
    df = df.loc[df['Report_Description'] != 'ED Discharge Summary']
    if verbose: 
        print('Step 2 Complete: Filtered out ED Discharge summaries')
        print(f'There are now {len(df.index)} total notes to format for {len(df["EMPI"].unique())} unique empi IDs after filtering out ED notes')
        print(f'There are now {len(df[["Report_Number", "EMPI"]].drop_duplicates())} unique EMPI/Report Numbers. {len(df[["Report_Number", "EMPI", "Report_Date_Time"]].drop_duplicates())} unique EMPI/Report Numbers/Report Time')
        print(f'We removed {original_n_notes-len(df.index)} notes for {original_n_patients-len(df["EMPI"].unique())} patients.')
        print('--------------')

    return df

def get_landd_by_regex(df):

    # filter out notes with no text
    df = df.loc[~pd.isnull(df['Report_Text'])]
    
    # get notes containing any of the following strings
    filtered_df = df.loc[df['Report_Text'].str.contains('labor |delivery|L&D|cesarean section|c-section|estimated EDC|pregnancy', regex=True, case=False)]
    print(f'There are {len(filtered_df.index)} notes likely to be related to a delivery.')

    return filtered_df


def main():
    # get non-ED, pre 2015 RPDR notes for women with a pregnancy
    df = get_non_ed_notes(verbose=True)
    
    # get L&D notes
    delivery_note_df = get_landd_by_regex(df, broad=True)
    print('\nStep 3 Complete: Filtered out non-delivery discharge summaries')
    print(f"There are {len(delivery_note_df.index)} notes  for {len(delivery_note_df['EMPI'].unique())} unique EMPI and {len(delivery_note_df['Report_Number'].unique())} unique report numbers")
    print(f'There are {len(delivery_note_df[["Report_Number", "EMPI"]].drop_duplicates())} unique EMPI/Report Numbers. {len(delivery_note_df[["Report_Number", "EMPI", "Report_Date_Time"]].drop_duplicates())} unique EMPI/Report Numbers/Report Time')
    print(f"Filtering of notes to identify delivery notes by broad regex removed {len(df.index) - len(delivery_note_df.index)} notes & {len(df['EMPI'].unique())- len(delivery_note_df['EMPI'].unique())} patients")
    print('--------------')
    
    # get only final notes
    filtered_delivery_note_df = delivery_note_df.loc[delivery_note_df['Report_Status'] == 'F']
    print('\nStep 4 Complete: Filtered out draft discharge summaries')
    print(f"There are {len(filtered_delivery_note_df.index)} notes after filtering to only those notes with a \"final\" status")
    print(f'There are now {len(filtered_delivery_note_df["EMPI"].unique())} unique patients in the regex-filtered, final delivery note dataset.') 
    print('--------------')

    # drop duplicates
    filtered_delivery_note_df = filtered_delivery_note_df.drop_duplicates(subset=['EMPI', 'Report_Date_Time', 'Report_Number'])
    print('\nStep 5 Complete: Filtered out duplicate discharge summaries')
    print(f'There are {len(filtered_delivery_note_df.index)} notes in the delivery df after de-duplication')
    print(f'There are {len(filtered_delivery_note_df["EMPI"].unique())} unique patients in the de-dup dataset.') 
    print(f'The process of removing draft/duplicate notes removed {len(delivery_note_df.index) - len(filtered_delivery_note_df.index)} notes for {len(delivery_note_df["EMPI"].unique()) - len(filtered_delivery_note_df["EMPI"].unique())} patients.')
    print('--------------')
    
    # sanitize report numbers
    filtered_delivery_note_df['Report_Number'] = filtered_delivery_note_df['Report_Number'].str.replace('/', '_')
    
    # remove notes containing discharge instructions 
    filtered_delivery_note_df = filtered_delivery_note_df.loc[~((filtered_delivery_note_df['Report_Text'].str.contains('Discharge Instructions')) & (filtered_delivery_note_df['Report_Text'].str.contains('Please bring these discharge instructions to your follow-up appointments.')))]
    print(f'The length of filtered_delivery_note_df is {len(filtered_delivery_note_df.index)} after removing discharge instruction notes.')

    # remove notes containing discharge orders 
    filtered_delivery_note_df = filtered_delivery_note_df.loc[~filtered_delivery_note_df['Report_Text'].str.contains('****** DISCHARGE ORDERS ******',  regex=False)]
    print(f'The length of filtered_delivery_note_df is {len(filtered_delivery_note_df.index)} after removing discharge orders.')

    
    filtered_delivery_note_df.to_csv(config.FILTERED_DATA_DIR  / 'filtered_discharge_summary_cohort.csv', index=False)






if __name__ == "__main__":
    main()

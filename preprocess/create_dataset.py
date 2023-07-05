import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


import sys
sys.path.append('../..')
sys.path.append('..')
import config
from utils import get_non_ed_notes


def get_landd_by_regex(df):

    # filter out notes with no text
    df = df.loc[~pd.isnull(df['Report_Text'])]
    
    # get notes containing any of the following strings
    filtered_df = df.loc[df['Report_Text'].str.contains('labor |delivery|L&D|cesarean section|c-section|estimated EDC|pregnancy', regex=True, case=False)]
    print(f'There are {len(filtered_df.index)} notes likely to be related to a delivery.')

    return filtered_df


def main():
    # get non-ED, pre 2015 RPDR notes for women with a pregnancy
    df = get_non_ed_notes(verbose=True, pre_05_2015=True)
    
    if (config.FILTERED_DATA_DIR /'landd_filtered'/ 'pre_2015_notes' / 'regex_filtered_landd_discharge_summaries.csv').exists():
        delivery_note_df = pd.read_csv(config.FILTERED_DATA_DIR /'landd_filtered'/ 'pre_2015_notes'  / 'regex_filtered_landd_discharge_summaries.csv')
    else:
        delivery_note_df = get_landd_by_regex(df, broad=True)
        delivery_note_df.to_csv(config.FILTERED_DATA_DIR /'landd_filtered'/ 'pre_2015_notes' / 'regex_filtered_landd_discharge_summaries.csv', index=False)
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
    
    all_notes['Report_Number'] = all_notes['Report_Number'].str.replace('/', '_')

    
    filtered_delivery_note_df.to_csv(config.FILTERED_DATA_DIR  / 'filtered_discharge_summary_cohort.csv', index=False)






if __name__ == "__main__":
    main()

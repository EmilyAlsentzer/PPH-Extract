import pandas as pd
import sys
import numpy as np
sys.path.append('../..')
import config
from preprocess.utils import  get_icd_codes
from model.data_utils import *
from preprocess.pph_notes.sample_and_format_pph_related_notes import format_as_txt

'''
Sample discharge summaries with PPH ICD codes
'''

def get_icd():
    icd = get_icd_codes(filetype = 'Dia.txt')
    icd['Date'] = pd.to_datetime(icd['Date'])
    icd['Code'] = icd['Code'].astype(str)
    icd = icd.rename(columns={'Diagnosis_Name':'Name', 'Diagnosis_Flag': 'Flag'})
    icd = icd[['EMPI', 'Date', 'Name', 'Code_Type', 'Code', 'Flag', 'Inpatient_Outpatient', 'Encounter_number']]
    return icd

def get_pph_icd_codes(icd):
    pph_icd = icd.loc[icd['Code'].str.startswith('666')]
    obstetrical_trauma_icd = icd.loc[(icd['Code'].str.startswith('665'))| icd['Code'].str.startswith('664')] 
    zeutlin_specific_icd = icd.loc[(icd['Code'].str.startswith('656'))| icd['Code'].str.startswith('641')] 
    relevant_icd = pd.concat([pph_icd, obstetrical_trauma_icd, zeutlin_specific_icd])
    return relevant_icd

def get_delivery_notes():
    notes_df = pd.read_csv(config.FILTERED_DATA_DIR  / 'filtered_discharge_summary_cohort.csv')
    print(f'Number of discharge summaries: {len(notes_df.index)}, # patients: {len(notes_df["EMPI"].unique())}, # unique EMPI/ReportNumbers: {len(notes_df[["Report_Number", "EMPI"]].drop_duplicates())}, # unique EMPI/ReportNumber/ReportDate: {len(notes_df[["Report_Number", "EMPI", "Report_Date_Time"]].drop_duplicates())}')
    notes_df['Report_Date_Time'] = pd.to_datetime(notes_df['Report_Date_Time'])
    return notes_df

def join_notes_icd(notes_df,  relevant_icd):
    joined_notes_icd = notes_df.set_index('EMPI').join(relevant_icd.set_index('EMPI'), how='left').reset_index()
    print(f'There are {len(joined_notes_icd.index)} rows in the joined DF')
    joined_notes_icd['diff_days'] = (joined_notes_icd['Report_Date_Time'] - joined_notes_icd['Date']) / np.timedelta64(1, 'D')
    joined_notes_icd['abs_diff_days'] = abs((joined_notes_icd['Report_Date_Time'] - joined_notes_icd['Date']) / np.timedelta64(1, 'D'))

    joined_notes_icd_anytime = joined_notes_icd.loc[~pd.isnull(joined_notes_icd['abs_diff_days'])].sort_values('abs_diff_days')
    joined_notes_icd_anytime = joined_notes_icd_anytime.groupby(['EMPI', 'Report_Number', 'Report_Date_Time']).head(1)
    print('Number of notes with any PPH-related ICD codes: ', len(joined_notes_icd_anytime.index))
    return joined_notes_icd

def filter_notes_icd(joined_notes_icd, pph_definition='butwick'):
    joined_notes_icd_14days = joined_notes_icd.loc[(joined_notes_icd['diff_days'] <= 14) & (joined_notes_icd['diff_days'] >= -14)]
    joined_notes_icd_14days_onecode = joined_notes_icd_14days.sort_values('abs_diff_days').groupby(['EMPI','Report_Number', 'Report_Date_Time']).head(1)
    print('Number of notes with ICD codes <= 14 days from note timestamp: ', len(joined_notes_icd_14days_onecode.index))
    #print('Number with nonzero time diff: ', len(joined_notes_icd_14days_onecode.loc[joined_notes_icd_14days_onecode['diff_days'] != 0]))
    
    if pph_definition == 'zheutlin':
        zheutlin_codes = ['666.32','666.24','666.22','666.20','666.14', '666.12', '666.10', '666.04', '666.02', '666.00', '665.24', '665.22', '665.20', '656.03', '656.00', '641.91', '641.81', '641.33', '641.31', '641.30', '641.11']
        pph_icd_14days = joined_notes_icd_14days.loc[joined_notes_icd_14days['Code'].isin(zheutlin_codes)]
        pph_icd_14days_onecode = pph_icd_14days.sort_values('abs_diff_days').groupby(['EMPI','Report_Number', 'Report_Date_Time']).head(1)
        print('Number of notes with ICD codes <= 14 days from note timestamp that meet Zheutlin definition: ', len(pph_icd_14days_onecode.index))
        print(f'14 days - There are {len(pph_icd_14days["Report_Number"].unique())} unique report numbers for {len(pph_icd_14days["EMPI"].unique())} patients with PPH according to zheutlin definition.')

    else:
        #filter to only include those notes that have pregnancy-related ICD codes according to Butwick/Goffman definition
        pph_icd_14days = joined_notes_icd_14days.loc[(joined_notes_icd_14days['Code'].str.startswith('666.0')) | (joined_notes_icd_14days['Code'].str.startswith('666.1')) | (joined_notes_icd_14days['Code'].str.startswith('666.2'))]
        pph_icd_14days_onecode = pph_icd_14days.sort_values('abs_diff_days').groupby(['EMPI','Report_Number', 'Report_Date_Time']).head(1)
        print(f'14 days - There are {len(pph_icd_14days["Report_Number"].unique())} unique notes for {len(pph_icd_14days["EMPI"].unique())} patients with PPH according to butwick/goffman definition.')

    return pph_icd_14days


def main():
    # get ICD codes 
    icd = get_icd()

    # get PPH ICD codes
    relevant_icd = get_pph_icd_codes(icd)
    
    # read in delivery notes
    notes_df = get_delivery_notes()
    
    # join notes to PPH ICD codes
    joined_notes_icd = join_notes_icd(notes_df,  relevant_icd)
    
    # filter to only include those notes that have pregnancy-related ICD codes according to Zheutlin definition within 14 days of the note
    filtered_pph_icd_14days = filter_notes_icd(joined_notes_icd, pph_definition='zheutlin')
    filtered_pph_icd_14days = filtered_pph_icd_14days.sample(frac = 1, random_state=42)  
    print(f'The length of filtered_pph_icd_14days is {len(filtered_pph_icd_14days.index)}')

    # filter to notes with inpatient in ICD "Inpatient_Outpatient" column
    filtered_pph_icd_14days = filtered_pph_icd_14days.loc[filtered_pph_icd_14days['Inpatient_Outpatient'] == 'Inpatient']
    print(f'The length of filtered_pph_icd_14days is {len(filtered_pph_icd_14days.index)} after removing notes where ICD code is listed as outpatient.')

    # filter to notes where diff in days between ICD code & note time is 0
    filtered_pph_icd_14days = filtered_pph_icd_14days.loc[filtered_pph_icd_14days['abs_diff_days'] == 0]
    print(f'The length of filtered_pph_icd_14days is {len(filtered_pph_icd_14days.index)} after removing notes where ICD code time diff is > 0.')


    filtered_pph_icd_14days.to_csv(config.FILTERED_DATA_DIR  / 'filtered_PPH_disch_summary_cohort.csv', index=False)


if __name__ == "__main__":
    main()
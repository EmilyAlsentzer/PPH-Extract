import pandas as pd
import re
import sys
from datetime import datetime
sys.path.append('../')
import config


def read_file(fname):
    '''
    Read file lines into list
    '''
    with open(fname, "r") as f:
        lines = f.readlines()
    return lines

def parse_notes(lines, note_type):
    '''
    Helper function to read in RPDR Notes
    '''
    notes = []
    note_text = []

    for line in lines[1:]:
        if len(re.findall('\|',line)) > 1:

            if len(note_text) > 0:
                note['Report_Text'] = ' '.join(note_text)
                notes.append(note)
            EMPI, EPIC_PMRN,MRN_Type, MRN, Report_Number, Report_Date_Time, Report_Description, Report_Status, Report_Type,Report_Text  = line.split('|')
            note={'EMPI':EMPI, 'EPIC_PMRN':EPIC_PMRN, 'MRN_Type': MRN_Type, 'MRN':MRN, 'Report_Number': Report_Number, 'Report_Date_Time':Report_Date_Time, 'Report_Description':Report_Description, 'Report_Status':Report_Status, 'Report_Type':Report_Type, 'Report_Text':Report_Text }
            note_text = []

        else:
            note_text.append(line)
    print('Total number of %s notes: %d' %(note_type, len(notes)))
    return(pd.DataFrame(notes))

def get_notes(filetype = 'Dis', verbose=False, overwrite=False):
    '''
    Read in all RPDR notes
    '''
    if (config.FILTERED_DATA_DIR / f'all_{filetype}_pre_0501_2015.csv').exists() and not overwrite:
        df = pd.read_csv(config.FILTERED_DATA_DIR / f'all_{filetype}_pre_0501_2015.csv', index_col=0)
        print(f'There are {len(df.index)} total {filetype} pre 0501_2015 notes for {len(df["EMPI"].unique())} patients. {len(df[["Report_Number", "EMPI"]].drop_duplicates())} unique EMPI/Report Numbers. {len(df[["Report_Number", "EMPI", "Report_Date_Time"]].drop_duplicates())} unique EMPI/Report Numbers/Report Time')
    else:
        all_dfs = []
        for filename in config.RAW_DATA_DIR.iterdir():
            if filename.suffix == '.txt' and filetype in str(filename):
                if verbose: print(f'Loading {filename}...')
                lines = read_file(str(filename))
                df = parse_notes(lines, 'Discharge Summary')
                all_dfs.append(df)
        if len(all_dfs) >0:
            df = pd.concat(all_dfs)
            df = df.sort_values('EMPI')
            df['Report_Date_Time'] = pd.to_datetime(df['Report_Date_Time'] )
            df = df.loc[df['Report_Date_Time'] < datetime(2015, 5, 1)]
            df.to_csv(config.FILTERED_DATA_DIR / f'all_{filetype}_pre_0501_2015.csv')
            return df
        else:
            print('ERROR - No files read in')
            return None

def get_icd_codes(filetype = 'Dia.txt', verbose=False, overwrite=False):
    if (config.FILTERED_DATA_DIR / f'all_{filetype}.csv').exists() and not overwrite:
        df = pd.read_csv(config.FILTERED_DATA_DIR / f'all_{filetype}.csv', index_col=0)
        print(f'There are {len(df.index)} total {filetype} for {len(df["EMPI"].unique())} patients. {len(df[["Report_Number", "EMPI"]].drop_duplicates())} unique EMPI/Report Numbers. {len(df[["Report_Number", "EMPI", "Report_Date_Time"]].drop_duplicates())} unique EMPI/Report Numbers/Report Time')
        return df
    else:
        all_dfs = []
        for filename in config.RAW_ICD_DATA_DIR.iterdir():
            if filename.suffix == '.txt' and filetype in str(filename):
                if verbose: print(f'Loading {filename}...')
                df = pd.read_csv(filename, sep='|') 
                all_dfs.append(df)
        if len(all_dfs) >0:
            df = pd.concat(all_dfs)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.loc[df['Date'] < datetime(2015, 5, 1)] # filter diagnoses to those before 05/01/2015
            df = df.sort_values('EMPI')
            df.to_csv(config.FILTERED_DATA_DIR / f'all_{filetype}.csv')
            return df
        else:
            print('ERROR - No files read in')
            return None


def get_demographics(filetype = 'Dem.txt', verbose=True, overwrite=False):
    if (config.FILTERED_DATA_DIR / f"all_{filetype.replace('.txt', '')}.csv").exists() and not overwrite:
        df = pd.read_csv(config.FILTERED_DATA_DIR / f"all_{filetype.replace('.txt', '')}.csv", index_col=0)
        print(f'There are {len(df.index)} total {filetype} for {len(df["EMPI"].unique())} patients. {len(df[["Report_Number", "EMPI"]].drop_duplicates())} unique EMPI/Report Numbers. {len(df[["Report_Number", "EMPI", "Report_Date_Time"]].drop_duplicates())} unique EMPI/Report Numbers/Report Time')
        return df
    else:
        all_dfs = []
        for filename in config.RAW_DEMOGRAPHIC_DATA_DIR.iterdir():
            if filename.suffix == '.txt' and filetype in str(filename):
                if verbose: print(f'Loading {filename}...')
                df = pd.read_csv(filename, sep='|') 
                if verbose: print(f'The file contains {len(df.index)} rows for {len(df["EMPI"].unique())} patients.')
                all_dfs.append(df)
        if len(all_dfs) >0:
            df = pd.concat(all_dfs)
            df = df.sort_values('EMPI')

            df.to_csv(config.FILTERED_DATA_DIR / f"all_{filetype.replace('.txt', '')}.csv")
            return df
        else:
            print('ERROR - No files read in')
            return None
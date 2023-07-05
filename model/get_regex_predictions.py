import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import re
from ast import literal_eval
import ast

sys.path.append('../')

# our imports
import config
from model.accelerate_zero_shot import calc_metrics, calc_ie_metrics
from model.accelerate_zero_shot import clean_predictions
from model.data_utils import *
from preprocess.annotation.annotation_utils import clean_str


label_to_regex_map = {
    'c-section': r'(c-section|cesarean section|c section|cesarean\s*delivery|c\/s|pltcs)',
    'uterine atony': r'(?<!no )(uterine atony|atony)', 
    'hemabate/carboprost' : r'(hemabate|carboprost)',
    'methergine/methylergometrine': r'(methergine|methylergonovine)',
    'cytotec/misoprostol - as uterotonic': r'(cytotec|misoprostol)',
    'placenta previa': r'(?<!no )(placenta\s*previa)',
    'placenta accreta spectrum': r'(?<!no )(placenta\s*accreta|placenta\s*percreta|placenta\s*increta)',
    'retained products of conception': r'(retained poc|retained placenta)',
    'disseminated intravascular coagulation (DIC)': r'(disseminated intravascular coagulation| dic )',
    'abruption of the placenta' : r'(?<!no )(placental abruption|abruptio placenta|abruptio placentae)',
    'hysterectomy' : r'(hysterectomy|removal of uterus|c-hyst)',
    'dilation & curettage' : r'(dilation (?:and|&) curettage|dilation (?:and|&) evacuation|d&c| D and C | D and E | d&e | d\+c | d\+e )',
    'manual extraction of placenta': r'(manual (?:removal|extraction) of (?:the )?placenta)',
    'Uterine rupture': r'(uterine\s*rupture|uterine\s*dehiscence)',
    'laceration': r'(?<!no )(laceration|tear)(?!.{0,10}no)',
    'cryo':  r'(cryo)',
    'platelets': r'((?<![0-9]{2} )(?<![0-9]{3} )platelets(?!.{0,10}[0-9]{2,6}))',
    'ffp': r'(ffp|plasma)',
    'rbc': r'(rbc|blood cells|red blood)',
    'Bakri Balloon': r'(bakri balloon|bakri)',
    'B-Lynch sutures': r'(b-lynch|blynch)',
    'O\'Leary sutures': r'(o\'leary|oleary)',
    'PPH':  r'(?<!hx of )(?<!no )(pphemorrhage|pph |postpartum\s*hemorrhage|post-partum\s*hemorrhage)',  
    'PPH - surgical causes': r'((?:uter|vagina).{0,10}extension|extension.{0,10}(?:uter|vagina)|artery.{0,10}lac|ligament.{0,10}lac|lac.{0,10}ligament|lac.{0,10}artery)',
    'vaginal - vacuum extraction': r'(vacuum\s*extraction)', 
    'vaginal - forceps': r'(forceps)',
    'vaginal - spontaneous vertex':   r'(spontaneous\s*vertex|vaginal\s*delivery:\s*spontaneous|svd)', 
    'estimated blood loss': r'(?:(?:ebl|pph|hemorrhage|blood loss).{0,10}?([0-9]{3,5}|one|two|three|four|five|six) ?(?:ml|liters| l |cc)?)|(?:([0-9]{3,5}|one|two|three|four|five|six) ?(?:ml|liters| l |cc)?.{0,10}?(?:ebl|pph|hemorrhage|blood loss))|(?: ([0-9])(?![0-9]) (?:liters| l )?.{0,5}?(?:ebl|pph|hemorrhage|blood loss))|(?:ebl|pph|hemorrhage|blood loss).{0,5}?(?: ([0-9])(?![0-9]) ?(?:liters| l )?)'
}

def main():
    parser = argparse.ArgumentParser(description='Extract PPH information with regexes')
    parser.add_argument('--label', type=str) 
    parser.add_argument('--dataset_type', type=str, default=None) 
    parser.add_argument('--output_filename', type=str) #config.MODEL_RUN_DIR / args.label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') / (f'regex_preds_{args.label}').replace('/', '_')
    parser.add_argument('--input_filename', type=str, help='Filename containing csv of annotated notes')
    args = parser.parse_args()
    
    # get notes 
    annotations, notes, joined_notes = prep_data(verbose=False)
    notes = pd.read_csv(config.ANNOTATED_DATA_DIR / args.input_filename.replace('/', '_') ) 

    # Get labels
    if args.label == 'estimated blood loss':
        notes = prep_notes_for_ie_label(joined_notes, notes, args.label, clean_span=False)
        clean_spans = clean_predictions('estimated blood loss', notes['spans'].to_list(), clean_type='annotation')
        notes['clean_spans'] = clean_spans
    else:
        notes = prep_binary_labels(annotations, notes, args.label)
    print(f'There are {len(notes.index)} notes')

    # get save filename
    if dataset_type != None: output_filename = Path(str(args.output_filename ) + '_' + dataset_type)
    else: output_filename = args.output_filename
    
    # assign predictions via regex
    if args.label in config.binary_labels: #binary labels
        preds_df = notes[['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type', args.label, 'Report_Text']]
        preds_df['predictions'] = 0
        preds_df.loc[preds_df['Report_Text'].str.contains(label_to_regex_map[args.label], flags=re.IGNORECASE), 'predictions'] = 1
        
        metrics = calc_metrics(preds_df[args.label], preds_df['predictions'])

    else: # IE labels
        colname = 'clean_spans' if 'clean_spans' in notes.columns else 'spans'
        if 'clean_spans' in notes.columns: preds_df = notes[['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type', 'spans', 'clean_spans', 'Report_Text']]
        else: preds_df = notes[['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type', 'spans', 'Report_Text']]
        preds_df['predictions'] = preds_df['Report_Text'].apply(lambda s:  re.findall(label_to_regex_map[args.label], s, flags=re.IGNORECASE))
        preds_df['predictions'] = preds_df['predictions'].apply(lambda l: [str(i) for s in l for i in s if i !='']  )
        preds = clean_predictions('estimated blood loss', preds_df['predictions'].to_list())
        preds_df['clean_predictions'] = preds

        metrics = calc_ie_metrics(preds_df[colname].tolist(),  preds_df['clean_predictions'].tolist())
    

    # save preds & metrics to file
    preds_df.drop(columns=['Report_Text']).to_csv(str(output_filename) + '.csv', index=False)
    with open(str(output_filename) + '_metrics.json', 'w') as f:
        json.dump(metrics, f)


    
if __name__ == "__main__":
    main()
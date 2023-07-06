import torch
from torch.utils.data import DataLoader

from collections import defaultdict, OrderedDict
import pandas as pd
import sys
from tqdm import tqdm
import time
import argparse 
import datetime as dt
import json
import numpy as np
import re
from ast import literal_eval
from pathlib import Path
from datetime import datetime

# Huggingface
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
from evaluate import load
from accelerate import Accelerator
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, classification_report

# Our library
sys.path.append('../')
import config
from  preprocess.annotation.annotation_utils import clean_str
from model.data_utils import *

#Huggingface metric
exact_match_metric = load("exact_match")


#######################################################
# perform zero-shot inference

def predict(accelerator, model, tokenizer, notes, prompt, batch_size, max_new_tokens, note_first, max_length, split_prompt=False, verbose=False):
    if note_first:
        input_ids = tokenizer(notes['Report_Text'].tolist(), [prompt] * len(notes['Report_Text']),
            return_tensors="pt", truncation="only_first", padding='max_length', return_overflowing_tokens=True,
        stride=128,return_offsets_mapping=True)  # max_length=512,
    else:
        input_ids = tokenizer([prompt] * len(notes['Report_Text']),notes['Report_Text'].tolist(), 
            return_tensors="pt", truncation="only_second", padding='max_length', return_overflowing_tokens=True,
        stride=128,return_offsets_mapping=True, max_length=max_length)  # max_length=512,
    chunk_to_note_map = input_ids.pop("overflow_to_sample_mapping")
    input_ids = input_ids.input_ids
    
    accelerator.print('input_ids', input_ids.shape)

    dataloader = DataLoader(
        input_ids, shuffle=False, batch_size=batch_size 
    )
    chunk_dataloader = DataLoader(
        chunk_to_note_map, shuffle=False, batch_size=batch_size 
    )

    accelerator.print('chunk_to_note_map', chunk_to_note_map)
    accelerator.print('len', len(dataloader))

    model, dataloader, chunk_dataloader = accelerator.prepare(model, dataloader, chunk_dataloader)
    assert len(dataloader) == len(chunk_dataloader)

    predictions = defaultdict(list)

    all_times = []
    t_start = time.time()
    total_n_examples = 0
    for data, note_idx in tqdm(zip(dataloader, chunk_dataloader), disable=not accelerator.is_local_main_process, total=len(dataloader)):
        t0 = time.time()
        total_n_examples += len(note_idx)

        if accelerator.state.deepspeed_plugin.zero_stage == 3:
            outputs = model.generate(data, max_new_tokens = max_new_tokens, synced_gpus=True)
        else:
            outputs = model.generate(data, max_new_tokens = max_new_tokens) 
        outputs = accelerator.pad_across_processes(outputs)
        note_idx = accelerator.pad_across_processes(note_idx)

        outputs = accelerator.gather(outputs)
        note_idx = accelerator.gather(note_idx)

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, n in enumerate(note_idx.cpu()):
            predictions[n.item()].append(preds[i])
        t1 = time.time()
        print(f'It took {t1-t0}s to process a single batch of size {batch_size}')
        all_times.append(t1-t0)
        
    t_end = time.time()
    print(f'It took {np.mean(all_times)}s on average to process a single batch of size {batch_size}')
    print(f'It took {t_end - t_start}s total to process all {batch_size * len(dataloader)} or {total_n_examples} messages')

    return predictions



#######################################################
# Helper functions for evaluation

def clean_predictions(label, preds, clean_type='prediction', verbose=True):
    '''
    Clean up model predictions for the IE tasks
    '''
    preds = [[clean_str(p, label, clean_type=clean_type, verbose=verbose) for p in pred_list if clean_str(p, label, clean_type=clean_type, verbose=verbose) != ''] for pred_list in preds] 

    if label == 'time of delivery' or 'date' in label or label == 'last menstrual period' or label == 'estimated blood loss': # sometimes a single string contains multiple times/dates
        preds = [[p.split('|') for p in pred_list ] for pred_list in preds] 
        preds = [[i for p in pred_list for i in p ] for pred_list in preds] 

    # remove extra whitespace
    preds = [[p.strip() for p in pred_list ] for pred_list in preds] 
    return preds


def format_preds(args, predictions_dict, pred_type):
    '''
    Helper function to format the predictions returned from the model according to whether the prediction task is a binary task or information extraction task
    '''
    if pred_type == 'binary':
        predictions_dict = OrderedDict(sorted(predictions_dict.items())) #ensures the predictions are ordered by note_id
        predictions_binary = [1 if 'Yes' in preds or 'yes' in preds else 0 for note_id, preds in predictions_dict.items()]
        raw_preds = [ preds for note_id, preds in predictions_dict.items()]

        return predictions_binary, raw_preds
    elif pred_type == 'extraction':
        predictions_dict = OrderedDict(sorted(predictions_dict.items())) #ensures the predictions are ordered by note_id

        #filter out unanswerable & get unique values
        preds = [list(np.unique([p for p in preds if p != 'unanswerable'])) for note_id, preds in predictions_dict.items()]

        # perform label-specific filtering
        clean_preds = clean_predictions(args.label, preds)
        clean_preds = [list(np.unique(p)) for p in clean_preds]
        return clean_preds, preds


#######################################################
# IE task evaluation

def calc_ie_metrics(labels, clean_preds):
    '''
    Calculate metrics for the information extraction labels
    '''
    total_found = 0
    total_labels = 0
    total_predictions = 0
    total_notes_complete_recall = 0
    total_exact_match = 0
    total_with_annotation = 0
    for l, p in zip(labels, clean_preds):
        try:
            intersect = set(l).intersection(set(p)) 
        except:
            print('l', l, 'p', p)
        total_found += len(intersect)
        total_labels += len(l)
        total_predictions += len(p)
        if len(intersect) == len(l): total_notes_complete_recall +=1 
        if len(set(l).difference(set(p))) == 0 and len(set(p).difference(set(l))) == 0: total_exact_match += 1
        if len(l) > 0: total_with_annotation += 1
    recall_labels = total_found/total_labels if total_labels != 0 else None # # retrieved relevant / # relevant
    precision_labels = total_found/total_predictions if total_predictions != 0 else None # # retrieved relevant / # retrieved

    recall_notes = total_notes_complete_recall/len(labels)
    exact_match_notes = total_exact_match/len(labels)
    metrics = {'recall_labels': recall_labels, 'precision_labels': precision_labels, 'recall_notes': recall_notes, 'exact_match_notes': exact_match_notes, 'total_notes_with_true_annotation': total_with_annotation}
    
    return metrics   

def evaluate_extracted_preds(accelerator, args, notes, predictions_dict, filename, prompt):
    labels = notes['spans'].tolist()
    labels = [np.unique(labs) for labs in labels]

    clean_preds, preds = format_preds(args, predictions_dict, 'extraction')

    # calc metrics
    metrics = calc_ie_metrics(labels, clean_preds)

    # save metrics to file
    if not args.no_save:
        with open(str(filename) + '_metrics.json', 'w') as f:
            json.dump(metrics, f)

    # save predictions to file
    preds_df = notes[['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type', 'spans']]
    preds_df['predictions'] = preds
    preds_df['clean_predictions'] = clean_preds
    preds_df['prompt'] = prompt
    accelerator.print('predictions', preds_df.head())
    if not args.no_save:
        preds_df.to_csv(str(filename) + '.csv', index=False)


#######################################################
# Binary task evaluation

def calc_metrics(labels, predictions_binary, verbose=True):
    '''
    Calculate metrics for the binary labels
    '''
    acc = accuracy_score(labels, predictions_binary)
    macro_f1 = f1_score(labels, predictions_binary, average='macro')
    micro_f1 = f1_score(labels, predictions_binary, average='micro')
    binary_f1 = f1_score(labels, predictions_binary, average='binary')

    precision = precision_score(labels, predictions_binary)
    recall = recall_score(labels, predictions_binary)

    if verbose: print(f'acc: {acc:0.3f}, micro f1: {micro_f1:0.3f} macro f1: {macro_f1:0.3f} binary_f1: {binary_f1:0.3f} precision: {precision:0.3f} recall: {recall:0.3f}')

    if verbose: print(classification_report(labels, predictions_binary))
    try:
        tn, fp, fn, tp  = confusion_matrix(labels, predictions_binary).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn)
    except:
        #print('Exception')
        #print(labels, predictions_binary)
        tn, fp, fn, tp  = -1, -1, -1, -1
        specificity, sensitivity, precision, npv =  -1, -1, -1, -1

    if verbose: print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn} ')
    if verbose: print(f'sensitivity/recall: {sensitivity:0.3f}, specificity: {specificity:0.3f} precision: {precision:0.3f}, NPV: {npv:0.3f}')

    # save metrics to file
    metrics = {'acc': float(acc), 'micro_f1': float(micro_f1), 'macro_f1': float(macro_f1), 'binary_f1': float(binary_f1), 'precision': float(precision), 'recall': float(recall),
                'specificity':float(specificity), 'NPV':float(npv), 'tp': int(tp), 'tn': int(tn), 'fp':int(fp), 'fn':int(fn), 'n':  int(tp) + int(fn)  }
    return metrics

def evaluate_binary_preds(accelerator, args, notes, predictions_dict, filename, prompt):
    '''
    Evaluate binary prediction tasks
    '''
    labels = notes[args.label].tolist()
    predictions_binary, raw_preds = format_preds(args, predictions_dict, 'binary')

    metrics = calc_metrics(labels, predictions_binary)
    if not args.no_save:
        with open(str(filename) + '_metrics.json', 'w') as f:
            json.dump(metrics, f)

    # save predictions to file
    preds_df = notes[['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type', args.label]]
    preds_df['predictions'] = predictions_binary
    preds_df['raw_preds'] = raw_preds
    preds_df['prompt'] = prompt

    accelerator.print('predictions', preds_df.head())
    if not args.no_save:
        preds_df.to_csv(str(filename) + '.csv', index=False)


#######################################################
# Get prompt & output filename

def get_prompt(accelerator, label, prompt_desc):
    accelerator.print('label', label)
    prompt = None
    if label == 'delivery_note':
        if prompt_desc == 'disch_summary': prompt = f'Answer the following yes/no question. Does this discharge summary note mention a woman\'s delivery?\n note: '
        else: raise Exception(f'Missing Prompt for {label}')
    elif label == 'cryo':
        if prompt_desc == 'transfusion': prompt = f'Answer the following yes/no question. Does the following discharge summary mention transfusion of cryoprecipitate (cryo) during the current delivery? \n note: '
        else: prompt_label = 'cryoprecipitate (cryo)'
    elif label == 'rbc':
        if prompt_desc == 'transfusion': prompt = f'Answer the following yes/no question. Does the following discharge summary mention transfusion of packed red blood cells (pRBCs) during the current delivery? \n note: '
        else: prompt_label = 'packed red blood cells'
    elif label == 'platelets':
        if prompt_desc == 'transfusion': prompt = f'Answer the following yes/no question. Does the following discharge summary mention transfusion of platelets during the current delivery? \n note: '
        else: prompt_label = 'platelets'
    elif label == 'ffp':
        if prompt_desc == 'transfusion': prompt = f'Answer the following yes/no question. Does the following discharge summary mention transfusion of frozen fresh plasma (ffp) during the current delivery? \n note: '
        else: prompt_label = 'fresh frozen plasma (FFP)'
    elif label == 'c-section':
        prompt_label = 'a cesarean section'
    elif label == 'vaginal - spontaneous vertex':
        prompt_label = 'a spontaneous vertex vaginal delivery'
    elif label == 'vaginal - forceps':
        prompt_label = 'a vaginal delivery with forceps'
    elif label == 'Bakri Balloon':
        prompt_label = 'a bakri balloon placement'
    elif label == 'hysterectomy':
        prompt_label = 'a hysterectomy'
    elif label == 'Uterine rupture':
        if prompt_desc == 'uterus_was':  prompt = f'Answer the following yes/no question. Does the following discharge summary mention that the uterus was ruptured during the current delivery? \n note: '
        elif prompt_desc == 'rupture_of':  prompt = f'Answer the following yes/no question. Does the following discharge summary mention rupture (dehiscence) of the uterus during the current delivery? \n note: '
        else:
            prompt_label = 'uterine rupture'
    elif label == 'PPH':
        prompt_label = 'postpartum hemorrhage'
        if prompt_desc == 'disch_summary_curr_delivery': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman had postpartum hemorrhage (PPH) during her current delivery? \n note: '
        elif prompt_desc == 'disch_summary_curr_delivery_V2': prompt = f'Answer the following yes/no question. Does the following discharge summary mention postpartum hemorrhage (PPH) during the current delivery? \n note: '
        elif prompt_desc == 'disch_summary_curr_delivery_V3': prompt = f'Answer the following yes/no question. Does the following discharge summary mention postpartum hemorrhage (PPH) during the current delivery? \n discharge summary: '
        elif prompt_desc == 'disch_summary': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman had postpartum hemorrhage? \n note: '
        elif prompt_desc == 'disch_summary_v2': prompt = f'Answer the following yes/no question. Does the following discharge summary mention postpartum hemorrhage? \n note: '
        elif prompt_desc == 'disch_summary_v3': prompt = f'Answer the following yes/no question. Does the following discharge summary mention postpartum hemorrhage (PPH)? \n note: '
        else: raise Exception(f'Missing Prompt for {label}')
    elif label == 'PPH - surgical causes':
        if prompt_desc == 'surgical_complication': prompt = f'Answer the following yes/no question. Does the following discharge summary mention a surgical complication during the current delivery? \n note: '
        elif prompt_desc == 'extension': prompt = f'Answer the following yes/no question. Does the following discharge summary mention extension of the uterine incision during the current delivery? \n note: '
        elif prompt_desc == 'cutting_artery': prompt = f'Answer the following yes/no question. Does the following discharge summary mention cutting of the uterine artery during the current delivery? \n note: '
        elif prompt_desc == 'damage_broad_ligament': prompt = f'Answer the following yes/no question. Does the following discharge summary mention damage to the broad ligament during the current delivery? \n note: '
        else: prompt_label = 'postpartum hemorrhage due to surgical causes'
    elif label == 'laceration':
        prompt_label = 'a laceration'
    elif label == 'uterine atony':
        prompt_label = 'uterine atony'
    elif label == 'abruption of the placenta':
        prompt_label = 'an abruption of the placenta'
    elif label == 'manual extraction of placenta':
        prompt_label = 'manual extraction of the placenta'
    elif label == 'placenta accreta spectrum':
        if prompt_desc == 'placenta_percreta': prompt = f'Answer the following yes/no question. Does the following discharge summary mention placenta percreta during the current delivery? \n note: '
        elif prompt_desc == 'placenta_increta': prompt = f'Answer the following yes/no question. Does the following discharge summary mention placenta increta during the current delivery? \n note: '
        else: prompt_label = 'placenta accreta'  
    elif label == 'retained products of conception':
        if prompt_desc == 'only_retained_poc': prompt =  f'Answer the following yes/no question. Does the following discharge summary mention retained POCs (products of conception) during the current delivery? \n note: '
        else: prompt_label = 'retained products of conception'  
    elif label == 'dilation & curettage':
        if prompt_desc == 'curretage': prompt = f'Answer the following yes/no question. Does the following discharge summary mention the procedure curretage (D&C) during the current delivery? \n note: '
        else: prompt_label = 'a dilation and curettage'    
    elif label == 'vaginal - vacuum extraction':
        prompt_label = 'a vaginal delivery with vaccum extraction'
    elif label == "O'Leary sutures":
        prompt_label = "O'Leary sutures"    
    elif label == 'placenta previa':
        prompt_label = 'placenta previa'
    elif label == 'disseminated intravascular coagulation (DIC)':
        if prompt_desc == 'coagulopathy': prompt = f'Answer the following yes/no question. Does the following discharge summary mention coagulopathy during the current delivery? \n note: '
        elif prompt_desc == 'low_platelets': prompt = f'Answer the following yes/no question. Does the following discharge summary mention low platelets during the current delivery? \n note: '
        else: prompt_label = 'disseminated intravascular coagulation (DIC)'
    elif label == 'cytotec/misoprostol - as uterotonic':
        if prompt_desc == 'disch_summary_curr_delivery': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman was given cytotec (misoprostol) for atony during her current delivery? \n note: '
        elif prompt_desc == 'disch_summary_curr_delivery_V2': prompt = f'Answer the following yes/no question. Does the following discharge summary mention cytotec (misoprostol) for atony during the current delivery? \n note: '
        elif prompt_desc == 'disch_summary': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman was given cytotec (misoprostol) for atony? \n note: '
        elif prompt_desc == 'default': prompt = f'Answer the following yes/no question. Did the patient receive cytotec (misoprostol) for atony in the following note? \n note: '
        elif prompt_desc == 'disch_summary_V3': prompt = f'Answer the following yes/no question. Does the following discharge summary mention that the patient was given the drug Cytotec (Misoprostol) for atony? \n note: '
        elif prompt_desc == 'given_for_PPH': prompt = f'Answer the following yes/no question. Does the following discharge summary mention that the patient was given the drug Cytotec (Misoprostol) for pph? \n note: '
        elif prompt_desc == 'after_delivery': prompt = f'Answer the following yes/no question. Does the following discharge summary mention that the patient was given the drug Cytotec (Misoprostol) after the delivery? \n note: '
        else: raise Exception(f'Missing Prompt for {label}')
    elif label == 'hemabate/carboprost':
        if prompt_desc == 'disch_summary_curr_delivery': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman was given hemabate or carboprost during her current delivery? \n note: '
        elif prompt_desc == 'disch_summary_curr_delivery_V2': prompt = f'Answer the following yes/no question. Does the following discharge summary mention hemabate or carboprost during the current delivery? \n note: '
        elif prompt_desc == 'disch_summary': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman was given hemabate or carboprost? \n note: '
        elif prompt_desc == 'default': prompt = f'Answer the following yes/no question. Did the patient receive hemabate or carboprost in the following note? \n note: '
        elif prompt_desc == 'hemabate_parenthesis_drug': prompt = f'Answer the following yes/no question. Does the following discharge summary mention the drug Hemabate (Carboprost)? \n note: '
        else: raise Exception(f'Missing Prompt for {label}')
    elif label == 'methergine/methylergometrine':
        if prompt_desc == 'disch_summary_curr_delivery': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman was given methergine or methylergometrine during her current delivery? \n note: '
        elif prompt_desc == 'disch_summary_curr_delivery_V2': prompt = f'Answer the following yes/no question. Does the following discharge summary mention methergine or methylergometrine during the current delivery? \n note: '
        elif prompt_desc == 'disch_summary': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman was given methergine or methylergometrine? \n note: '
        elif prompt_desc == 'default': prompt = f'Answer the following yes/no question. Did the patient receive methergine in the following note? \n note: '
        elif prompt_desc == 'methergine_parenthesis_drug': prompt = f'Answer the following yes/no question. Does the following discharge summary mention the drug Methergine (Methylergometrine)? \n note: '
        else: raise Exception(f'Missing Prompt for {label}')
    elif label == 'estimated blood loss':
        if prompt_desc == 'default': prompt = 'Extract the estimated blood loss in the following note. If you can\'t find the answer, please respond "unanswerable".\n note:' 
        elif prompt_desc == 'disch_summary': prompt = 'Extract the estimated blood loss in the following discharge summary. If you can\'t find the answer, please respond "unanswerable".\n note:' 
        elif prompt_desc == 'disch_summary_V2': prompt = 'Extract the estimated blood loss (EBL) in the following discharge summary. If you can\'t find the answer, please respond "unanswerable".\n note:' 
        elif prompt_desc == 'disch_summary_current_delivery': prompt = 'Extract the estimated blood loss (EBL) from the current delivery in the following discharge summary. If you can\'t find the answer, please respond "unanswerable".\n note:' 
        else: raise Exception(f'Missing Prompt for {label}')
    elif label == 'transfusion amount':
        prompt = 'Extract the transfusion amount in the following note. If you can\'t find the answer, please respond "unanswerable".\n note:' 
    elif label == 'transfusion type':
        prompt = 'What is being transfused in the following note? If you can\'t find the answer, please respond "unanswerable".\n note:' 
    else:
        raise NotImplementedError

    # get model type
    if label in config.binary_labels:
        prediction_type = 'binary' 
        if prompt is None:
            if prompt_desc == 'disch_summary': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman had {prompt_label}? \n note: '
            elif prompt_desc == 'disch_summary_curr_delivery': prompt = f'Answer the following yes/no question. Does this discharge summary note mention that the woman had {prompt_label} during her current delivery? \n note: '
            elif prompt_desc == 'default':  prompt = f'Answer the following yes/no question. Does the following note describe {prompt_label}? \n note: ' 
            elif prompt_desc == 'disch_summary_curr_delivery_V2':  prompt = f'Answer the following yes/no question. Does the following discharge summary mention {prompt_label} during the current delivery? \n note: '
            else: raise Exception(f'Missing Prompt for {label}')
    else:
        prediction_type = 'extraction'   
    return prompt, prediction_type

def get_output_fname(args):
    output_filename = args.output_directory / args.label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')  / (f'zero_shot_{args.label}_model_{args.model}_{args.prompt_desc}').replace('/', '_')
    if args.dataset_type != None: output_filename = Path(str(output_filename) + f'_dataset_type={args.dataset_type}')
    if args.debug: output_filename = Path(str(output_filename) + f'_debug={args.debug}')
    return output_filename

#######################################################

def main():
    parser = argparse.ArgumentParser(description='Extract PPH information')
    parser.add_argument('--label', type=str, default='c-section', help='Which PPH concept to extract.') 
    parser.add_argument('--model', type=str, default='xxl', help='Flan-T5 model size to use.') 
    parser.add_argument('--prompt_desc', type=str,  help='Which prompt to use.') 
    parser.add_argument('--debug', action='store_true', help='Whether to subset the notes to 5 for debugging purposes.') 
    parser.add_argument('--verbose', action='store_true', help='Whether to include verbose output.')
    parser.add_argument('--no_save', action='store_true', help='Whether to forgo saving metrics and predictions to file')
    parser.add_argument('--unlabelled', action='store_true', help='Whether we\'re running on unlabeled data.')
    parser.add_argument('--annotations_filename', type=str, default=None, help='Annotations dataframe.')
    parser.add_argument('--all_notes_filename', type=str, default=None, help='All notes dataframe.')
    parser.add_argument('--labeled_filename', type=str, default=None, help='Annotated dataset to run on.')
    parser.add_argument('--dataset_type', type=str, default=None, help='Description to include in the output file to describe the run.')
    parser.add_argument('--unlab_filename', type=str, default=None, help='Unlabelled dataset to run on.')
    parser.add_argument('--output_directory', type=str, default = config.MODEL_RUN_DIR, help='Output directory path. Predictions/metrics will be saved to subfolders for each label within this directory.')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=5)
    parser.add_argument('--note_first', action='store_true')
    args = parser.parse_args()
    
    #initialize Huggingface accelerator to speed up inference
    accelerator = Accelerator()
    
    # get save filename
    output_filepath = get_output_fname(args)
    accelerator.print('output_filepath', output_filepath)
    
    # get prompt & prediction type (binary or information extraction)
    prompt, prediction_type = get_prompt(accelerator, args.label, args.prompt_desc)

    if args.unlabelled: # run on unlabelled notes
        
        # get notes 
        notes = pd.read_csv(config.UNLABELLED_DATA_DIR / args.unlab_filename)
        accelerator.print(f"There are {len(notes.index)} unlabelled notes for {len(notes['EMPI'].unique())} unique EMPI and {len(notes['Report_Number'].unique())} unique report numbers")

        # subset notes if debug mode is on
        if args.debug:
            notes = notes.head(n=5)
            accelerator.print('DEBUG mode is on. Subsetting # of notes to 5')
            accelerator.print(notes, notes.columns)

        # get tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(f'google/flan-t5-{args.model}')
        model = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{args.model}")
        model.eval()
        
        # get predictions & save to file
        predictions = predict(accelerator, model, tokenizer, notes, prompt, args.batch_size, args.max_new_tokens, args.note_first, args.max_length, verbose=args.verbose)
        preds, raw_preds = format_preds(args, predictions, prediction_type)
        preds_df = notes[['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']]
        preds_df['predictions'] = preds
        preds_df['raw_preds'] = raw_preds
        if not args.no_save:
            preds_df.to_csv(str(filename_prefix) + '.csv', index=False)
            
            
    else: # run on notes with gold-standard annotations

        # get data
        annotations, notes, joined_notes = prep_data(annotations_filepath=args.annotations_filename, notes_filepath=args.all_notes_filename)
        if args.labeled_filename is not None:
            notes = pd.read_csv(config.ANNOTATED_DATA_DIR / args.labeled_filename.replace('/', '_') ) 
            print(f"There are {len(notes.index)}  notes to evaluate for {len(notes['EMPI'].unique())} unique EMPI and {len(notes['Report_Number'].unique())} unique report numbers")
        else:
            raise Exception
        
        # get labels
        if args.label == 'delivery_note':
            notes_one_lab = notes.rename(columns={'Is_Delivery_Note': 'delivery_note'})
        else:
            if prediction_type == 'extraction':
                notes_one_lab = prep_notes_for_ie_label(joined_notes, notes, args.label)
                accelerator.print('labels', notes_one_lab['spans'].tolist())
            else:
                notes_one_lab = prep_binary_labels(annotations, notes, args.label)

        # subset notes if debug mode is on
        if args.debug:
            accelerator.print('DEBUG mode is on. Subsetting # of notes to 5')
            notes_one_lab = notes_one_lab.head(n=5) 
            accelerator.print(notes_one_lab, len(notes_one_lab), notes_one_lab.columns)

        # get tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(f'google/flan-t5-{args.model}')
        model = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{args.model}")
        model.eval()

        # get predictions
        predictions = predict(accelerator, model, tokenizer, notes_one_lab, prompt, args.batch_size, args.max_new_tokens, args.note_first, args.max_length, verbose=args.verbose)

        # evaluate model predictions
        if prediction_type =='binary':
            evaluate_binary_preds(accelerator, args, notes_one_lab, predictions, .output_filepath, prompt)
        else: #IE
            evaluate_extracted_preds(accelerator, args, notes_one_lab, predictions, output_filepath, prompt)

if __name__ == "__main__":
    main()
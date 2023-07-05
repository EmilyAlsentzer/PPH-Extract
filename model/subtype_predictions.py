import pandas as pd
import sys
from collections import Counter
sys.path.append('../')
import config
from evaluation.eval_utils import get_ebl_csection_preds

'''
Get subtype predictions based on extracted PPH concepts

NOTE: You will need to manually modify the file paths in get_term_predictions to run this file. You may also want to get rid of the runtype argument



# Nonspecific terms that we're not currently using in the subtypes
# - Bakri Balloon
# - O'Leary
# - hysterectomy
# - B-Lynch sutures
# - placenta previa
# - abruption of the placenta
'''

def get_term_predictions(path_to_preds, runtype):
    # read in predictions for all NLP terms important to each subtype    

    atony_preds = pd.read_csv(path_to_preds /'uterine_atony' / f'zero_shot_preds_uterine atony_model_xxl_disch_summary_curr_delivery_V2_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=['raw_preds'])
    methergine_preds = pd.read_csv(path_to_preds/'methergine_methylergometrine' / f'zero_shot_preds_methergine_methylergometrine_model_xxl_methergine_parenthesis_drug_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    hemabate_preds = pd.read_csv(path_to_preds / 'hemabate_carboprost' / f'zero_shot_preds_hemabate_carboprost_model_xxl_hemabate_parenthesis_drug_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    cytotec_preds = pd.read_csv(path_to_preds /'cytotec_misoprostol_-_as_uterotonic' / f'zero_shot_preds_cytotec_misoprostol - as uterotonic_model_xxl_after_delivery_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=['raw_preds'])

    uterine_rupture_preds = pd.read_csv(path_to_preds /'Uterine_rupture' / f'zero_shot_preds_Uterine rupture_model_xxl_rupture_of_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    laceration_preds = pd.read_csv(path_to_preds/'laceration' / f'zero_shot_preds_laceration_model_xxl_disch_summary_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    surgical_pph_extension_preds = pd.read_csv(path_to_preds /'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_extension_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    surgical_pph_damage_artery_preds = pd.read_csv(path_to_preds/'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_cutting_artery_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    surgical_pph_damage_ligament_preds = pd.read_csv(path_to_preds /'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_damage_broad_ligament_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])

    dic_preds = pd.read_csv(path_to_preds / 'disseminated_intravascular_coagulation_DIC' / f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_combined_dic_coagulopathy_low_platelet_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    cryo_preds = pd.read_csv(path_to_preds /'cryo' / f'zero_shot_preds_cryo_model_xxl_transfusion_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    ffp_preds = pd.read_csv(path_to_preds /'ffp' / f'zero_shot_preds_ffp_model_xxl_transfusion_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    platelets_preds = pd.read_csv(path_to_preds  / 'platelets' / f'zero_shot_preds_platelets_model_xxl_transfusion_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    oleary_preds = pd.read_csv(path_to_preds  / "O'Leary_sutures" / f"zero_shot_preds_O'Leary sutures_model_xxl_disch_summary_curr_delivery_V2_{runtype}_unlabelled.csv").set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])

    retained_poc_preds = pd.read_csv(path_to_preds / 'retained_products_of_conception' / f'zero_shot_preds_retained products of conception_model_xxl_only_retained_poc_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    d_and_c_preds = pd.read_csv(path_to_preds /'dilation_&_curettage' / f'zero_shot_preds_dilation & curettage_model_xxl_curretage_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    manual_extraction_preds = pd.read_csv(path_to_preds /'manual_extraction_of_placenta' / f'zero_shot_preds_manual extraction of placenta_model_xxl_disch_summary_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    placenta_accreta_preds = pd.read_csv(path_to_preds / 'placenta_accreta_spectrum' / f'zero_shot_preds_placenta accreta spectrum_model_xxl_combined_retained_poc_accreta_increta_percreta_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds', 'raw_preds_percreta', 'raw_preds_increta', 'predictions_increta', 'predictions_percreta',  'predictions_accreta'])

    if runtype == 'subtype_annotated':
        csection_preds = pd.read_csv(path_to_preds /'c-section' / f'zero_shot_preds_c-section_model_xxl_disch_summary_curr_delivery_V2_{runtype}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type']).drop(columns=[ 'raw_preds'])
    elif runtype == 'model_predicted_PPH':
        EBL_preds, csection_preds = get_ebl_csection_preds(all_preds=True, clean_ebl_preds = False)
        # filter csection preds to only include those with an EMPI in the above dfs
        manual_extraction_preds_reset = manual_extraction_preds.reset_index()
        csection_preds = csection_preds.loc[(csection_preds['EMPI'].isin(manual_extraction_preds_reset['EMPI'].unique())) & (csection_preds['Report_Number'].isin(manual_extraction_preds_reset['Report_Number'].unique())) & (csection_preds['Report_Date_Time'].isin(manual_extraction_preds_reset['Report_Date_Time'].unique()))]
        csection_preds = csection_preds.rename(columns={'c-section_prediction': 'predictions'})
        assert len(csection_preds.index) == len(manual_extraction_preds.index)
        csection_preds = csection_preds.set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
    else:
        raise Exception
    
    ########################
    # join all predictions into one dataframe
    joined_preds = atony_preds.join(methergine_preds, how='left', rsuffix='_methergine')
    joined_preds = joined_preds.join(cytotec_preds, how='left', rsuffix='_cytotec')
    joined_preds = joined_preds.join(hemabate_preds, how='left', rsuffix='_hemabate')
    joined_preds = joined_preds.join(placenta_accreta_preds, how='left', rsuffix='_accreta')
    joined_preds = joined_preds.join(retained_poc_preds, how='left', rsuffix='_retained_poc')
    joined_preds = joined_preds.join(d_and_c_preds, how='left', rsuffix='_d_and_c')
    joined_preds = joined_preds.join(manual_extraction_preds, how='left', rsuffix='_manual_extraction')
    joined_preds = joined_preds.join(uterine_rupture_preds, how='left', rsuffix='_uterine_rupture')
    joined_preds = joined_preds.join(dic_preds, how='left', rsuffix='_dic')
    joined_preds = joined_preds.join(cryo_preds, how='left', rsuffix='_cryo')
    joined_preds = joined_preds.join(ffp_preds, how='left', rsuffix='_ffp')
    joined_preds = joined_preds.join(platelets_preds, how='left', rsuffix='_platelets')
    joined_preds = joined_preds.join(laceration_preds, how='left', rsuffix='_laceration')
    joined_preds = joined_preds.join(oleary_preds, how='left', rsuffix='_oleary')
    joined_preds = joined_preds.join(surgical_pph_extension_preds, how='left', rsuffix='_PPH_surgical_extension') 
    joined_preds = joined_preds.join(surgical_pph_damage_artery_preds, how='left', rsuffix='_PPH_surgical_damage_artery')
    joined_preds = joined_preds.join(surgical_pph_damage_ligament_preds, how='left', rsuffix='_PPH_surgical_damage_ligament')
    joined_preds = joined_preds.join(csection_preds, how='left', rsuffix='_csection')


    joined_preds = joined_preds.rename(columns={'predictions': 'predictions_atony'}).reset_index()
    return joined_preds

def get_tone_predictions(joined_preds, verbose=True):
    '''
    Uterine atony OR methergine/methylergonovine OR hemabate/carboprost OR cytotec/misoprostol for atony (after time of delivery)
    '''
    joined_preds['has_tone_subtype_model'] = joined_preds[['predictions_atony', 'predictions_methergine', 'predictions_cytotec', 'predictions_hemabate']].sum(axis=1)
    if verbose: print(Counter(joined_preds['has_tone_subtype_model']))
    predicted_tone = joined_preds.loc[joined_preds['has_tone_subtype_model'] >= 1]
    if verbose: print(f'There are {len(predicted_tone.index)} notes predicted as tone PPH')
    return joined_preds

def get_tissue_predictions(joined_preds, verbose=True):
    '''
    placenta accreta spectrum OR retained products of conception OR dilation and curettage OR manual extraction of placenta 
    '''
    joined_preds['has_tissue_subtype_model'] = 0
    joined_preds.loc[joined_preds['predictions_accreta'] == 1, 'has_tissue_subtype_model'] = 1
    joined_preds.loc[joined_preds['predictions_retained_poc'] == 1, 'has_tissue_subtype_model'] = 1
    joined_preds.loc[joined_preds['predictions_d_and_c'] == 1, 'has_tissue_subtype_model'] = 1
    joined_preds.loc[(joined_preds['predictions_manual_extraction'] == 1) & (joined_preds['predictions_csection'] == 0), 'has_tissue_subtype_model'] = 1
    if verbose: print(Counter(joined_preds['has_tissue_subtype_model']))
    predicted_tissue = joined_preds.loc[joined_preds['has_tissue_subtype_model'] >= 1]
    if verbose: print(f'There are {len(predicted_tissue.index)} notes predicted as tissue PPH')
    return joined_preds

def get_thrombin_predictions(joined_preds, verbose=True):
    '''
    DIC OR cryo administration OR platelets administration OR FFP administration
    '''
    joined_preds['has_thrombus_subtype_model'] = joined_preds[['predictions_dic', 'predictions_cryo', 'predictions_platelets', 'predictions_ffp']].sum(axis=1)
    if verbose: print(Counter(joined_preds['has_thrombus_subtype_model']))
    predicted_thrombus = joined_preds.loc[joined_preds['has_thrombus_subtype_model'] >= 1]
    if verbose: print(f'There are {len(predicted_thrombus.index)} notes predicted as thrombus PPH')

    return joined_preds
    
def get_trauma_predictions(joined_preds, verbose=True):
    '''
    uterine rupture OR PPH due to surgical causes OR O'leary stitches OR laceration (if no other subtype is valid)
    '''

    joined_preds['has_trauma_subtype_model_lacerations_no_other_cause'] = 0
    joined_preds.loc[joined_preds['predictions_uterine_rupture'] == 1, 'has_trauma_subtype_model_lacerations_no_other_cause'] = 1
    joined_preds.loc[joined_preds['predictions_oleary'] == 1, 'has_trauma_subtype_model_lacerations_no_other_cause'] = 1
    joined_preds.loc[joined_preds['predictions_PPH_surgical_extension'] == 1, 'has_trauma_subtype_model_lacerations_no_other_cause'] = 1
    joined_preds.loc[joined_preds['predictions_PPH_surgical_damage_artery'] == 1, 'has_trauma_subtype_model_lacerations_no_other_cause'] = 1
    joined_preds.loc[joined_preds['predictions_PPH_surgical_damage_ligament'] == 1, 'has_trauma_subtype_model_lacerations_no_other_cause'] = 1

    joined_preds.loc[(joined_preds['predictions_laceration'] == 1) & (joined_preds['has_thrombus_subtype_model'] == 0) & (joined_preds['has_tone_subtype_model'] == 0) & (joined_preds['has_tissue_subtype_model'] == 0), 'has_trauma_subtype_model_lacerations_no_other_cause'] = 1
    predicted_trauma = joined_preds.loc[joined_preds['has_trauma_subtype_model_lacerations_no_other_cause'] >= 1]
    if verbose: print(f'There are {len(predicted_trauma.index)} notes predicted as trauma PPH (if we include lacerations with no other PPH cause)')

    return joined_preds




def main():
    parser = argparse.ArgumentParser(description='Generate subtype predictions.')
    parser.add_argument('--path_to_label_predictions', type=str, default=config.MODEL_RUN_DIR, help='Path to where the predictions from zero_shot_predictions.py are saved') 
    parser.add_argument('--output_filename', type=str,  help='Filename where predictions should be saved') 
    parser.add_argument('--runtype', type=str, default='model_predicted_PPH'  help='Prediction run type', options=['model_predicted_PPH', 'subtype_annotated']) 

    args = parser.parse_args()
    joined_preds = get_term_predictions(args.path_to_label_predictions, args.runtype)

    ########################
    # get subtype predictions
    # TONE
    joined_preds = get_tone_predictions(joined_preds)
    # TISSUE
    joined_preds = get_tissue_predictions(joined_preds)
    # THROMBIN
    joined_preds = get_thrombin_predictions(joined_preds)
    # TRAUMA
    joined_preds = get_trauma_predictions(joined_preds)

    ########################
    # convert predictions to binary
    joined_preds.loc[joined_preds['has_tone_subtype_model'] >= 1, 'has_tone_subtype_model'] = 1
    joined_preds.loc[joined_preds['has_tissue_subtype_model'] >= 1, 'has_tissue_subtype_model'] = 1
    joined_preds.loc[joined_preds['has_thrombus_subtype_model'] >= 1, 'has_thrombus_subtype_model'] = 1
    joined_preds.loc[joined_preds['has_trauma_subtype_model_lacerations_no_other_cause'] >= 1, 'has_trauma_subtype_model_lacerations_no_other_cause'] = 1

    ########################
    # save model subtype predictions to file
    joined_preds.to_csv(args.output_filename)
        
        
if __name__ == "__main__":
    main()




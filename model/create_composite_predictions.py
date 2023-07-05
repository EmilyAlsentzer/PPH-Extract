import sys
import json
import pandas as pd
sys.path.append('../')

import config
from model.accelerate_zero_shot import calc_metrics

'''
The goal of this script is to generate composite predictions across Flan-T5 runs with different prompts. We generate these composite predictions for only a few of the PPH concepts.

NOTE: This script requires manually modifying the paths to point to the outputs from the Flan-T5 runs for each prompt
'''


def create_composite_pas_predction(save_fname, placenta_accreta_preds, placenta_percreta_preds, placenta_increta_preds, unlabelled=False): 
    joined_preds = placenta_accreta_preds.join(placenta_percreta_preds, how='left', rsuffix='_percreta')
    joined_preds = joined_preds.join(placenta_increta_preds, how='left', rsuffix='_increta').reset_index()
    joined_preds = joined_preds.rename(columns={'predictions': 'predictions_accreta'})

    joined_preds['predictions'] = joined_preds[[ 'predictions_accreta', 'predictions_percreta', 'predictions_increta']].sum(axis=1)

    joined_preds.loc[joined_preds['predictions'] >= 1, 'predictions'] = 1
    print('Sanity Check: ', len(placenta_accreta_preds.index), len(joined_preds.index))
    print(joined_preds[['EMPI',  'predictions_accreta', 'predictions_percreta', 'predictions_increta', 'predictions']])

    joined_preds.to_csv(str(save_fname) + '.csv', index=False)

    if not unlabelled: 
        metrics = calc_metrics(joined_preds['placenta accreta spectrum'], joined_preds['predictions'])
        with open(str(save_fname) + '.json', 'w') as f:
            json.dump(metrics, f)
 
def create_composite_dic_predction(save_fname, dic_preds, coagulopathy_preds, low_platelets_preds,unlabelled=False):
    joined_preds = dic_preds.join(coagulopathy_preds, how='left', rsuffix='_coagulopathy')
    joined_preds = joined_preds.join(low_platelets_preds, how='left', rsuffix='_low_platelets').reset_index()
    joined_preds = joined_preds.rename(columns={'predictions': 'predictions_dic'})

    joined_preds['predictions'] = joined_preds[['predictions_dic', 'predictions_coagulopathy', 'predictions_low_platelets']].sum(axis=1)
    joined_preds.loc[joined_preds['predictions'] >= 1, 'predictions'] = 1

    joined_preds.to_csv(str(save_fname) + '.csv', index=False)

    if not unlabelled: 
        metrics = calc_metrics(joined_preds["disseminated intravascular coagulation (DIC)"], joined_preds['predictions'])
        with open(str(save_fname) + '.json', 'w') as f:
            json.dump(metrics, f)   
                    
def create_composite_surgical_pph_predction(save_fname, extension_preds, artery_damage_preds, ligament_damage_preds, unlabelled=False):
    joined_preds = extension_preds.join(artery_damage_preds, how='inner', rsuffix='_bleeding_artery')
    joined_preds = joined_preds.join(ligament_damage_preds, how='left', rsuffix='_ligament_damage').reset_index()
    joined_preds = joined_preds.rename(columns={'predictions': 'predictions_extension'})

    joined_preds['predictions'] = joined_preds[['predictions_extension', 'predictions_bleeding_artery', 'predictions_ligament_damage']].sum(axis=1)
    joined_preds.loc[joined_preds['predictions'] >= 1, 'predictions'] = 1
    joined_preds.to_csv(str(save_fname) + '.csv', index=False)

    if not unlabelled: 
        metrics = calc_metrics(joined_preds["PPH - surgical causes"], joined_preds['predictions'])
        with open(str(save_fname) + '.json', 'w') as f:
            json.dump(metrics, f)   
                    
def create_composite_manual_extraction_prediction(save_fname, manual_extraction_preds, csection_preds, unlabelled=False):
    joined_preds = manual_extraction_preds.join(csection_preds, how='left', rsuffix='_csection')
    joined_preds = joined_preds.rename(columns={'predictions': 'predictions_manual_extraction'}).reset_index()
    print('Sanity Check: ', len(manual_extraction_preds.index),  len(csection_preds.index), len(joined_preds.index))
    
    joined_preds['predictions'] = 0
    joined_preds.loc[(joined_preds['predictions_manual_extraction'] == 1) & (joined_preds['predictions_csection'] == 0), 'predictions'] = 1

    joined_preds.to_csv(str(save_fname) + '.csv', index=False)

    if not unlabelled: 
        metrics = calc_metrics(joined_preds["manual extraction of placenta"], joined_preds['predictions'])
        with open(str(save_fname) + '.json', 'w') as f:
            json.dump(metrics, f)   
  
  
 ##################################


###################################
# PLACENTA ACCRETA SPECTRIM

# labelled predictions
dataset='test' 
save_fname = config.MODEL_RUN_DIR / 'placenta_accreta_spectrum' / f'zero_shot_preds_placenta accreta spectrum_model_xxl_combined_retained_poc_accreta_increta_percreta_prompt_{dataset}'
placenta_accreta_preds = pd.read_csv(config.MODEL_RUN_DIR /'placenta_accreta_spectrum' / f'zero_shot_preds_placenta accreta spectrum_model_xxl_disch_summary_curr_delivery_V2_prompt_{dataset}.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type', 'placenta accreta spectrum'])
placenta_percreta_preds = pd.read_csv(config.MODEL_RUN_DIR /'placenta_accreta_spectrum' / f'zero_shot_preds_placenta accreta spectrum_model_xxl_placenta_percreta_prompt_{dataset}.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type', 'placenta accreta spectrum'])
placenta_increta_preds = pd.read_csv(config.MODEL_RUN_DIR /'placenta_accreta_spectrum' / f'zero_shot_preds_placenta accreta spectrum_model_xxl_placenta_increta_prompt_{dataset}.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type', 'placenta accreta spectrum'])
create_composite_pas_predction(save_fname, placenta_accreta_preds, placenta_percreta_preds, placenta_increta_preds)

# unlabelled predictions
run_type = 'model_predicted_PPH' 
save_fname_confirmed_PPH = config.MODEL_RUN_DIR / 'placenta_accreta_spectrum' / f'zero_shot_preds_placenta accreta spectrum_model_xxl_combined_retained_poc_accreta_increta_percreta_{run_type}_unlabelled'
placenta_accreta_preds = pd.read_csv(config.MODEL_RUN_DIR /'placenta_accreta_spectrum' / f'zero_shot_preds_placenta accreta spectrum_model_xxl_disch_summary_curr_delivery_V2_{run_type}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
placenta_percreta_preds = pd.read_csv(config.MODEL_RUN_DIR /'placenta_accreta_spectrum' /  f'zero_shot_preds_placenta accreta spectrum_model_xxl_placenta_percreta_{run_type}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
placenta_increta_preds = pd.read_csv(config.MODEL_RUN_DIR /'placenta_accreta_spectrum' /  f'zero_shot_preds_placenta accreta spectrum_model_xxl_placenta_increta_{run_type}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
create_composite_pas_predction(save_fname_confirmed_PPH, placenta_accreta_preds, placenta_percreta_preds, placenta_increta_preds, unlabelled=True)
 
    
###################################
# DIC

# labelled predictions
dataset='test' 
save_fname_confirmed_PPH = config.MODEL_RUN_DIR / 'disseminated_intravascular_coagulation_DIC' / f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_combined_dic_coagulopathy_low_platelet_prompt_{dataset}'
dic_preds = pd.read_csv(config.MODEL_RUN_DIR /'disseminated_intravascular_coagulation_DIC' / f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_disch_summary_curr_delivery_V2_prompt_{dataset}.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
coagulopathy_preds = pd.read_csv(config.MODEL_RUN_DIR /'disseminated_intravascular_coagulation_DIC' /  f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_coagulopathy_prompt_{dataset}.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type' ])
low_platelets_preds = pd.read_csv(config.MODEL_RUN_DIR /'disseminated_intravascular_coagulation_DIC' /  f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_low_platelets_prompt_{dataset}.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
create_composite_dic_predction(save_fname_confirmed_PPH, dic_preds, coagulopathy_preds, low_platelets_preds, unlabelled=False)

# unlabelled predictions
run_type = 'model_predicted_PPH' #confirmed_PPH_EBL subtype_annotated
save_fname = config.MODEL_RUN_DIR / 'disseminated_intravascular_coagulation_DIC' / f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_combined_dic_coagulopathy_low_platelet_{run_type}_unlabelled'
dic_preds = pd.read_csv(config.MODEL_RUN_DIR /'disseminated_intravascular_coagulation_DIC' / f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_disch_summary_curr_delivery_V2_{run_type}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
coagulopathy_preds = pd.read_csv(config.MODEL_RUN_DIR /'disseminated_intravascular_coagulation_DIC' /  f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_coagulopathy_{run_type}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type' ])
low_platelets_preds = pd.read_csv(config.MODEL_RUN_DIR /'disseminated_intravascular_coagulation_DIC' /  f'zero_shot_preds_disseminated intravascular coagulation (DIC)_model_xxl_low_platelets_{run_type}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type' ])
create_composite_dic_predction(save_fname, dic_preds, coagulopathy_preds, low_platelets_preds, unlabelled=True)

##################################
# PPH - surgical causes

# labelled predictions
dataset='test' 
save_fname = config.MODEL_RUN_DIR / 'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_combined_surgical_PPH_prompt_{dataset}'
extension_preds = pd.read_csv(config.MODEL_RUN_DIR / 'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_extension_prompt_{dataset}.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
artery_damage_preds = pd.read_csv(config.MODEL_RUN_DIR /'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_cutting_artery_prompt_{dataset}.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
ligament_damage_preds = pd.read_csv(config.MODEL_RUN_DIR /'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_damage_broad_ligament_prompt_{dataset}.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
create_composite_surgical_pph_predction(save_fname, extension_preds, artery_damage_preds, ligament_damage_preds)

# unlabelled predictions
dataset='model_predicted_PPH'
save_fname = config.MODEL_RUN_DIR / 'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_combined_surgical_PPH_{dataset}_unabelled'
extension_preds = pd.read_csv(config.MODEL_RUN_DIR / 'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_extension_{dataset}_unlabelled.csv')
print('lengths', len(extension_preds.index), len(extension_preds.drop_duplicates().index))
extension_preds = extension_preds.set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
artery_damage_preds = pd.read_csv(config.MODEL_RUN_DIR /'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_cutting_artery_{dataset}_unlabelled.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
ligament_damage_preds = pd.read_csv(config.MODEL_RUN_DIR /'PPH_-_surgical_causes' / f'zero_shot_preds_PPH - surgical causes_model_xxl_damage_broad_ligament_{dataset}_unlabelled.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
create_composite_surgical_pph_predction(save_fname, extension_preds, artery_damage_preds, ligament_damage_preds, unlabelled=True)


##################################
# Manual Extraction of placenta

# labelled predictions
dataset= 'test' 
save_fname = config.MODEL_RUN_DIR / 'manual_extraction_of_placenta' / f'zero_shot_preds_manual extraction of placenta_model_xxl_composite_prompt_{dataset}'
manual_extraction_preds = pd.read_csv(config.MODEL_RUN_DIR / 'manual_extraction_of_placenta' / f'zero_shot_preds_manual extraction of placenta_model_xxl_disch_summary_prompt_{dataset}.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
csection_preds = pd.read_csv(config.MODEL_RUN_DIR / 'c-section' / f'zero_shot_preds_c-section_model_xxl_disch_summary_curr_delivery_V2_prompt_{dataset}.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
create_composite_manual_extraction_prediction(save_fname, manual_extraction_preds, csection_preds)

# unlabelled predictions
dataset='model_predicted_PPH' 
save_fname = config.MODEL_RUN_DIR / 'manual_extraction_of_placenta' / f'zero_shot_preds_manual extraction of placenta_model_xxl_composite_{dataset}_unlabelled'
manual_extraction_preds = pd.read_csv(config.MODEL_RUN_DIR / 'manual_extraction_of_placenta' / f'zero_shot_preds_manual extraction of placenta_model_xxl_disch_summary_{dataset}_unlabelled.csv').set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
csection_preds = pd.read_csv(config.MODEL_RUN_DIR / 'c-section' / f'zero_shot_preds_c-section_model_xxl_disch_summary_curr_delivery_V2_{dataset}.csv').drop_duplicates().set_index(['Report_Number', 'EMPI', 'Report_Date_Time', 'MRN_Type'])
create_composite_manual_extraction_prediction(save_fname, manual_extraction_preds, csection_preds)



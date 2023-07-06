from pathlib import Path 

#TODO: modify this path to be the path where you want data/results to go
PROJECT_DIR = Path('/home/ema51')

# Path to model outputs
MODEL_RUN_DIR = PROJECT_DIR / 'model_runs/PPH' 

# Path to data folders
DATA_DIR = PROJECT_DIR / 'data'
FILTERED_DATA_DIR = DATA_DIR / 'filtered' 
ANNOTATED_DATA_DIR = DATA_DIR / 'annotated'
UNLABELLED_DATA_DIR = DATA_DIR / 'unlabelled'
ANALYSIS_DIR = DATA_DIR / 'analysis'
RAW_DIR = DATA_DIR / 'raw'

# path to Prancer configs and data
PRANCER_LABEL_DIR = PROJECT_DIR / 'containerdata/config'
PRANCER_DATA_DIR = PROJECT_DIR / 'containerdata/data'

# binary & information extraction PPH labels
binary_labels = [ 'delivery_note', 'Bakri Balloon', 'c-section', 'vaginal - spontaneous vertex', 'laceration', 'uterine atony', 'abruption of the placenta', 
                    'methergine/methylergometrine', 'manual extraction of placenta', 'PPH', 'placenta accreta spectrum', 'dilation & curettage', 'hysterectomy',
                     'vaginal - vacuum extraction', 'Uterine rupture',  'cytotec/misoprostol - as uterotonic', 'hemabate/carboprost',
                      "O'Leary sutures", 'placenta previa', 'disseminated intravascular coagulation (DIC)', 'PPH - surgical causes', 'vaginal - forceps', 
                      'cryo', 'ffp', 'platelets', 'rbc', 'retained products of conception']
ie_labels = ['estimated blood loss'] 


from pathlib import Path
import re
import pandas as pd
import sys
sys.path.append('../')
sys.path.append('../..')

import config

'''
Create CSV of preannotations for all txt files in config.PRANCER_DATA_DIR
'''

def get_annotations(contents, annotations, regex, label, ngroups=1, verbose=False):
    '''
    Find all annotations given the provided regex for the provided label
    '''
    if verbose:
        print('\nregex: ', regex)
    all_matches = re.finditer(regex, contents, flags=re.IGNORECASE)
    if all_matches is not None:
        for match in all_matches:
            if verbose: print('match', match)
            groups = [(n+1, match.group(n+1)) for n  in range(ngroups)]
            groups_ind = [n for n, g in groups if g is not None]
            if verbose: print('groups', groups,'groups_ind', groups_ind)
            start,end = match.span(groups_ind[0])
            annot = {'start': start, 'end': end, 'cui': label}
            if verbose: print('annot', annot)
            annotations.append(annot)

    return annotations


def suggest(filename):
    '''
    create a dataframe with suggested annotations for each label 
    '''
    annotations = []
    with open(filename) as f:
        contents = f.read()

        #dates 
        ga_wks_days = '([0-9]{1,2}w[0-9]{1,2}d)'
        annotations = get_annotations(contents, annotations, f'(?:([0-9\.-\/]+)\s+(?:weeks|wk|wks|week)\s+gestation)|(?:gestational Age:\s+{ga_wks_days})|(?:{ga_wks_days} gestation)|(?:([0-9 \-\/\+]+)(?:wk|w|wks|week|weeks) ga)', 'gestational age', ngroups=4, verbose=False) 

        time_regex = '([0-9]{1,2}:[0-9]{1,2} (?:AM|PM))'
        date_regex = '([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})'
        annotations = get_annotations(contents, annotations, f'delivery date: {date_regex}|(?:delivered.*on {date_regex})|(?:date of delivery|delivery date).*?{date_regex}', 'date of delivery', ngroups=3) 
        annotations = get_annotations(contents, annotations, f'(?: delivery time: {time_regex})|(?:delivered.*at {time_regex})|(?:time of delivery|delivery date).*?{time_regex}', 'time of delivery', ngroups=3) 
        annotations = get_annotations(contents, annotations, r'(?:edc.{0,10}?([0-9\/\-]+))|(?:estimated date of delivery.*?([0-9\/\-]+))|(?:estimated edc.*?([0-9]{1,2}\/[0-9]{1,2}\/[0-9]{1,2}))', 'estimated date of delivery', ngroups=3) 
        annotations = get_annotations(contents, annotations, r'(?:last menstrual period.{0,50}?([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4}))|(?:lmp ([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4}))', 'last menstrual period', ngroups=2) 

        # post-partum hemorrhage
        annotations = get_annotations(contents, annotations, r'(?<!hx of )(?<!no )(pphemorrhage|pph |postpartum\s*hemorrhage|post-partum\s*hemorrhage)', 'PPH')   
        annotations = get_annotations(contents, annotations, r'(?:([0-9]+) ?(?:ml)? blood loss)|(?:blood loss.*?([0-9]+)(?: *cc)?)|(?:EBL:?\s*([0-9]+)|(?:pph.{0,10}([0-9]+) ?cc))', 'estimated blood loss', ngroups=4)  

        #medications
        annotations = get_annotations(contents, annotations, r'(methergine|methylergonovine)', 'methergine/methylergometrine')
        annotations = get_annotations(contents, annotations, r'(hemabate|carboprost)', 'hemabate/carboprost')
        annotations = get_annotations(contents, annotations, r'(cytotec|misoprostol)', 'cytotec/misoprostol - as uterotonic')

        #atony & associated procedures
        annotations = get_annotations(contents, annotations, r'(?<!no )(uterine atony|atony)', 'uterine atony')   
        annotations = get_annotations(contents, annotations, r'(bakri balloon|bakri)', 'Bakri Balloon')
        annotations = get_annotations(contents, annotations, r'(b-lynch|blynch)', 'B-Lynch sutures')
        annotations = get_annotations(contents, annotations, r'(o\'leary|oleary)', 'O\'Leary sutures')

        #placental problems
        annotations = get_annotations(contents, annotations, r'(?<!no )(placenta\s*previa)', 'placenta previa')
        annotations = get_annotations(contents, annotations, r'(?<!no )(placenta\s*accreta|placenta\s*percreta|placenta\s*increta)', 'placenta accreta spectrum')  
        annotations = get_annotations(contents, annotations, r'(?<!no )(retained poc|retained products of conception)', 'retained products of conception')  

        annotations = get_annotations(contents, annotations, r'(?<!no )(placental abruption|abruptio placenta|abruptio placentae)', 'abruption of the placenta') 

        #coagulation
        annotations = get_annotations(contents, annotations, r'(disseminated intravascular coagulation| dic )', 'disseminated intravascular coagulation (DIC)') 

        # delivery type
        annotations = get_annotations(contents, annotations, r'(vacuum\s*extraction)', 'vaginal - vacuum extraction') 
        annotations = get_annotations(contents, annotations, r'(svd|spontaneous\s*vertex|vaginal\s*delivery:\s*spontaneous)', 'vaginal - spontaneous vertex') 
        annotations = get_annotations(contents, annotations, r'(forceps)', 'vaginal - forceps') 
        annotations = get_annotations(contents, annotations, r'(c/s|pltcs|c-section|cesarean section|c section|cesarean\s*delivery)', 'c-section') 

        # other procedures
        annotations = get_annotations(contents, annotations, r'(hysterectomy|removal of uterus|c-hyst)', 'hyterectomy') 
        annotations = get_annotations(contents, annotations, r'(dilation (?:and|&) curettage|dilation (?:and|&) evacuation|d&c| D and C | D and E | d&e | d\+c | d\+e |curretage)', 'dilation & curettage') 
        annotations = get_annotations(contents, annotations, r'(manual (?:removal|extraction) of (?:the )?placenta)', 'manual extraction of placenta') 
        annotations = get_annotations(contents, annotations, r'(uterine\s*rupture|uterine\s*dehiscence)', 'Uterine rupture') 

        # transfusion
        annotations = get_annotations(contents, annotations, r'(packed\s*red\s*blood\s*cells|red\s*blood\s*cells|plasma|(?<![0-9]{2} )(?<![0-9]{3} )platelets(?!.{0,10}[0-9]{2,6})|prbc|cryoprecipitate|cryo|ffp)', 'transfusion type') 
        annotations = get_annotations(contents, annotations, r'transfusion.*([0-9]) units|([0-9]) units prbc', 'transfusion amount', ngroups=2) 

        # laceration
        annotations = get_annotations(contents, annotations, r'(?<!no )(laceration|tear)(?!.{0,10}no)', 'laceration') 
        annotations = get_annotations(contents, annotations, r'(?:(?<!no )(?:tear|laceration).{0,100}?(vaginal|periurethral region|perineal)(?! delivery))|(?:(vaginal|periurethral region|perineal)(?! delivery).{0,100}(?:laceration|tear)(?!.{0,10}no))','laceration anatomy', ngroups=2, verbose=True ) 
        annotations = get_annotations(contents, annotations, r'(?:([a-zA-Z0-9]*) degree (?:laceration|tear))|(?:laceration.*?([a-zA-Z0-9]*) degrees?)', 'laceration grade',ngroups=2) 

    df = pd.DataFrame(annotations)
    return df


# NOTE: changing preannotations should only affect notes without a json file
def main():
    for filename in config.PRANCER_DATA_DIR.iterdir():
        if filename.suffix == '.txt':
            suggestions_df = suggest(filename)
            suggestions_df.to_csv(str(filename).replace('.txt', '.csv'), index=False)

if __name__ == "__main__":
    main()
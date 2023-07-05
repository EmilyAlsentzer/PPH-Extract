import pickle as pkl
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../..')

import config

'''
Creates the pkl files necessary to run PRANcER
'''

# map from annotation to category
annotations_type_dict = { 
            'PPH': 'PPH Definiton',  
            'PPH - surgical causes': 'PPH Definiton', 
            'estimated blood loss': 'PPH Definiton', 
            'transfusion type': 'Procedure', 
            'transfusion amount': 'Procedure',  
            'laceration': 'Problem', 
            'methergine/methylergometrine': 'Medication', 
            'hemabate/carboprost': 'Medication',  
            'cytotec/misoprostol - as uterotonic': 'Medication', 
            'dilation & curettage': 'Procedure', 
            'manual extraction of placenta': 'Procedure', 
            'hyterectomy': 'Procedure',
            'B-Lynch sutures': 'Procedure', 
            'O\'Leary sutures': 'Procedure', 
            'Bakri Balloon': 'Procedure',
            'placenta previa': 'Problem', 
            'placenta accreta spectrum': 'Problem', 
            'retained products of conception': 'Problem', 
            'Uterine rupture': 'Problem', 
            'abruption of the placenta': 'Problem',
            'disseminated intravascular coagulation (DIC)': 'Problem', 
            'uterine atony': 'Problem',
            'vaginal - vacuum extraction': 'Delivery Type', 
            'c-section': 'Delivery Type', 
            'vaginal - spontaneous vertex': 'Delivery Type',
            'vaginal - forceps': 'Delivery Type', 
            'not a delivery note': 'Other', 
            'Tone': 'PPH Subtype', 
            'Tissue': 'PPH Subtype', 
            'Trauma': 'PPH Subtype', 
            'Thrombin': 'PPH Subtype', 
            'Unable to determine': 'PPH Subtype'} 

def main():
    #color_lookup.pk category -> color
    colors = ['#FF4081', '#00BCD4', '#4CAF50', '#FFC107', '#FF5722', '#CDDC39', '#673AB7', '#2196F3', '#9E9E9E']
    assert len(np.unique(list(annotations_type_dict.values()))) <= len(colors), f'{len(annotations_type_dict.values())} annotations vs {len(colors)} colors'
    d = {category:color for category, color in zip(np.unique(list(annotations_type_dict.values())), colors)}
    with open(str(config.PRANCER_LABEL_DIR / "color_lookup.pk"), "wb") as output_file: 
        pkl.dump(d, output_file)

    # EXAMPLE: [['gestational age', 'gestational age', 1, ['Obstetric History'], []], ['delivery time', 'delivery time', 2, ['Dates'], []]]# list of CUIs [CUI, name, number??, [types], [synonyms]]
    umls = [[annotation, annotation, i + 1, [category]] for i , (annotation, category) in enumerate(annotations_type_dict.items())]
    # EXAMPLE: {'gestational age':[0], 'delivery time':[1]} # name -> [index] into umls list
    lookup = {annotation:[i] for i , (annotation, category) in enumerate(annotations_type_dict.items())}
    umls_lookup_snomed = [umls, lookup]
    with open(str(config.PRANCER_LABEL_DIR / "umls_lookup_snomed.pk"), "wb") as output_file: 
        pkl.dump(umls_lookup_snomed, output_file)

    suggestions = {} # text -> CUI
    with open(str(config.PRANCER_LABEL_DIR / "suggestions.pk"), "wb") as output_file:
        pkl.dump(suggestions, output_file)

        #type_tui_lookup.pk
    types = {} # type -> [tui, abbrev]
    tuis = {} # tui -> category
    type_tui_lookup = [types, tuis]
    with open(str(config.PRANCER_LABEL_DIR / "type_tui_lookup.pk"), "wb") as output_file:
        pkl.dump(type_tui_lookup, output_file)


    index_snomed = {} # word stem -> set(label_id) i.e. indices into umls list
    with open(str(config.PRANCER_LABEL_DIR / "index_snomed.pk"), "wb") as output_file:
        pkl.dump(d, output_file)




if __name__ == "__main__":
    main()
import re
import pandas as pd
from datetime import datetime
from fractions import Fraction

'''
utils used to clean up annotations
'''

def has_single_digit_or_10(s):
    '''
    check to see if string contains the number 0-10
    '''
    s = s.lower()
    return len(re.findall(r'(?<!\d)(\d|10)(?!\d)', s, flags=re.I)) > 0

def has_numbers(s):
    '''
    Check to see if string contains a number
    '''
    return len(re.findall(r'\d+', s, flags=re.I)) > 0

def is_text_number(s):
    '''
    Check if the string contains a number from 1-10 as a word (e.g. "one") or if it contains an ordinal (e.g. "first")
    '''
    s = s.lower()
    return len(re.findall(r'first|second|third|fourth|fifth|sixth|seventh|one|two|three|four|five|six|seven|eight|nine|ten', s, flags=re.I)) > 0

def clean_dates(s):
    '''
    Extract valid date
    '''
    try:
        datetime_object = datetime.strptime(s, '%m/%d/%y').date()
    except:
        try: 
            datetime_object = datetime.strptime(s, '%m/%d/%Y').date()
        except:
            try:
                datetime_object = datetime.strptime(s, '%m-%d-%y').date()
                print(f'CHECK: Converted {s} to {datetime_object}')
            except:
                try:
                    datetime_object = datetime.strptime(s, '%B %d, %Y').date()
                except:
                    try:
                        datetime_object = datetime.strptime(s, '%d %B, %Y').date()
                    except:
                        try:
                            datetime_object = datetime.strptime(s, '%d %B').date()
                            datetime_object = datetime_object.strftime('%m-%d')

                        except:
                            try:
                                datetime_object = datetime.strptime(s, '%B %d').date()
                                datetime_object = datetime_object.strftime('%m-%d')

                            except:
                                try:
                                    datetime_object = datetime.strptime(s, '%m/%d')
                                    datetime_object = datetime_object.strftime('%m-%d')
                                except:
                                    try:
                                        datetime_object = datetime.strptime(s, '%m/%y')
                                        datetime_object = datetime_object.strftime('%m-%Y')
                                    except:
                                        try:
                                            datetime_object = datetime.strptime(s, '%m/%Y')
                                            datetime_object = datetime_object.strftime('%m-%Y')
                                        except:
                                            print(f'ERROR converting {s} to a date')
                                            datetime_object = s

    return str(datetime_object)

def clean_times(s):
    '''
    Extract valid time
    '''
    found_time = True
    try:
        datetime_object = datetime.strptime(s, '%I:%M %p').time()
    except:
        try:
            datetime_object = datetime.strptime(s, '%H:%M').time()
        except:
            try:
                datetime_object = datetime.strptime(s, '%I: %M %p').time()
            except:
                try:
                    datetime_object = datetime.strptime(s, '%H:%M:%S').time()
                except:
                    try:
                        datetime_object = datetime.strptime(s, '%I %p').time()
                    except:
                        try:
                            datetime_object = datetime.strptime(s, '%I%M %p').time()
                            print(f'CHECK: Converted {s} to {datetime_object}')
                        except:
                            try:
                                if len(s) > 2: 
                                    datetime_object = datetime.strptime(s, '%H%M').time()
                                    print(f'CHECK: Converted {s} to {datetime_object}')
                                else:
                                    raise Exception
                            
                            except:
                                print(f'ERROR converting {s} to a time')
                                found_time=False
                                datetime_object = s
    return str(datetime_object), found_time

def convert_frac_to_float(frac_str):
    '''
    convert string as a fraction (e.g. 1/2) to a float
    '''
    frac_str = str(frac_str).replace(',', '')
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

number_str_to_int_map = {'one': "1000", 'two': "2000", 'three': "3000", 'four': "4000", 'five':"5000", 'six': "6000", 'seven':"7000", 'eight':"8000", 'nine':"9000", 'ten':"10000"}

def clean_str(s, label, clean_type='prediction', verbose=True):
    
    '''
    Sometimes the models generate text that doesn't *only* contain the correct information or that contains a wrong answer.
    This function cleans up the model output (or the human annotation if clean_type != 'prediction').
    '''
    
    if label == 'transfusion amount':
        numbers = re.findall(r'one|two|three|four|five|six|seven|eight|nine|ten|(?:(?<!\d|\/|\.)(?:\d|10)(?!\d|\/|\.))', s, flags=re.I) 
        if len(numbers) == 1: return numbers[0]
        if len(numbers) == 0: return ''
        if len(numbers) > 1: 
            print(f'There are multiple numbers in the string: {s}')
            return numbers[0]
    elif label == 'laceration grade':
        if not (has_single_digit_or_10(s) or is_text_number(s)): return ''
        s = re.sub(r"\-degree| degree|degree", "", s, flags=re.I)
        s = re.sub(r'(?<=[1-9])(nd|st|rd|th)', "", s, flags=re.I)
    #TODO: remove gestational ages that are extracted
    elif label == 'estimated blood loss':     #TODO: handle '1-1/2 liters; '1200 + additional 250'
        # get low hanging fruit
        number_str = r'(?:(?<![a-zA-Z])one(?![a-zA-Z])|(?<![a-zA-Z])two(?![a-zA-Z])|(?<![a-zA-Z])three(?![a-zA-Z])|(?<![a-zA-Z])four(?![a-zA-Z])|(?<![a-zA-Z])five(?![a-zA-Z])|(?<![a-zA-Z])six(?![a-zA-Z])|(?<![a-zA-Z])seven(?![a-zA-Z])|(?<![a-zA-Z])eight(?![a-zA-Z])|(?<![a-zA-Z])nine(?![a-zA-Z]))'
        numbers = re.findall(f'([\d\.\/\-]+).{1,7}?ebl|ebl.{1,7}?([\d\.\/\-]+)|(?:([\d\.\-\/\-]+) ?(?:l(?![a-zA-Z])|liters?))|(?:(\d,?\d\d\d?)+? ?(?:ml|cc|c))|(^[<>]?[\d\.\,\-\/]+$)|(^{number_str}$)|(?:({number_str}) (?:l(?![a-zA-Z])|liters?))', s, flags=re.I)
        numbers = [n for tup in numbers for n in tup if n != '']
        numbers = [re.findall(f'[\d\.\-]+') if 'ebl' in n.lower() else [n] for n in numbers  ]
        numbers = [n.split('-')[1] if n.count('-') == 1 and n.count('/') == 0 else n.replace('-', ' ')  for l in numbers for n in l  ] # we take the upper bound of a range 

        numbers = [number_str_to_int_map[n] if n in number_str_to_int_map else n for n in numbers]
        numbers = [n.replace('>', '').replace('<', '') for n in numbers if not str(n).startswith('0')]
        float_numbers = []
        for n in numbers:
            try: 
                if '/' in n: 
                    if verbose: print(f'WARNING: converting fraction {n} to float')
                    n = convert_frac_to_float(n)
                    if verbose: print(f'Result: {n}')
                float_numbers.append(float(str(n).replace(',', '')))
            except: 
                if verbose: print(f'WARNING: Cannot convert to float: {n} from "{s}"')
        numbers = float_numbers
        numbers = [n for n in numbers if n != 0]
        numbers = [n*1000 if n < 10 else n for n in numbers ] #convert to ml/cc
        if clean_type == 'prediction': numbers = [n for n in numbers if not( n >= 10 and n < 100)] #TODO: confirm we should be removing these
        # for n in numbers: 
        #     if n >= 10 and n < 100: print(f'WARNING: double digit EBL: {n}')
        if len(numbers) == 1: return str(numbers[0])
        if len(numbers) == 0: return ''
        if len(numbers) > 1: 
            if verbose: print(f'There are multiple numbers in the string: {s}')
            return '|'.join([ str(n) for n in numbers])

    elif label == 'gestational age':
        #print('start: ', s)
        if not has_numbers(s): return ''
        s = re.sub(r' weeks|weeks|wks|wk| week|week', "", s, flags=re.I)
        s = re.sub(r'\d{1,2}\/\d{1,2}\/\d{4}(?!:)', "", s, flags=re.I)
        s = re.sub(r'\d{1,2}\/\d{1,2}\/\d{2}(?!:)', "", s, flags=re.I)
        #print('end: ', s)
    elif label == 'time of delivery':
        s = re.sub(r'\d{1,2}\/\d{1,2}\/\d{4}(?!:)', " ", s, flags=re.I)
        s = re.sub(r'\d{1,2}\/\d{1,2}\/\d{2}(?!:)', " ", s, flags=re.I)
        s = re.sub(r'\d{1,2}\/\d{2,4}', " ", s, flags=re.I)
        s = re.sub(r'p\.m\.|p\.m|pm|p |p$', " PM", s, flags=re.I)
        s = re.sub(r'a\.m\.|a\.m|am|a |a$', " AM", s, flags=re.I)
        s = re.sub(r'EST|EDT|PST|PDT|CDT|CST|CT|PT|ET', "", s, flags=re.I)
        s = re.sub(r'\s+|\n', " ", s, flags=re.I)
        s = s.strip()
        times = re.findall(r'\d{1,2}\:\d{2}(?:\:\d{2})?(?: AM| PM)?|\d{3,4}|\d{1}(?: AM| PM)', s, flags=re.I)
        if len(times) == 0: return ''
        else:
            if len(times) > 1: print(f'There are multiple times in the string: {s}')
            cleaned_times = [clean_times(d) for d in times]
            cleaned_times = [t for t, found_time in cleaned_times if found_time]
            if len(cleaned_times) == 0: return ''
            s = '|'.join(cleaned_times)
    elif 'date' in label or label == 'last menstrual period':
        dates = re.findall(r'(\d{1,2}[\/-]\d{1,2}(?:[-\/]\d{4})|\d{1,2}[\/-]\d{1,2}(?:[-\/]\d{2})|\d{1,2}[-\/]\d{1,4}|(?:january|jan|february|feb|march|mar|april|apr|may|june|july|august|aug|september|sept|october|oct|november|nov|december|dec) \d{1,2}(?:,? \d{1,4})?)(?! weeks)', s, flags=re.I)
        
        if len(dates) == 0: return ''
        else:
            if len(dates) > 1: print(f'There are multiple dates in the string: {s}')
            cleaned_dates = [clean_dates(d) for d in dates]
            cleaned_dates = [d for d in cleaned_dates if d != '']
            s = '|'.join(cleaned_dates)
    elif label == 'transfusion type': 
        s = re.sub(r'units|unit|u', "", s, flags=re.I)
        s = re.sub(r'\d+', "", s, flags=re.I)
        lower_s = s.lower()
        if 'prbc' in lower_s or 'blood' in lower_s or 'cryo' in lower_s or 'plasma' in lower_s or 'ffp' in lower_s or 'platelet' in lower_s or 'rbc' in lower_s or 'rb' in lower_s or 'fibrinogen' in lower_s or 'cell' in lower_s:
            s = s 
        else:
            return ''  
    elif label == 'laceration anatomy': 
        s = re.sub(r',', "", s, flags=re.I)
        s = re.sub(r'\d+', "", s, flags=re.I)
        s = re.sub(r'degrees|degree', "", s, flags=re.I)
        s = re.sub(r'first|second|third|fourth|fifth|sixth', "", s, flags=re.I)
        s = re.sub(r'1st|2nd|3rd|4th|5th|6th', "", s, flags=re.I)
    else:
        print(f'You entered an invalid label: {label}')

    s = s.strip()
    return str(s)

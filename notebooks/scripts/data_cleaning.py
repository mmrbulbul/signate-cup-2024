# import libraries
import re
import unicodedata
import pandas as pd
import numpy as np


column_names = ['id', 'age', 'typeof_contact', 'city_tier', 'duration_of_pitch', 'occupation', 'gender', 
 'number_of_person_visiting', 'number_of_followups', 'product_pitched', 'preferred_property_star', 
 'number_of_trips', 'passport', 'pitch_satisfaction_score', 'designation', 'monthly_income', 
 'customer_info', 'prod_taken']

japanese_numerals = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
}

car_ownership_keywords = ["所持", "所有", "あり", "保有"]
no_car_ownership_keywords = ["未所持", "未所有", "なし", "保有なし"]


jpn_to_eng = {
    "結婚済み": "Married",
    "離婚済み": "Divorced",
    "独身": "Single",
    "未婚": "Unmarried"
}

def process_age(japanese_str):
    
    if isinstance(japanese_str, int) or isinstance(japanese_str, float):
        return japanese_str
    try:
        return  int(japanese_str[:-1])
    except:
        # Remove the age suffix (歳)
        if japanese_str.endswith('歳'):
            japanese_str = japanese_str[:-1]
        
        # Convert Japanese numerals to integers
        total = 0
        current_value = 0
        factor = 1
        
        for char in reversed(japanese_str):
            if char in japanese_numerals:
                num = japanese_numerals[char]
                if num == 10:
                    # When '十' is encountered, we need to handle it specially
                    if current_value == 0:
                        current_value = 10
                    else:
                        total += current_value * 10
                        current_value = 0
                    factor = 10
                else:
                    current_value += num * factor
                    factor = 1
            else:
                raise ValueError(f"Unexpected character {char} in input string.")
        
        total += current_value
        return total


def convert_duration_of_pitch(x):
    try:
        return  int(x[:-1])
    except:
        return np.nan



def extract_num_trips(text):
    try: 
        return int(text)
    except:
        try:
            # Define the regex pattern to extract the integer
            pattern = r'\d+'

            # Use re.search to find the first occurrence of the pattern
            match = re.search(pattern, text)

            if match:
                extracted_integer = int(match.group())
            if "年に" in text:
                return extracted_integer
            if "半年に" in text:
                return extracted_integer*2
            if "四半期に" in text:
                return extracted_integer*4
        except:
            return np.nan


def process_monthly_income(x):
    """
    Args: x: str
    example: 2000000, 250万円
    """
    try:
        return  float(x)
    except:
        
        pattern = r'\d+'
        match = re.search(pattern, x)

        if match:
            extracted_integer = int(match.group())
        return extracted_integer*10000


def normalize_gender(gender):
    """
    Convert full-width to half-width and convert to lowercase
    """
    normalized = unicodedata.normalize('NFKC', gender).lower().strip().replace(" ", "")
    return normalized

def process_product_pitched(x):
    try:
        if len(x)==5:
            return "basic"
        if len(x)==4:
            return "king"
        if len(x)==8:
            return "standard"
        if len(x)==12:
            return "super deluxe"
        if len(x)==6:
            return "deluxe"
    except:
        return np.nan

def process_designation(x):
    try:
        if len(x)==9:
            return "executive"
        if len(x)==7:
            return "manager"
        if len(x)==14:
            return "senior manager"
        if len(x)==3:
            return "avp"
        if len(x)==2:
            return "vp"
    except:
        return np.nan

def process_customer_info(x):

    x = x.replace("、", " ")
    x = x.replace("/", " ")
    x = x.replace("／", " ")
    x = x.replace(",", " ")
    x = x.replace("\t", " ")
    x = x.replace("\n", " ")
    x = x.replace("\u3000", " ")

    return pd.Series(x.split(" "))




def categorize_car_ownership(text):
    for keyword in car_ownership_keywords:
        if keyword in text:
            return "Car Ownership"
    for keyword in no_car_ownership_keywords:
        if keyword in text:
            return "No Car Ownership"
    return "Unknown"


 

    
# Categorize based on the number of children
def categorize_children(text):
    if "1人" in text or "1児" in text:
        return "1 Child"
    elif "2人" in text or "2児" in text:
        return "2 Children"
    elif "3人" in text or "3児" in text:
        return "3 Children"
    elif "なし" in text or "無し" in text or "ゼロ" in text or "非育児家庭" in text or "無子" in text:
        return "No Children"
    elif "子供有り" in text:
        return "has child"
    else:
        return "unknown"
 

def clean_data(df, drop_columns=["id", "customer_info"]):
#     ['typeof_contact', 'city_tier', , 'occupation', 'gender', 
#  'number_of_person_visiting', 'number_of_followups', 'product_pitched', 'preferred_property_star', 
# , 'passport', 'pitch_satisfaction_score', 'designation', 'monthly_income', 
#  'customer_info', 'prod_taken']
    # convert columns names to snake casing 
    if df.shape[1] == len(column_names):
        df.columns = column_names
    else:
        df.columns = column_names[:-1]
    df = df.copy()
    df['age'] = df['age'].apply(process_age)
    df['duration_of_pitch'] = df['duration_of_pitch'].apply(convert_duration_of_pitch)
    df['number_of_trips'] = df['number_of_trips'].apply(extract_num_trips)
    df['monthly_income'] = df['monthly_income'].apply(process_monthly_income)
    df['gender'] = df['gender'].apply(normalize_gender)
    df['product_pitched'] = df['product_pitched'].apply(process_product_pitched)
    df['designation'] = df['designation'].apply(process_designation)
    # create three series based on customer_info column
    splitted_customer_info = df['customer_info'].apply(process_customer_info)
    df["marital_status"] = splitted_customer_info[0]
    df["car_ownership"] = splitted_customer_info[1]
    df["children"] = splitted_customer_info[2]
    df['marital_status'] = df['marital_status'].map(jpn_to_eng)
    df['car_ownership'] = df['customer_info'].apply(categorize_car_ownership)
    df['children'] = df['customer_info'].apply(categorize_children)
    df.drop(columns=drop_columns, inplace=True)
    return df

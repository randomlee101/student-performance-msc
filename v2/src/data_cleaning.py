import pandas as pd
import numpy as np

dataframe = pd.read_csv('../assets/Performance_Augmented_3000.csv')
dataframe.drop(['Timestamp'], axis=1, inplace=True)
dataframe.drop_duplicates()
print(dataframe.shape)


def normalize_classes(data):
    if str(data).lower().startswith('j'):
        return "JS " + str(data)[-1]
    elif str(data).lower().startswith('s'):
        return "SS " + str(data)[-1]
    return data


def pattern_matching(data):
    if str(data).startswith("SS") and str(data)[-1].isdigit() and (int(str(data)[-1]) <= 3):
        return True
    return False


def convert_activities_to_array(data):
    return [r.strip() for r in str(data).split(",")]


def categorise_household_size(data):
    if str(data).isdigit():
        if int(data) < 4:
            return '1 - 3'
        elif int(data) < 7:
            return '4 - 6'
        else:
            return '7 - 10'
    return '1 - 3'


def highlight_postgraduate(data=""):
    if (data == 'Masters Degree') | (data == 'Phd'):
        return 'Postgraduate'
    return data


def grade_binary_categorisation(data=""):
    if data != '65 & Above':
        return '64 & Below'
    return data


def position_binary_categorisation(data=""):
    if data != 'Top 10':
        return 'Top 5'
    return data


dataframe['Class'] = dataframe['Class'].apply(normalize_classes)
dataframe['is_senior'] = dataframe['Class'].apply(pattern_matching)
dataframe = dataframe.drop(dataframe[dataframe['is_senior'] == False].index)
dataframe = dataframe.drop(dataframe[dataframe['is_senior'] == False].index)
dataframe.drop('is_senior', inplace=True, axis=1)
dataframe['After-School Activities'] = dataframe['After-School Activities'].apply(convert_activities_to_array)
dataframe = dataframe.explode('After-School Activities')
value_counts = dataframe['After-School Activities'].value_counts()
rows_to_keep = value_counts[value_counts > 1000].index
dataframe = dataframe[dataframe['After-School Activities'].isin(rows_to_keep)]
dataframe['Household Size'] = dataframe['Household Size'].apply(categorise_household_size)
dataframe['Qualification - Father'] = dataframe['Qualification - Father'].apply(highlight_postgraduate)
dataframe['Qualification - Mother'] = dataframe['Qualification - Mother'].apply(highlight_postgraduate)
dataframe['Grade'] = dataframe['Grade'].apply(grade_binary_categorisation)
dataframe['Position'] = dataframe['Position'].apply(position_binary_categorisation)

value_counts = dataframe['Age'].value_counts()
rows_to_keep = value_counts[value_counts > 2000].index
dataframe = dataframe[dataframe['Age'].isin(rows_to_keep)]

all_columns = list(dataframe.columns)


def check_index(data, arr=None):
    if arr is None:
        arr = []
    return arr.index(data)


for column in all_columns:
    print(dataframe[column].value_counts())
    unique_y = np.unique(dataframe[column]).tolist()
    dataframe[column] = dataframe[column].apply(lambda x: check_index(x, unique_y))

dataframe.to_csv('../assets/Performance_Augmented_Cleaned.csv', index=False)

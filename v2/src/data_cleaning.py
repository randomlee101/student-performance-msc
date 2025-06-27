import pandas as pd
import numpy as np

# Load the Augmented Data
dataframe = pd.read_csv('../assets/Performance_Augmented_3000.csv')
# Drop Timestamp of the Survey, it is not need
dataframe.drop(['Timestamp'], axis=1, inplace=True)
# Drop redundant rows
dataframe.drop_duplicates()
# Print the shape
print(dataframe.shape)


# Make sure Senior and Junior classes are uniformly represented
def normalize_classes(data):
    if str(data).lower().startswith('j'):
        return "JS " + str(data)[-1]
    elif str(data).lower().startswith('s'):
        return "SS " + str(data)[-1]
    return data


# Highlight the students in senior class
def pattern_matching(data):
    if str(data).startswith("SS") and str(data)[-1].isdigit() and (int(str(data)[-1]) <= 3):
        return True
    return False


# Multi-choice responses are comma separated, this is to convert them into an array
def convert_activities_to_array(data):
    return [r.strip() for r in str(data).split(",")]


# Categorise Household Sizes into ranges
def categorise_household_size(data):
    if str(data).isdigit():
        if int(data) < 4:
            return '1 - 3'
        elif int(data) < 7:
            return '4 - 6'
        else:
            return '7 - 10'
    return '1 - 3'


# Function to fuse Master's Degree and PhD to Postgraduate
def highlight_postgraduate(data=""):
    if (data == 'Masters Degree') | (data == 'Phd'):
        return 'Postgraduate'
    return data


# Split grades between 0 and 64 and 65 to 100
def grade_binary_categorisation(data=""):
    if data != '65 & Above':
        return '64 & Below'
    return data


# Split postion between Top 5 and Top 10
def position_binary_categorisation(data=""):
    if data != 'Top 10':
        return 'Top 5'
    return data


# Apply 'normalize_class' function to the 'Class' column
dataframe['Class'] = dataframe['Class'].apply(normalize_classes)
# Create an 'is_senior' column by applying 'pattern_matching' function to the 'Class' column
dataframe['is_senior'] = dataframe['Class'].apply(pattern_matching)
# Drop rows where the class is not a senior class
dataframe = dataframe.drop(dataframe[dataframe['is_senior'] == False].index)
# Drop the 'is_senior' column as it is no longer needed
dataframe.drop('is_senior', inplace=True, axis=1)
# Apply the 'convert_activities_to_array' function to the 'After-School Activities' column
dataframe['After-School Activities'] = dataframe['After-School Activities'].apply(convert_activities_to_array)
# Now that the 'After-School Activities' is an array the 'explode' method to create an entirely separate row
# of each entry
dataframe = dataframe.explode('After-School Activities')
# Count of all your 'After-School Activities'
value_counts = dataframe['After-School Activities'].value_counts()
# Find the index of rows where the 'value_counts' is greater than 1000
rows_to_keep = value_counts[value_counts > 1000].index
# Keep the rows that are greater than 1000
dataframe = dataframe[dataframe['After-School Activities'].isin(rows_to_keep)]
# Set range of household sizes
dataframe['Household Size'] = dataframe['Household Size'].apply(categorise_household_size)
# Categorise qualification to Postgraduate
dataframe['Qualification - Father'] = dataframe['Qualification - Father'].apply(highlight_postgraduate)
dataframe['Qualification - Mother'] = dataframe['Qualification - Mother'].apply(highlight_postgraduate)
# Categorise Grade
dataframe['Grade'] = dataframe['Grade'].apply(grade_binary_categorisation)
# Categorise Position
dataframe['Position'] = dataframe['Position'].apply(position_binary_categorisation)

# Keep Rows with Age greater than 10
value_counts = dataframe['Age'].value_counts()
rows_to_keep = value_counts[value_counts > 2000].index
dataframe = dataframe[dataframe['Age'].isin(rows_to_keep)]

# List all the available columns in the dataframe
all_columns = list(dataframe.columns)


# Function for number categorisation of unique items in a particular column
def check_index(data, arr=None):
    if arr is None:
        arr = []
    return arr.index(data)


# loop through the listed columns and apply the number categorisation
for column in all_columns:
    print(dataframe[column].value_counts())
    unique_y = np.unique(dataframe[column]).tolist()
    dataframe[column] = dataframe[column].apply(lambda x: check_index(x, unique_y))

# Save the augmented data
dataframe.to_csv('../assets/Performance_Augmented_Cleaned.csv', index=False)

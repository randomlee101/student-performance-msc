import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import StringLookup

dataframe = pd.read_csv('../assets/student_performance.csv')
dataframe.dropna(inplace=True)
dataframe.drop_duplicates(inplace=True)


def check_frame(d):
    frame_name = d.name
    print(d.dtype)
    if d.dtype == "object":
        # object_frames.append(frame_name)
        uniqueness = np.unique(d)
        d_lookup = StringLookup(vocabulary=uniqueness)
        d = pd.Series(d_lookup(d))
        vocab = d_lookup.get_vocabulary(include_special_tokens=False)
        # save_vocabulary_to_asset_folder(frame_name, vocab)
    return d


def check_pass_or_fail(data):
    if data > 60:
        return 1
    else:
        return 0


dataframe = dataframe.apply(check_frame)
# dataframe["math score"] = dataframe["math score"].apply(check_pass_or_fail)


# dataframe["reading score"] = dataframe["reading score"].apply(check_pass_or_fail)
dataframe["writing score"] = dataframe["writing score"].apply(check_pass_or_fail)
# print(dataframe["math score"].value_counts())
# print(dataframe["reading score"].value_counts())
# print(dataframe["writing score"].value_counts())


def train_test_split(inp, percentage=0.0):
    inp = np.array(inp)
    input_size = inp.__len__()
    test_size = round(input_size * percentage)
    train_size = input_size - test_size
    return inp[0:train_size], inp[train_size:]


# grouped_dataframe = dataframe.groupby('Pass/Fail')
# minimum_count = grouped_dataframe["Pass/Fail"].value_counts().min()
# grouped_dataframe = grouped_dataframe.apply(lambda df: df.sample(minimum_count))
# new_dataframe = grouped_dataframe.reset_index(drop=True)
# new_dataframe.dropna(inplace=True)
# new_dataframe.drop_duplicates(inplace=True)
# print(new_dataframe["Pass/Fail"].value_counts())

correlation_matrix = dataframe.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Dataset With Preprocessed Writing Score")
plt.show()

x = dataframe[["reading score", "math score", "lunch", "race/ethnicity"]]
y = dataframe["writing score"]
#
# print(x[0:3])
smote = SMOTE(sampling_strategy='auto', random_state=42)
x, y = smote.fit_resample(x, y)
# print(x[0:3])
#
#
#
print(y.value_counts())


#
#
def get_train_and_test_values(percent=0.0):
    x_train, x_test = train_test_split(x, percent)
    y_train, y_test = train_test_split(y, percent)
    return x_train, x_test, y_train, y_test
#
#
# get_train_and_test_values(0.2)

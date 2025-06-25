import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import StringLookup

dataframe = pd.read_csv('../assets/StudentPerformanceFactors.csv')
object_frames = []

dataframe = dataframe[dataframe['exam_score'] <= 100]
dataframe = dataframe[dataframe['previous_scores'] <= 100]
dataframe.dropna(inplace=True)
dataframe.drop_duplicates()


def save_vocabulary_to_asset_folder(folder_name="", vocabulary=None):
    if vocabulary is None:
        vocabulary = []
    with open(f'../assets/{folder_name}.txt', 'w') as file:
        file.write(','.join(vocabulary))
        file.close()


def train_test_split(inp, percentage=0.0):
    inp = np.array(inp)
    input_size = inp.__len__()
    test_size = round(input_size * percentage)
    train_size = input_size - test_size
    return inp[0:train_size], inp[train_size:]


def check_improvements(data):
    previous_score = data['previous_scores']
    exam_score = data['exam_score']
    if previous_score == exam_score:
        return "Neutral"
    elif previous_score < exam_score:
        return "Improved"
    elif previous_score > exam_score:
        return "Declined"
    else:
        return None


def encoding_by_mean(data, resulting_mean=0.0):
    if data < resulting_mean:
        return 0
    else:
        return 1


def check_frame(d):
    frame_name = d.name
    print(d.dtype)
    if d.dtype == "object":
        object_frames.append(frame_name)
        uniqueness = np.unique(d)
        d_lookup = StringLookup(vocabulary=uniqueness)
        d = pd.Series(d_lookup(d))
        vocab = d_lookup.get_vocabulary(include_special_tokens=False)
        save_vocabulary_to_asset_folder(frame_name, vocab)
    return d


def reduce_output_to_zero_and_one(out):
    return out - 1


dataframe["improvement"] = dataframe.apply(check_improvements, axis=1)
# dataframe = dataframe[dataframe["improvement"] != "Neutral"]
dataframe["hours_studied"] = dataframe["hours_studied"].apply(
    lambda hs: encoding_by_mean(hs, resulting_mean=dataframe["hours_studied"].mean()))
dataframe["attendance"] = dataframe["attendance"].apply(
    lambda att: encoding_by_mean(att, resulting_mean=dataframe["attendance"].mean()))
dataframe["sleep_hours"] = dataframe["sleep_hours"].apply(
    lambda sh: encoding_by_mean(sh, resulting_mean=dataframe["sleep_hours"].mean()))
dataframe["tutoring_sessions"] = dataframe["tutoring_sessions"].apply(
    lambda ts: encoding_by_mean(ts, resulting_mean=dataframe["tutoring_sessions"].mean()))
dataframe["physical_activity"] = dataframe["physical_activity"].apply(
    lambda pa: encoding_by_mean(pa, resulting_mean=dataframe["physical_activity"].mean()))
# dataframe = dataframe.select_dtypes(exclude=['int64'])
grouped_dataframe = dataframe.groupby('improvement')
minimum_count = grouped_dataframe["improvement"].value_counts().min()
grouped_dataframe = grouped_dataframe.apply(lambda df: df.sample(minimum_count))
new_dataframe = grouped_dataframe.reset_index(drop=True)
new_dataframe.dropna(inplace=True)
new_dataframe.drop_duplicates(inplace=True)
# new_dataframe.drop(["previous_scores", "exam_score"], axis=1, inplace=True)
new_dataframe = new_dataframe.sample(frac=1)
new_dataframe = new_dataframe.apply(check_frame)

print(new_dataframe["hours_studied"].value_counts())
print(new_dataframe["attendance"].value_counts())
print(new_dataframe["sleep_hours"].value_counts())
print(new_dataframe["tutoring_sessions"].value_counts())
print(new_dataframe["physical_activity"].value_counts())
x = new_dataframe.drop(['improvement'], axis=1)
# x = new_dataframe["attendance"]
# print(x.value_counts())
y = new_dataframe['improvement']
# y = y.apply(reduce_output_to_zero_and_one)

correlation_matrix = new_dataframe.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Dataset")
plt.show()


# def get_train_and_test_values(percent=0.0):
#     x_train, x_test = train_test_split(x, percent)
#     y_train, y_test = train_test_split(y, percent)
#     return x_train, x_test, y_train, y_test
#
#
# get_train_and_test_values(0.2)

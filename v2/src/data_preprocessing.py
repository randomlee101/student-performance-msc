import pandas as pd

dataframe = pd.read_csv('../assets/Performance_Augmented_Cleaned.csv')

minimum_count = dataframe['Grade'].value_counts().min()
grouped_dataframe = dataframe.groupby('Grade')
grouped_dataframe = grouped_dataframe.apply(lambda d: d.sample(minimum_count))
dataframe = grouped_dataframe.reset_index(drop=True)
dataframe.drop('Academic Balance', axis=1, inplace=True)

dataframe.to_csv('../assets/Performance_Augmented_Preprocessed.csv', index=False)

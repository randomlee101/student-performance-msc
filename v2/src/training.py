import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.metrics import Precision, Recall

dataframe = pd.read_csv('../assets/Performance_Augmented_Preprocessed.csv')
dataframe = dataframe.sample(frac=1, random_state=42)
print(dataframe.head())

x = dataframe.drop(['Grade', 'Position'], axis=1)
y = dataframe[['Grade']]

print(x.shape)
model = Sequential(
    [
        Conv1D(filters=16, kernel_size=2, input_shape=(x.shape[1], 1), activation='relu'),
        MaxPooling1D(pool_size=2, strides=1),
        Flatten(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ]
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
model.fit(x, y, validation_split=0.02, epochs=100, batch_size=32)

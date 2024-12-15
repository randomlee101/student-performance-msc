from src.preprocessing import get_train_and_test_values
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision

x_train, x_test, y_train, y_test = get_train_and_test_values(0.3)

model = Sequential(
    [
        Flatten(),
        BatchNormalization(),
        # Dense(1024, activation='relu'),
        # Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ]
)

model.compile(optimizer='adam', metrics=['accuracy', Precision()], loss='binary_crossentropy')
model.fit(x_train, y_train, validation_split=0.3, epochs=75)
model.evaluate(x_test, y_test)

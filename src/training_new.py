from src.preprocessing_2 import get_train_and_test_values
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, AUC, Recall
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

x_train, x_test, y_train, y_test = get_train_and_test_values(0.2)

model = Sequential(
    [
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
    ]
)

model.compile(optimizer='adam', metrics=['accuracy', Precision(), AUC(), Recall()], loss='binary_crossentropy')
history = model.fit(x_train, y_train, validation_split=0.3, shuffle=True, epochs=100)

model.summary()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Writing Score ANN Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# extract actual results from the dataset
cm_labels = np.reshape(np.array(y_test), (1, y_test.shape[0]))[0]
# predict the labels from the corresponding training set and round them to result to 0 or 1
predictions = np.round(model.predict(x_test))
# flatten the prediction to match the shape of the labels
predictions = np.reshape(np.array(predictions), (1, predictions.shape[0]))[0]

# plot confusion matrix with 500 sample data
cm = tf.math.confusion_matrix(cm_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Writing Score ANN Confusion Matrix')
plt.show()

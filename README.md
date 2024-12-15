## Student Performance

There are three scores with reference to SAT Scores [Reading, Writing and Math]. This project aims to analyse the
interrelationship between these factors and a few select other factors

### Preprocessing

- Preprocessing was split into three different sub datasets with each of Reading, Writing and Math as Target for each
  iteration
- The pass or fail category was achieved by applying a function to return 1 for true [pass] in the case of scores over
  60 and the alternate being 0 for false [fail]
- Correlative Matrix was developed to confirm the correlation between the existing covariates
- Due to imbalance, the data was augmented
- The data was augment using SMOTE for oversampling as undersampling renders the data too small

### Train - Test Split

- The score to be categorically analysed was selected as y while the other scores were paired with lunch and
  race/ethnicity as input based on the result of the correlative analysis .i.e. if we are to predict the math score the
  input will be [writing score, reading score, lunch, race/ethnicity]
- A function was created to split x and y into x_train, x_test, y_train, y_test

### Files To Consider

- _src/training_new.py_ - for training
- _src/preprocessing_2.py_ - for preprocessing

### Worth Mentioning

When training for `writing` the split percentage was changed to 0.2 compared to 0.3 as the model had a slight
overfitting with the 30% split

```python
x_train, x_test, y_train, y_test = get_train_and_test_values(0.2)
```

For the other models `reading` and `math` there were no dropouts required compared to that of `writing` to also prevent
overfitting

```python
model = Sequential(
    [
        ...
        Dropout(0.6),
        ...
    ]
)
```

`Sigmoid` Activation Function was used for the final layer of the Neural Network as the result was either 0 or 1 which
also meant that the loss function would be a `binary_crossentropy` which is binary as the name implies
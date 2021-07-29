from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, average_precision_score, accuracy_score


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l1(0.01)))
    model.add(layers.Dropout(0.5))
    for i in range(number_of_hidden_layers - 1):
        model.add(layers.Dense(size, activation='relu',kernel_regularizer = regularizers.l2(l2)))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def build_keras_pipeline():
    classifier = KerasClassifier(build_model, epochs=20, batch_size=256, verbose = 0, l2 = 0.1, dropout = 0.5,
                                     size = 64, number_of_hidden_layers = 3)
    col_transformer = ColumnTransformer(transformers=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0),["reviews_per_month"]),
        # ("encoder", TargetEncoder(), kategoriale_spalten)
        ('onehot', OneHotEncoder(sparse=False))
    ], remainder="passthrough")

    pipe = Pipeline(steps=[
        ("preprocessing", col_transformer),
        # ("scaler", StandardScaler()),
        ("scaling", MinMaxScaler()),
        ("model", classifier)
    ])

    random_seed()
    pipe.fit(X_train, y_train)

    # paramter nochmal optimieren
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'model__epochs': [20, 50, 100],
        'model__dropout': [0.3, 0.5],
        'model__size': [64, 128, 256],
        'model__number_of_hidden_layers': [4, 5],
        'model__l2': [0.1, 0.3],
    }
    random_seed()
    search = GridSearchCV(pipe, param_grid, cv=2, verbose=1).fit(X_train_valid, y_train_valid)
    print("mae: {}, mse: {}".format(
        mean_absolute_error(y_valid, search.predict(X_valid)),
        mean_squared_error(y_valid, search.predict(X_valid))
    ))
    print("{:.2%} wurden richtig erkannt.".format(classifier_pipe.score(X_test, y_test_binary)))


def k_fold_split():
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print(skf.get_n_splits(X, y))
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


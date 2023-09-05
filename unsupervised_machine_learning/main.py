import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)


# K-Nearest Neighbors
def knn():
    scaler = StandardScaler()
    # fits the StandardScaler on the feature data in the DataFrame df, excluding the target column 'diagnosis'.
    scaler.fit(df.drop('diagnosis', axis=1))
    # applies the scaling to the feature data and stores the result in the variable scaled_features.
    scaled_features = scaler.transform(df.drop('diagnosis', axis=1))
    # creates a new DataFrame df_scaled from the scaled feature data, using the column names from the original DataFrame df excluding the target column 'diagnosis'.
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])

    # creates an instance of the KNeighborsClassifier class
    from sklearn.neighbors import KNeighborsClassifier
    KNN_model = KNeighborsClassifier(n_neighbors=1)
    # Creates a KFold object with 100 folds and shuffled order of samples.
    kfold = KFold(n_splits=100, shuffle=True, random_state=101)
    # uses the cross_val_score function to evaluate the performance of the KNN model
    results = cross_val_score(KNN_model, scaled_features, df['diagnosis'], cv=kfold)
    # Output
    print("KNN | Accuracy: " + str(round(results.mean() * 100, 2)) + "%")

# One Class Support Vector Machines
def ocsvm():
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score

    scaler = StandardScaler()
    # fits the StandardScaler on the feature data in the DataFrame df, excluding the target column 'diagnosis'.
    scaler.fit(df.drop('diagnosis', axis=1))
    # applies the scaling to the feature data and stores the result in the variable scaled_features.
    scaled_features = scaler.transform(df.drop('diagnosis', axis=1))
    # creates a new DataFrame df_scaled from the scaled feature data, using the column names from the original DataFrame df excluding the target column 'diagnosis'.
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    X = df_scaled.values

    # Convert M to -1 and B to 1
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    Y = df['diagnosis'].values
    Y = encoder.fit_transform(Y)
    Y = np.where(Y == 0, -1, 1)

    # A KFold object is created with 100 splits, shuffled order and random seed 101 for reproducibility.
    kfold = KFold(n_splits=100, shuffle=True, random_state=101)
    cvscores = []
    # The loop performs 100 iterations of the following steps:
    for train, test in kfold.split(X):
        svm = OneClassSVM(nu=0.999, kernel='rbf')
        # The SVM model is trained on the current fold's training data using svm.fit.
        # The training data is passed in as X[train] and Y[train] where X and Y are arrays or matrices that
        # hold the features and labels respectively.
        svm.fit(X[train])
        # The model is evaluated on the current fold's test data using the accuracy_score metric,
        # the results of which are stored in the scores variable.
        y_pred = svm.predict(X[test])
        scores = accuracy_score(Y[test], y_pred)
        # The accuracy of the model, obtained from the scores variable, is multiplied by 100 and appended to the list cvscores.
        cvscores.append(scores * 100)

    # Output
    print("OneClassSVM | Accuracy: " + str(round(np.mean(cvscores), 2)) + "%")

# Convolutional Neural Network
def cnn():
    import numpy as np
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.utils import to_categorical
    scaler = StandardScaler()
    # fits the StandardScaler on the feature data in the DataFrame df, excluding the target column 'diagnosis'.
    scaler.fit(df.drop('diagnosis', axis=1))
    # applies the scaling to the feature data and stores the result in the variable scaled_features.
    scaled_features = scaler.transform(df.drop('diagnosis', axis=1))
    # creates a new DataFrame df_scaled from the scaled feature data, using the column names from the original DataFrame df excluding the target column 'diagnosis'.
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    X = df_scaled.values

    # Covert M to 1 and B to 0
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    Y = df['diagnosis'].values
    Y = encoder.fit_transform(Y)
    Y = to_categorical(Y)

    # Creates a Sequential model in Keras.
    def create_model():
        model = Sequential()
        # first layer
        model.add(Dense(32, activation='relu', input_dim=30))
        # second layer
        model.add(Dense(16, activation='relu'))
        # third layer
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # A KFold object is created with 100 splits, shuffled order and random seed 101 for reproducibility.
    kfold = KFold(n_splits=100, shuffle=True, random_state=101)
    cvscores = []
    for train, test in kfold.split(X, Y):
        model = create_model()
        # The model is trained on the current fold's training data using model.fit.
        model.fit(X[train], Y[train], epochs=10, batch_size=10, verbose=0)
        # The model is evaluated on the current fold's test data using model.evaluate.
        scores = model.evaluate(X[test], Y[test], verbose=0)
        # Accuracy
        cvscores.append(scores[1] * 100)

    # Output
    print("CNN | Accuracy: " + str(round(np.mean(cvscores), 2)) + "%")


if __name__ == "__main__":
    for i in range(20):
        knn()
    for i in range(20):
        ocsvm()
    for i in range(20):
        cnn()

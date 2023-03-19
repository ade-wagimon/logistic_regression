# logistic_regression

    # Import necessary libraries
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    # Load iris dataset
    iris = load_iris()

    # Create feature matrix and target vector
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression()

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Predict the output for the testing set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = np.mean(y_pred == y_test)

    # Print the accuracy of the model
    print("Accuracy:", accuracy)



In this example, we first load the iris dataset and create a feature matrix X and a target vector y. We then split the data into training and testing sets using the train_test_split function from scikit-learn.

Next, we create an instance of the LogisticRegression class and fit the model on the training set using the fit() method.

Finally, we predict the output for the testing set using the predict() method and calculate the accuracy of the model by comparing the predicted output with the actual output.

This is just a simple example to get you started with logistic regression in Python. For more complex datasets, you may need to preprocess the data, perform feature engineering, and tune the hyperparameters of the model to achieve better performance.

from train import train_perceptron

def test(weights, X_test):
    # Get the number of test examples
    num_test_examples = X_test.shape[0]
    predicted_labels = []

    # Make predictions for each test example
    for i in range(num_test_examples):
        # Insert bias term and get input values
        x = np.insert(X_test.iloc[i].values, 0, 1)
        
        # Make a prediction and append to the list of predicted labels
        prediction = predict(weights, x)
        predicted_labels.append(prediction)

    return predicted_labels




# Load test data from file
test_file_path = 'test_data.txt'
test_data = load_data(test_file_path)

X_test = test_data.drop(columns=['label'])
y_test = test_data[['label']]

# Train the perceptron on the training set
train_file_path = 'train_data.txt'
train_data = load_data(train_file_path)

X_train = train_data.drop(columns=['label'])
y_train = train_data[['label']]

# Train the perceptron on the training set
learned_weights = train_perceptron(X_train, y_train)

# Test the perceptron on the test set
predicted_labels = test(learned_weights, X_test)

# Display the predicted labels
print("Predicted Labels:", predicted_labels)



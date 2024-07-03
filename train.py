import numpy as np
def load_data(filename):
    with open(filename, 'r') as file:
        num_samples = int(file.readline().strip())
        data = [[float(x) for x in line.split()] for line in file]
        columns = [f'feature_{i}' for i in range(len(data[0]) - 1)] + ['target']
        X = pd.DataFrame(data, columns=columns)

    return X

def initialize_weights(input_size):
    # Initialize weights with random values between -1 and 1
    return np.random.uniform(-1, 1, input_size + 1)

def threshold_function(x):
    # Binary threshold activation function
    return 1 if x > 0 else 0

def predict(weights, x):
    # Calculate the net input and apply the threshold function for prediction
    net_input = np.dot(weights[1:], x[1:]) + weights[0]
    return threshold_function(net_input)

def update_weights(weights, x, y, learning_rate):
    # Update weights based on the perceptron learning rule
    prediction = predict(weights, x)
    error = y - prediction
    weights[1:] += learning_rate * error * x[1:]
    weights[0] += learning_rate * error

def train_perceptron(X_train, y_train, learning_rate=0.01, max_epochs=50):
    # Initialize weights and get the number of examples and features
    num_examples, num_features = X_train.shape
    weights = initialize_weights(num_features)

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(num_examples):
            # Insert bias term and get input and target values
            x = np.insert(X_train.iloc[i].values, 0, 1)
            y = y_train.iloc[i]["label"]

            # Make a prediction, calculate error, and update weights
            prediction = predict(weights, x)
            error = abs(y - prediction)
            total_error += error
            update_weights(weights, x, y, learning_rate)

        # Check for convergence
        if total_error == 0:
            print(f"Converged in {epoch + 1} epochs.")
            break

    # Print a message if maximum epochs reached without convergence
    if total_error > 0:
        print("Maximum number of epochs reached. Model may not have converged.")

    return weights




# Load training data from file
file_path = 'train_data.txt'
train_data = load_data(file_path)

# Assuming the target variable is in a column named 'target' in your DataFrame
X_train = train_data.drop(columns=['label'])
y_train = train_data[['label']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the perceptron
learned_weights = train_perceptron(X_train, y_train)

# Display the learned weights
print("Learned Weights:", learned_weights)




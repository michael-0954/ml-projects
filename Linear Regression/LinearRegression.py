#a multiple linear regression model utilizing batch gradient descent, implementing automatic convergence checks and feature scaling, plotting the cost over time using matplotlib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.w = None
        self.b = None
        self.cost_history = []
    
    #__str__ method returns model parameters
    def __str__(self):
        if self.w is None or self.b is None:
            return "LinearRegression model is not fitted"
        return f"LinearRegression Model \nWeights: {self.w.round(5)} \nBias: {self.b.round(5)}"
    
    #cost function 
    def compute_cost(self, x, y):
        m = x.shape[0]
        cost = 0
        for i in range(m):
            f_wbi = np.dot(self.w, x[i]) + self.b
            cost += (f_wbi-y[i])**2
        return cost / (2*m)

    #computes the gradient for all parameters (partial derivative of cost function)
    def compute_gradient(self, x, y):
        m, n = x.shape
        gradient_w = np.zeros(n)
        gradient_b = 0
        for i in range(m):
            err = (np.dot(self.w, x[i]) + self.b) - y[i]
            for j in range(n):
                gradient_w[j] += err*x[i][j]
            gradient_b += err
        gradient_w = gradient_w / m
        gradient_b = gradient_b / m
        return gradient_w, gradient_b
    
    #feature scaling (z-score normalization)
    def scale(self, x):
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        x_scaled = (x - self.x_mean) / self.x_std
        return x_scaled
    
    #bulk of the model, computes w parameters and b using gradient descent and saves the cost over time
    def fit(self, x, y, lr=0.1, iters=10000, epsilon=0.001):
        x_scaled = self.scale(x)
        self.w = np.zeros(x.shape[1])
        self.b = 0
        self.cost_history = []
        for i in range(iters):
            grad_w, grad_b = self.compute_gradient(x_scaled, y)
            self.w = self.w - lr * grad_w
            self.b = self.b - lr * grad_b
            cost = self.compute_cost(x_scaled, y)
            self.cost_history.append(cost)
            if(i%(math.ceil(iters/10)) == 0):
                print(f"\nIteration {i} \nCost: {cost.round(5)}; Gradients: dj_dw={grad_w.round(5)}, dj_db={grad_b.round(5)}")
            if (np.abs(grad_w) <= epsilon).all() and math.fabs(grad_b) <= epsilon:
                print(f"Converged at iteration {i}\n")
                break

    #method to make a prediction given new data (any amount of rows of features as a np array)
    def predict(self, data):
        if self.w is None or self.b is None or self.x_mean is None or self.x_std is None:
            raise ValueError("Fit the model using fit() before making predictions")
        pred = np.zeros(data.shape[0])
        data_scaled = (data - self.x_mean) / self.x_std
        for i in range(data.shape[0]):
            pred_i= np.dot(self.w, data_scaled[i]) + self.b
            pred[i] = pred_i
        return pred

    #plot cost history (will plot two plots if over 1000 iterations for better visualization)
    def plot(self):
        if not self.cost_history:
            raise ValueError("Fit the model using fit() before plotting")
        if len(self.cost_history) >= 1000:
            fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
            ax1.plot(self.cost_history[:100])
            ax1.set_title("Cost vs. iteration(start)")
            ax1.set_xlabel("Iterations")
            ax1.set_ylabel("Cost")
            ax2.plot(1000 + np.arange(len(self.cost_history[1000:])), self.cost_history[1000:])
            ax2.set_title("Cost vs. iteration(end)")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Cost")
        else:
            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6,4))
            ax.plot(self.cost_history)
            ax.set_title("Cost vs. iteration")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Cost")
        plt.show()

#read data from csv using pandas and remove any null data  
def read_data(filepath, feature_cols, target_col):
    df = pd.read_csv(filepath)
    df = df.dropna(subset = feature_cols + [target_col])
    x = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    return x, y

#split the data into training and testing sets
def train_test_split(x, y, test_size=0.2):
    split_index = int(len(x) * (1 - test_size))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return x_train, x_test, y_train, y_test

#example usage of the LinearRegression class, reading data from a csv cleaned in clean_data.ipynb, training the model, making predictions, and plotting cost history
def main():
    x, y = read_data("Student_Performance_Cleaned.csv", ["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours"], "Performance Index")
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(f"{model}\n")
    predictions = model.predict(x_test)
    print(f"Predictions: {predictions.round(5)} \nActual: {y_test.round(5)}")
    model.plot()

if __name__ == "__main__":
    main()
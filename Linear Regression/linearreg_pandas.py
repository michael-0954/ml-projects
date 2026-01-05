#a simple linear regression model utilizing gradient descent and a pandas DataFrame, plotting the regression, cost over time, and parameters using matplotlib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#cost function 
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_xi = w * x[i] + b
        cost += (f_xi-y[i])**2
    return cost/(2*m)

#computes the gradient for both parameters (partial derivative of cost function)
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    gradient_w = 0
    gradient_b = 0
    for i in range(m):
        f_xi = w * x[i] + b
        gradient_w += (f_xi-y[i])*x[i]
        gradient_b += (f_xi-y[i])
    gradient_w = gradient_w/m
    gradient_b = gradient_b/m
    return gradient_w, gradient_b

#bulk of the model, computes w and b using gradient descent and saves the cost and parameters over time
def gradient_descent(x, y, w, b, lr, iters, cost_function, gradient_function):
    cost_history = []
    param_history = []
    for i in range(iters):
        grad_w, grad_b = gradient_function(x, y, w, b)
        w = w-lr*grad_w
        b = b-lr*grad_b
        cost = cost_function(x, y, w, b)
        cost_history.append(cost)
        param_history.append((w, b))
        if(i%(math.ceil(iters/10))==0):
            print(f"Iteration {i} \n Cost: {cost.round(5)}; Derivatives: dj_dw={grad_w.round(5)}, dj_db={grad_b.round(5)}; Parameters: w={w.round(5)}, b={b.round(5)}\n")
        if math.fabs(grad_w)<=lr and math.fabs(grad_b)<=lr:
            print(f"Converged at iteration {i}")
            break
    return w, b, cost_history, param_history

def main():
    #sample data that i cleaned in a separate notebook, could be played around with
    df = pd.read_csv("titanic_cleaned.csv")
    x_train = df["Age"].to_numpy()
    y_train = df["Fare"].to_numpy()

    #initial values, could be played around with 
    w_init = 0
    b_init = 0
    lr = 0.001 
    iters = 10000

    #run model and print results
    w_final, b_final, cost_hist, param_hist = gradient_descent(x_train, y_train, w_init, b_init, lr, iters, compute_cost, compute_gradient)
    print(f"Final Parameters: \n w={w_final.round(5)}, b={b_final.round(5)}")
    
    #4 plots showing linear regression, cost, and parameters 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, figsize=(12,4))

    #first plot: linear regression 
    ax1.scatter(x_train, y_train, c="red")
    ax1.set_title("Linear Regression ")
    ax1.set_ylabel('Y') 
    ax1.set_xlabel('X')
    x=np.arange(0, x_train.max()+math.ceil(x_train.mean()))
    y=w_final * x + b_final
    ax1.plot(x, y)
    
    #second plot: parameters w and b
    w, b = zip(*param_hist)
    ax2.scatter(b, w)
    ax2.set_title("W vs. B")
    ax2.set_ylabel('W') 
    ax2.set_xlabel('B')

    #third and fourth plots: cost over time
    ax3.plot(cost_hist[:100])
    ax4.plot(1000 + np.arange(len(cost_hist[1000:])), cost_hist[1000:])
    ax3.set_title("Cost vs. iteration(start)")  
    ax4.set_title("Cost vs. iteration (end)")
    ax3.set_ylabel('Cost')      
    ax4.set_ylabel('Cost') 
    ax3.set_xlabel('iteration step')    
    ax4.set_xlabel('iteration step')

    plt.show()

if __name__ == '__main__':
    main()



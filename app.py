from flask import Flask, render_template, request
import math
import sympy as sp
import numpy as np

# Function to read and process the mathematical function
def read_function(func_str: str):
    x = sp.symbols('x')
    
    # Parse the function string into a SymPy expression
    func = sp.sympify(func_str)
    
    # Compute the derivative of the function
    func_prime = sp.diff(func, x)
    
    # Convert the SymPy expressions into numpy-compatible lambda functions
    np_func = sp.lambdify(x, func, 'numpy')
    np_func_prime = sp.lambdify(x, func_prime, 'numpy')
    
    return np_func, np_func_prime

# Bisection Method
def bisection_method(f, a, b, max_iter=100, epsilon=1e-5):
    if f(a) * f(b) >= 0:
        return "The function must have opposite signs at the interval endpoints."

    iter_count = 0
    while (b - a) / 2 > epsilon and iter_count < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c  # Found exact root
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return (a + b) / 2  # Approximate root

# Fixed Point Iteration
def fixed_point_iteration(g, x_val, max_iter=100, epsilon=1e-5):
    iter_count = 0
    while iter_count < max_iter:
        x_new = g(x_val)
        if np.abs(x_new - x_val) < epsilon:
            return x_new
        x_val = x_new
        iter_count += 1
    return x_val  # Approximate root

# Newton's Method
def newtons_method(f, f_prime, x_val, max_iter=100, epsilon=1e-5):
    iter_count = 0
    while iter_count < max_iter:
        fx = f(x_val)
        fpx = f_prime(x_val)
        if fpx == 0:
            return "Derivative is zero, method cannot proceed."
        x_new = x_val - fx / fpx
        if np.abs(x_new - x_val) < epsilon:
            return x_new
        x_val = x_new
        iter_count += 1
    return x_val  # Approximate root

# Flask Application Setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def root_finding():
    if request.method == "POST":
        input1 = request.form.get("input1")  # Function input as string
        input2 = float(request.form.get("input2"))  # Float input for initial guess or interval
        input3 = float(request.form.get("input3"))  # Float input for second parameter
        radio_choice = request.form.get("radio")  # Method choice

        np_func, np_func_prime = read_function(input1)  # Convert function string to function and its derivative
        result = None
        
        if radio_choice == "0":  # Bisection Method
            a, b = input2, input3  # interval endpoints
            result = bisection_method(np_func, a, b)
        elif radio_choice == "1":  # Fixed Point Iteration
            g = np_func  # g(x) for fixed point iteration
            result = fixed_point_iteration(g, input2)
        elif radio_choice == "2":  # Newton's Method
            result = newtons_method(np_func, np_func_prime, input2)
        
        answered = True
        return render_template("index.html", answered=answered, result=result, radio_choice=radio_choice)
    
    return render_template("index.html", answered=False)

if __name__ == "__main__":
    app.run(debug=True)

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    f = (a*x**2)+(b*x)+c
    for _ in range(steps):
        grad = 2 * a * x + b
        x = x - lr * grad
    return x
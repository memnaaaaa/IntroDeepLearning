# Gradient descent on f(w) = (w - 3)^2

w = 0.0 # initial guess
eta = 0.1 # learning rate


# define the function f(w) = (w -3)^2
def f(w):
    return (w - 3) ** 2


# calculate the gradient
def grad_f(w):
    return 2 * (w - 3)


# Run gradient descent
for i in range (50):
    grad = grad_f(w)
    w = w - eta * grad
    if i % 10 == 0:
        print ( f" Step {i}: w = {w:.4f} , f(w) = {f(w):.4f}")


print (" Final w:", w )
import copy, math
import numpy as np


x_train = np.array([
    [2104,5,1,45],
    [1416,3,2,40],
    [852,2,1,35]])
y_train = np.array([460,232,178])

print(x_train)
print(x_train.shape)
print(type(x_train))

b_init = 0.0
w_init = np.zeros_like(x_train[0]) * 0.0
print(w_init)

def predict_single_loop(w,x,b):
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = w[i] * x[i] + b
        p = p+p_i

    p=p+b
    return p

def predict(x, w, b):
    p = np.dot(w,x)+b;

    return p

def compute_cost(X, y, w, b):
    m = X.shape[0]
    J = 0.0
    for i in range(m):
        f = predict(X[i], w, b)
        j_xi = (f - y[i])**2
        J=J+j_xi
    J=J/(2*m)
    return J

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    dw = np.zeros_like(w)
    db = 0.0
    for i in range(m):
        f = predict(X[i], w, b)
        dwi = (f - y[i])*X[i]
        dw = dw +dwi
        dbi = (f - y[i])
        db = db + dbi
    dw = dw / m
    db = db / m
    return dw,db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        j_i = cost_function(X, y, w, b)
        J_history.append(j_i)

        dw, db = gradient_function(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db

        '''
        {i:4d} 是格式化表达式，表示变量 i 将被格式化为一个整数（d 代表整数），并且占用至少4个字符的宽度。如果 i 的位数少于4位，前面会补空格以对齐。
        J_history[-1] 是从名为 J_history 的列表中获取最后一个元素。假设这是一个存储每次迭代成本值的历史记录列表。
        {:8.2f} 是另一个格式化表达式，表示对应的值将被格式化为一个浮点数，整体宽度至少为8个字符，小数点后保留2位。和整数格式化一样，不足部分会用空格填充。
        '''
        if i % math.ceil(num_iters/10)==0:
            print(f"Iter {i:4d} : cost {J_history[-1]:8.2f}")
    return w, b, J_history

alpha = 5.0e-7
iterations = 1000
w, b, J = gradient_descent(x_train,y_train,w_init,b_init,compute_cost,compute_gradient,alpha,iterations)


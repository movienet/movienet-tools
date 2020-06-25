
__all__ = ['acc_list']

def acc_list(x):
    y = [0]
    for x_ in x:
        y.append(y[-1] + x_)
    return y[1:]
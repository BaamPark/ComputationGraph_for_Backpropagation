import numpy as np

def term1_sin_square(x, w):
    return np.sin(x*w) ** 2

def term2_cos(x, w):
    return np.cos(x*w)

def term3(term1, term2):
    return 1 / (2 + term1 + term2)


def df_dx(x1, w1, x2, w2):
    # Derivative of the outer function: df/du = -1/u^2
    v = np.sin(x1 * w1) ** 2
    u = 2 + v + np.cos(x2 * w2)
    #f = 1/u

    df_du = -1 / u**2
    # Derivative of the inner function: dv/dx = 2*sin(x)*cos(x)
    du_dv = 1
    # Derivative of the middle function: du/dv = 1
    dv_dw1 = 2 * x1 * np.sin(x1 * w1) * np.cos(x1 * w1)
    df_dw1 = df_du * du_dv * dv_dw1

    du_dw2 = -1 * x2 * np.sin(x2 * w2)
    df_dw2 = df_du * du_dw2

    # Combine using the chain rule: df/dx = df/du * du/dv * dv/dx
    return df_dw1, df_dw2

print(df_dx(np.pi/4, 1, np.pi/4, 1))
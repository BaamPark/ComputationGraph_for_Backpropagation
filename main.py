import numpy as np
import random
class GradientGraph:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.w1 = random.random()
        self.w2 = random.random()
        self.grad1 = 0
        self.grad2 = 0

    def setweights(self, w1, w2):
        self.w1 = w1
        self.w2 = w2
    def forward(self):
        u = np.sin(self.x1 * self.w1) ** 2
        v = 2 + u + np.cos(self.x2 * self.w2)
        f = 1 / v
        return f

    def compute_gradw1(self):
        u = np.sin(self.x1 * self.w1) ** 2
        v = 2 + u + np.cos(self.x2 * self.w2)

        df_dv = -1 / v ** 2
        dv_du = 1
        dv_dw1 = 2 * self.x1 * np.sin(self.x1 * self.w1) * np.cos(self.x1 * self.w1)
        df_dw1 = df_dv * dv_du * dv_dw1
        self.grad1 = df_dw1
        return df_dw1

    def compute_gradw2(self):
        u = np.sin(self.x1 * self.w1) ** 2
        v = 2 + u + np.cos(self.x2 * self.w2)

        df_dv = -1 / v ** 2
        dv_dw2 = -1 * self.x2 * np.sin(self.x2 * self.w2)

        df_dw2 = df_dv * dv_dw2
        self.grad2 = df_dw2

a = GradientGraph(np.pi/4, np.pi/4)
a.setweights(1, 1)
print(a.w1)
print(a.w2)

a.compute_gradw1()
a.compute_gradw2()

print(a.grad1)
print(a.grad2)

# def df_dx(x1, w1, x2, w2):
#     v = np.sin(x1 * w1) ** 2
#     u = 2 + v + np.cos(x2 * w2)
#
#     df_du = -1 / u ** 2
#     du_dv = 1
#     dv_dw1 = 2 * x1 * np.sin(x1 * w1) * np.cos(x1 * w1)
#     df_dw1 = df_du * du_dv * dv_dw1
#
#     du_dw2 = -1 * x2 * np.sin(x2 * w2)
#     df_dw2 = df_du * du_dw2
#
#     return df_dw1, df_dw2

# print(df_dx(np.pi/4, 1, np.pi/4, 1))
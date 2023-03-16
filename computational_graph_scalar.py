import numpy as np
import random
class GradientGraph:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.w1 = random.random() #random weight
        self.w2 = random.random() #random weight
        self.grad1 = 0
        self.grad2 = 0

    def setweights(self, w1, w2): #use this method when you want to use specific weight
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
        du_dw1 = 2 * self.x1 * np.sin(self.x1 * self.w1) * np.cos(self.x1 * self.w1)
        df_dw1 = df_dv * dv_du * du_dw1 #Key line: apply chain rule to get gradient with respect to w1
        self.grad1 = df_dw1
        return df_dw1

    def compute_gradw2(self):
        u = np.sin(self.x1 * self.w1) ** 2
        v = 2 + u + np.cos(self.x2 * self.w2)

        df_dv = -1 / v ** 2
        dv_dw2 = -1 * self.x2 * np.sin(self.x2 * self.w2)

        df_dw2 = df_dv * dv_dw2 #Key line: apply chain rule to get gradient with respect to w2
        self.grad2 = df_dw2
        return df_dw2

a = GradientGraph(np.pi/4, np.pi/4)
a.setweights(1, 1)
a.compute_gradw1()
a.compute_gradw2()
#
print(a.grad1) #gradient of w1 #when x1 is pi/4 and w1 is 1, the gradient with respect to w1 is -0.07
print(a.grad2) #gradient of w2 #when x2 is pi/4 and w2 is 1, the gradient with respect to w2 is 0.05

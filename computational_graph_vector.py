import numpy as np
class GradientGraphVector:
    def __init__(self, x, w, index): #x = [[1, 2, 3]]
        self.x = x.T #numpy vector 3x1
        self.w = w #numpy matrix 3x3
        self.index = index #this index is used to specify the target weight to calculate the gradient

    def inner_product(self): #score function a = WX
        return np.dot(self.w, self.x)
    def derivative_inner_product(self): #derivative of score function
        return self.x[self.index[1]]
    def sigmoid(self):
        return 1/(1 + np.exp(-self.inner_product()))
    def derivative_sigmoid(self):
        return (self.sigmoid()[self.index[0]]) * (1 - (self.sigmoid()[self.index[0]]))
    def l2(self):
        return np.sum(np.square(self.sigmoid()))
    def derivative_l2(self):
        return 2*self.sigmoid()[self.index[0]]


X = np.array([[1, 1, 1]])
W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
index = [0, 0] #By specifying index, you can calculate the gradient with respect to W[0][1]
a = GradientGraphVector(X, W, index)
print(a.l2())
chain = a.derivative_inner_product() * a.derivative_sigmoid() * a.derivative_l2()

print(a.derivative_l2())
print(a.derivative_sigmoid())
print(a.derivative_inner_product())
print(chain)
# print(a.derivative_inner_product())
# print(a.derivative_sigmoid())
# print(a.derivative_l2())

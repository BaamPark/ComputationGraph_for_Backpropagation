# ComputationalGraph_for_Backpropagation

#Introduction
Computational graph for backpropagation is a tool used in deep learning to efficiently compute gradients 
during the training process. It represents the computations performed in a neural network as a graph, and 
enables efficient computation of gradients during the backward pass.
Two python files are designed to implement computational graph that is used to demonstrate backpropagation. 

## computational_graph_scalar.py
The implementated computal graph instance calculate gradients of the below function with respect *x* and *w*.
$$f(x,w)=\dfrac{1}{2+\sin^2(x_1w_1)+\cos(x_2w_2)}$$

## computational_graph_vector.py
The implemted computational graph instance compute gradients of the below function with respect *x* and *W*. The W is 3x3 matrix and x is 3x1 vector.
$$f(x,w)=||sigmoid(Wx)||^2$$

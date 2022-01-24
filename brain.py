import numpy as np

N_INPUT = 4
N_HIDDEN = 8
N_OUTPUT = 6

def main():
    W1 = np.random.uniform(low=-1, high=1, size=(N_HIDDEN, N_INPUT))
    W2 = np.random.uniform(low=-1, high=1, size=(N_OUTPUT, N_HIDDEN))
    brain = Brain(N_INPUT, N_HIDDEN, N_OUTPUT, W1, W2)
    brain.print_brain()
    print(brain.evaluate([0.1,-0.5,0.7,-0.2]))
 
def sigmoid(x):
    """
    x can be an array-like
    returns 2 * (1/(1-exp(x))) - 1,
    which is between -1 and 1
    """
    return 2./(1+np.exp(-x)) - 1.

class Brain:
    """
    Create single neural net with n1 input nodes, n2 hidden nodes, 
    and n3 output nodes

    Inputs
    n1, n2, n3 (ints): number of nodes in input layer, hidden layer,
        and output layer respectively
    W1 (2d array of floats): n2xn1 array of weights for n1 -> n2.
        These should be in the range [-1, 1].
    W2 (2d array of floats): n3xn2 array of weights for n2 -> n3
        These should be in the range [-1, 1].

    Methods
    evaluate(inputs)
    """
    def __init__(self, n1, n2, n3, W1, W2):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.W1 = W1 
        self.W2 = W2 
        assert W1.shape == (n2,n1), "W1 weight array must be n2xn1"
        assert W2.shape == (n3,n2), "W2 weight array must be n3xn2"
    
    def print_brain(self):
        print("n1: ", self.n1)
        print("n2: ", self.n2)
        print("n3: ", self.n3)
        print("W1: ", self.W1)
        print("W2: ", self.W2)

    def evaluate(self, inputs):
        """
        Inputs: input array, length n_input_nodes
        """
        assert len(inputs) == self.n1
        v2 = sigmoid(np.matmul(self.W1, inputs))
        v3 = sigmoid(np.matmul(self.W2, v2))
        return(v3)
        
main()

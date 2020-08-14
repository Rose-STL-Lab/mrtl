# Source: https://github.com/ktcarr/salinity-corn-yields/tree/master/mrtl

import numpy as np
import torch
import torch.nn.functional as F


def kr(matrices):
    """Khatri-Rao product of a list of matrices"""
    #n_col=3
    n_col = matrices[0].shape[1]
    for i, e in enumerate(matrices[1:]):
        if not i:
            res = matrices[0]

        a = res.reshape(-1, 1, n_col)
        b = e.reshape(1, -1, n_col)
        #print("a.shape")
        #print(a.shape)
        #print("b.shape")
        #print(b.shape)
        #print("(a*b).shape")
        #print((a*b).shape)

        res = (a * b).reshape(-1, n_col)

    return res


def kruskal_to_tensor(factors):
    """Turns the Khatri-product of matrices into a full tensor"""
    
    
    shape = [factor.shape[0] for factor in factors]

    #print(factors[2])
    

    #if len(factors[2].shape)!=2:

        #if factors[2].shape[2]!=1:
            #print("inside k")
            #shape=[shape[0],2,int(factors[2].shape[0])]
            #full_tensor = torch.mm(factors[0], kr(factors[1:]).T)
        #if factors[2].shape[0]==1:
            #shape=[shape[0],8,int(factors[2].shape[3])]
            #full_tensor = torch.mm(factors[0], kr(factors[1:]).T)

  
    full_tensor = torch.mm(factors[0], kr(factors[1:]).T)

  

    return full_tensor.reshape(*shape)


class my_regression(torch.nn.Module):
    def __init__(self, lead, input_shape, output_shape):
        super().__init__()
        self.lead=lead
        self.input_shape=input_shape
        self.output_shape=output_shape

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = np.prod([self.lead, 2, *input_shape])
        #This is the best way to initilize a neural network
        #n is the number of inputs
        k = 1. / (n**(1 / 2))
        #         k = 1./n**2
        #self.w is the weight tensor:it's the coefficient for every grid point at each time step. 
        #torch.FloatTensor() initializes a random tensor, and the arguments to that are the shape, the first dimension is lead, 
        #so out of the cube that's the tensor, one dimension is lead, which is 12 months
        #2 corresponds to two variables, salinity and temperature
        #the third dimension is the spatila dimension, so input_shhape is the lat and lon, but take the product, in order to flatten it
        #the shape of the input would be 12x2xlatxlon
        #if you do a feed forward neural network you would have some layers, then you initialize the network, and PyTorch initilizes
        #all of the weights automatically,but since it's defined manually, and we define a torch.nn.Paramter we have to initilize the weights ourself
        #here we are just initializing the weights with randomly selected numbers from a uniform distribution, that ranges from -k to k
        #This is the best way to initilize a neural network
        #.uniform fills the tensor with random numbers

        self.w = torch.nn.Parameter(torch.FloatTensor(
            self.lead, 2, np.prod(self.input_shape)).uniform_(-k, k).to(device),
                                    requires_grad=True)
        self.b = torch.nn.Parameter(torch.FloatTensor(self.output_shape).uniform_(
            0, 1).to(device),
                                    requires_grad=True)

    def forward(self, x):
        #print("x in forward")
        #print(x)
        #print("x.shape")
        #print(x.shape)
        #print("self.w.shape")
        #print(self.w.shape)
        #print("self.w")
        #print(self.w)
        out = torch.einsum('abcd,bcd->a', x, self.w)
        #print("out.shape")
        #print(out.shape)
        #print("out")
        #print(out)
        #print("out+b")
        #print(out+self.b)

        return out + self.b


class mini_net(torch.nn.Module):
    def __init__(self, lead, input_shape, output_shape, hidden_neurons=3):
        super().__init__()
        self.lead=lead


        n = np.prod([self.lead, *input_shape])
        k = 1. / np.sqrt(n)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(
            self.lead, 2, np.prod(input_shape), hidden_neurons).uniform_(-k, k),
                                     requires_grad=True)
        self.b1 = torch.nn.Parameter(
            torch.FloatTensor(hidden_neurons).uniform_(-k, k),
            requires_grad=True)
        #the weight tensor in each layer contains 
        self.w2 = torch.nn.Parameter(
            torch.FloatTensor(hidden_neurons).uniform_(-k, k),
            requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.FloatTensor(output_shape).uniform_(
            -k, k),
                                     requires_grad=True)

    def forward(self, x):
        out = F.relu(torch.einsum('abcd,bcde->ae', x, self.w1)) + self.b1
        out = torch.einsum('ae,e->a', out, self.w2) + self.b2
        return out


class my_regression_low(torch.nn.Module):
    #this class is for the low rank model
    def __init__(self, lead, input_shape, output_shape, K):
        # input_shape represents spatial dimensions (i.e. lat/lon)
        # lead represents number of months in advance used for prediction
        # K is rank of decomposition
        super().__init__()
        self.lead=lead
        self.input_shape=input_shape
        self.output_shape=output_shape
        #print("self.input_shape")

        #print(self.input_shape)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = np.prod([self.lead, *input_shape])
        k_A = 1. / np.sqrt((self.lead))
        k_B = 1. / np.sqrt(2)
        k_C = 1. / np.sqrt(np.prod(self.input_shape))

        n = np.prod([self.lead, 2, *input_shape])
        k_b = 1. / (n**(1 / 2))

        # Different initialization
        # Latent factors
        #the size of self.A is the lead
        self.A = torch.nn.Parameter(torch.FloatTensor(self.lead, K).normal_(
            0, k_A).to(device),
                                    requires_grad=True)
        #the size of B is 2
        self.B = torch.nn.Parameter(torch.FloatTensor(2, K).normal_(
            0, k_B).to(device),
                                    requires_grad=True)
        #the size of C is latxlon
        #self.C = torch.nn.Parameter(torch.FloatTensor(np.prod(self.input_shape+[lead]), K).normal_(0, k_C).to(device),requires_grad=True)
        self.C = torch.nn.Parameter(torch.FloatTensor(
            np.prod(self.input_shape), K).normal_(0, k_C).to(device),
                                    requires_grad=True)
   

        # Bias
        self.b = torch.nn.Parameter(torch.FloatTensor(self.output_shape).normal_(
            0, k_b).to(device),
                                    requires_grad=True)
        

    def forward(self, X):
        """
        maps an input tensor to an output tensor
        all Pytorch models have a forward method
        """

        #contains thhe different operations/layers
        #used for low rank model
        #print("self.A")
        #print(self.A.shape)
        #print("self.B")
        #print(self.B.shape)
        #print("self.C")
        #print(self.C.shape)
        #if X.shape[1]==4:
           
            #X=(X.reshape(X.shape[0],1,8,324))
            
        
        
        
        #print("((kruskal_to_tensor([self.A, self.B, self.C])).shape)")
        #print((kruskal_to_tensor([self.A, self.B, self.C])).shape)
        #print("(X * kruskal_to_tensor([self.A, self.B, self.C]).shape)")
        #print((X*(kruskal_to_tensor([self.A, self.B, self.C]))).shape)
        #b=self.B.reshape(*8)
     

        
        core = (X * kruskal_to_tensor([self.A, self.B, self.C])).reshape(X.shape[0], -1).sum(1)

        #print("core.shape")
        #print(core.shape)

        out = core.add_(self.b)
        #print("self.b in forward")
        #print(self.b.shape)

        #print("out.shape")
        #print(out.shape)

        return out
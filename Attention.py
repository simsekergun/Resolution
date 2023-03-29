from tqdm.autonotebook import tqdm
import os,sys,humanize,psutil,GPUtil
import time
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

########################################################################
# Define mem_report function
def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
########################################################################

def visualize2DSoftmax(X, y, model):
    """Function to visualize the classification boundary of a learned model on a 2-D dataset
    Arguments:
    X -- a numpy array of shape (2, N), where N is the number of data points. 
    y -- a numpy array of shape (N,), which contains values of either "0" or "1" for two different classes
    model -- a PyTorch Module object that represents a classifer to visualize. s
    """
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20), np.linspace(y_min, y_max, num=20), indexing='ij')
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    with torch.no_grad():
        preds = model(torch.tensor(xy_v, dtype=torch.float32))
        preds = F.softmax(preds, dim=1).numpy()

    cs = plt.contourf(xv, yv, preds[:,0].reshape(20,20), levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu)
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=cs.ax)
########################################################################
def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    """
    model -- the PyTorch model / "Module" to run for one epoch
    optimizer -- the object that will update the weights of the network
    data_loader -- DataLoader object that returns tuples of (input, label) pairs. 
    loss_func -- the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    device -- the compute lodation to perform training
    score_funcs -- a dictionary of scoring functions to use to evalue the performance of the model
    prefix -- a string to pre-fix to any scores placed into the _results_ dictionary. 
    desc -- a description to use for the progress bar.     
    """
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        #Move the batch to the device we are using. 
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs) #this just computed f_Θ(x(i))
        # Compute loss.
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            #moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            #add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    #end training epoch
    end = time.time()
    
    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1: #We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    #Else, we assume we are working on a regression problem
    
    results[prefix + " loss"].append( np.mean(running_loss) )
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append( score_func(y_true, y_pred) )
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end-start #time spent on epoch
########################################################################
def run_epoch_reg2(model, optimizer, data_loader, loss_func, device, results, score_funcs,y_true,y_pred, prefix="", desc=None):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        #Move the batch to the device we are using. 

        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)
        
        y_hat = model(inputs) #this just computed f_Θ(x(i))

        # Compute loss.
        loss = loss_func(y_hat, labels)
         
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        labels = labels.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()        
        y_true.extend(labels.tolist())
        y_pred.extend(y_hat.tolist())

    #end training epoch
    end = time.time()
    plt.figure(1)
    plt.scatter(np.asarray(y_true),np.asarray(y_pred))
    
    y_pred = np.asarray(y_pred)
    return y_pred, y_true
########################################################################
def train_simple_network(model, loss_func, train_loader, test_loader=None, score_funcs=None, 
                         epochs=50, device="cpu", checkpoint_file=None, lr=0.001):
    """Train simple neural network
    
    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs. 
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training
    
    """
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if test_loader is not None:
            to_track.append("test " + eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []
        
    #SGD is Stochastic Gradient Decent.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train()#Put our model in training mode
        
        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")

        results["total time"].append( total_train_time )
        results["epoch"].append( epoch )
        
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
                    
    if checkpoint_file is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results' : results
            }, checkpoint_file)

    return pd.DataFrame.from_dict(results)
########################################################################
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape) 
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)
    
class DebugShape(nn.Module):
    """
    Module that is useful to help debug your neural network architecture. 
    Insert this module between layers and it will print out the shape of 
    that layer. 
    """
    def forward(self, input):
        print(input.shape)
        return input
########################################################################    
def weight_reset(m):
    """
    Go through a PyTorch module m and reset all the weights to an initial random state
    """
    if "reset_parameters" in dir(m):
        m.reset_parameters()
    return
########################################################################
def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj
########################################################################        
def train_network(model, loss_func, train_loader, val_loader=None, test_loader=None,score_funcs=None, 
                         epochs=50, device="cpu", checkpoint_file=None, 
                         lr_schedule=None, optimizer=None, disable_tqdm=False
                        ):
    """Train simple neural networks
    
    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs. 
    val_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training
    lr_schedule -- the learning rate schedule used to alter \eta as the model trains. If this is not None than the user must also provide the optimizer to use. 
    optimizer -- the method used to alter the gradients for learning. 
    
    """
    if score_funcs == None:
        score_funcs = {}#Empty set 
    
    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if val_loader is not None:
            to_track.append("val " + eval_score )
        if test_loader is not None:
            to_track.append("test "+ eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []

        
    if optimizer == None:
        #The AdamW optimizer is a good default optimizer
        optimizer = torch.optim.AdamW(model.parameters())
        del_opt = True
    else:
        del_opt = False

    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch", disable=disable_tqdm):
        model = model.train()#Put our model in training mode

        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")
        
        results["epoch"].append( epoch )
        results["total time"].append( total_train_time )
        
      
        if val_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, val_loader, loss_func, device, results, score_funcs, prefix="val", desc="Validating")
                
        #In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()
                
        if test_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
        
        
        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results' : results
                }, checkpoint_file)
    if del_opt:
        del optimizer

    return pd.DataFrame.from_dict(results)
########################################################################
### RNN utility Classes 
class LastTimeStep(nn.Module):
    """
    A class for extracting the hidden activations of the last time step following 
    the output of a PyTorch RNN module. 
    """
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1    
    
    def forward(self, input):
        #Result is either a tupe (out, h_t)
        #or a tuple (out, (h_t, c_t))
        rnn_output = input[0]

        last_step = input[1]
        if(type(last_step) == tuple):
            last_step = last_step[0]
        batch_size = last_step.shape[1] #per docs, shape is: '(num_layers * num_directions, batch, hidden_size)'
        
        last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
        #We want the last layer's results
        last_step = last_step[self.rnn_layers-1] 
        #Re order so batch comes first
        last_step = last_step.permute(1, 0, 2)
        #Finally, flatten the last two dimensions into one
        return last_step.reshape(batch_size, -1)
########################################################################    
class EmbeddingPackable(nn.Module):
    """
    The embedding layer in PyTorch does not support Packed Sequence objects. 
    This wrapper class will fix that. If a normal input comes in, it will 
    use the regular Embedding layer. Otherwise, it will work on the packed 
    sequence to return a new Packed sequence of the appropriate result. 
    """
    def __init__(self, embd_layer):
        super(EmbeddingPackable, self).__init__()
        self.embd_layer = embd_layer 
    
    def forward(self, input):
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            # We need to unpack the input, 
            sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(input.cpu(), batch_first=True)
            #Embed it
            sequences = self.embd_layer(sequences.to(input.data.device))
            #And pack it into a new sequence
            return torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.cpu(), 
                                                           batch_first=True, enforce_sorted=False)
        else:#apply to normal data
            return self.embd_layer(input)


########################################################################
### Attention Mechanism Layers
class ApplyAttention(nn.Module):
    """
    This helper module is used to apply the results of an attention mechanism toa set of inputs. 
    """

    def __init__(self):
        super(ApplyAttention, self).__init__()
        
    def forward(self, states, attention_scores, mask=None):
        """
        states: (B, T, H) shape giving the T different possible inputs
        attention_scores: (B, T, 1) score for each item at each context
        mask: None if all items are present. Else a boolean tensor of shape 
            (B, T), with `True` indicating which items are present / valid. 
            
        returns: a tuple with two tensors. The first tensor is the final context
        from applying the attention to the states (B, H) shape. The second tensor
        is the weights for each state with shape (B, T, 1). 
        """
        
        if mask is not None:
            #set everything not present to a large negative value that will cause vanishing gradients 
            attention_scores[~mask] = -1000.0
        #compute the weight for each score
        weights = F.softmax(attention_scores, dim=1) #(B, T, 1) still, but sum(T) = 1
    
        final_context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D)
        return final_context, weights
########################################################################
class AttentionAvg(nn.Module):

    def __init__(self, attnScore):
        super(AttentionAvg, self).__init__()
        self.score = attnScore
    
    def forward(self, states, context, mask=None):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, D), a weighted av
        
        """
        
        B = states.size(0)
        T = states.size(1)
        D = states.size(2)
        
        scores = self.score(states, context) #(B, T, 1)
        
        if mask is not None:
            scores[~mask] = float(-10000)
        weights = F.softmax(scores, dim=1) #(B, T, 1) still, but sum(T) = 1
        
        context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D, 1)
        
        
        return context.view(B, D) #Flatten this out to (B, D)

########################################################################
class AdditiveAttentionScore(nn.Module):
    def __init__(self, D):
        super(AdditiveAttentionScore, self).__init__()
        self.v = nn.Linear(D, 1)
        self.w = nn.Linear(2*D, D)
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        #Repeating the values T times 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, D) -> (B, T, D)
        state_context_combined = torch.cat((states, context), dim=2) #(B, T, D) + (B, T, D)  -> (B, T, 2*D)
        scores = self.v(torch.tanh(self.w(state_context_combined)))
        return scores
########################################################################
class GeneralScore(nn.Module):
    def __init__(self, D):
        super(GeneralScore, self).__init__()
        self.w = nn.Bilinear(D, D, 1)
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        D = states.size(2)
        #Repeating the values T times 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, D) -> (B, T, D)
        scores = self.w(states, context) #(B, T, D) -> (B, T, 1)
        return scores
########################################################################
class DotScore(nn.Module):

    def __init__(self, D):
        super(DotScore, self).__init__()
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        D = states.size(2)
        
        scores = torch.bmm(states,context.unsqueeze(2)) / np.sqrt(D) #(B, T, D) -> (B, T, 1)
        return scores
########################################################################    
def getMaskByFill(x, time_dimension=1, fill=0):
    """
    x: the original input with three or more dimensions, (B, ..., T, ...)
        which may have unsued items in the tensor. B is the batch size, 
        and T is the time dimension. 
    time_dimension: the axis in the tensor `x` that denotes the time dimension
    fill: the constant used to denote that an item in the tensor is not in use,
        and should be masked out (`False` in the mask). 
    
    return: A boolean tensor of shape (B, T), where `True` indicates the value
        at that time is good to use, and `False` that it is not. 
    """
    to_sum_over = list(range(1,len(x.shape))) #skip the first dimension 0 because that is the batch dimension
    
    if time_dimension in to_sum_over:
        to_sum_over.remove(time_dimension)
       
    with torch.no_grad():
        #Special case is when shape is (B, D), then it is an embedding layer. We just return the values that are good
        if len(to_sum_over) == 0:
            return (x != fill)
        #(x!=fill) determines locations that might be unused, beause they are 
        #missing the fill value we are looking for to indicate lack of use. 
        #We then count the number of non-fill values over everything in that
        #time slot (reducing changes the shape to (B, T)). If any one entry 
        #is non equal to this value, the item represent must be in use - 
        #so return a value of true. 
        mask = torch.sum((x != fill), dim=to_sum_over) > 0
    return mask
########################################################################
class LanguageNameDataset(Dataset):    
    def __init__(self, lang_name_dict, vocabulary):
        self.label_names = [x for x in lang_name_dict.keys()]
        self.data = []
        self.labels = []
        self.vocabulary = vocabulary
        for y, language in enumerate(self.label_names):
            for sample in lang_name_dict[language]:
                self.data.append(sample)
                self.labels.append(y)
        
    def __len__(self):
        return len(self.data)
    
    def string2InputVec(self, input_string):
        """
        This method will convert any input string into a vector of long values, according to the vocabulary used by this object. 
        input_string: the string to convert to a tensor
        """
        T = len(input_string) #How many characters long is the string?
        
        #Create a new tensor to store the result in
        name_vec = torch.zeros((T), dtype=torch.long)
        #iterate through the string and place the appropriate values into the tensor
        for pos, character in enumerate(input_string):
            name_vec[pos] = self.vocabulary[character]
            
        return name_vec
    
    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]
        
        #Conver the correct class label into a tensor for PyTorch
        label_vec = torch.tensor([label], dtype=torch.long)
        
        return self.string2InputVec(name), label
########################################################################    
def pad_and_pack(batch):
    #1, 2, & 3: organize the batch input lengths, inputs, and outputs as seperate lists
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:
        input_tensors.append(x)
        labels.append(y)
        lengths.append(x.shape[0]) #Assume shape is (T, *)
    #4: create the padded version of the input
    x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    #5: create the packed version from the padded & lengths
    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)
    #Convert the lengths into a tensor
    y_batched = torch.as_tensor(labels, dtype=torch.long)
    #6: return a tuple of the packed inputs and their labels
    return x_packed, y_batched

# The following function creates the indicies of all independent monomials of 'n_vars' number of variables in degree 'degree'. 
def create_indices(n_vars, degree):
  idx = [[j] for j in range(n_vars)]
  s = idx.copy()
  for d in range(2, degree+1):
    new_s = [i + [j] for i in s for j in range(min(i)+1)]
    s = new_s
    for k in range(len(new_s)):
      idx.append(new_s[k])
  return idx

########################################################################
# The following function create all independent monomials of 'n_vars' number of variables in degree 'degree'.
def monomials(x, degree):
  x_monomials = []
  for idx in create_indices(len(x), degree):
    x_prod = 1
    for i in idx:
      x_prod *= x[i]
    x_monomials.append(x_prod)
  return x_monomials

# The following function create all independent monomials of 'n_vars' number of variables in degree 'degree'.

def monomials_alt(x, degree):
  x_tensor = torch.zeros(x.shape[0], len(create_indices(x.shape[1], degree)))
  for b in range(x.shape[0]):
    x_monomials = []
    for idx in create_indices(x.shape[1], degree):
      x_prod = 1
      for i in idx:
        x_prod *= x[b, i]
      x_monomials.append(x_prod)
    x_tensor[b, :] = torch.tensor(np.array(x_monomials)) 
  return x_tensor
########################################################################
def run_epoch_reg(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        #Move the batch to the device we are using. 

        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)
        
        y_hat = model(inputs) #this just computed f_Θ(x(i))

        # Compute loss.
        loss = loss_func(y_hat, labels)
         
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            #moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            #add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    #end training epoch
    end = time.time()
    
    y_pred = np.asarray(y_pred)
    
    results[prefix + " loss"].append( np.mean(running_loss) )
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append( score_func(y_true, y_pred) )
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end-start #time spent on epoch

# We modified the train_network function of the textbook by replacing the run_epoch function by run_epoch_reg defined above.
########################################################################
def train_network_reg(model, loss_func, train_loader, val_loader=None, test_loader=None,score_funcs=None, 
                         epochs=50, device="cpu", checkpoint_file=None, 
                         lr_schedule=None, optimizer=None, disable_tqdm=False
                        ):

    if score_funcs == None:
        score_funcs = {}#Empty set 
    
    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if val_loader is not None:
            to_track.append("val " + eval_score )
        if test_loader is not None:
            to_track.append("test "+ eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    if optimizer == None:
        #The AdamW optimizer is a good default optimizer
        optimizer = torch.optim.AdamW(model.parameters())

    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch", disable=disable_tqdm):
        model = model.train()#Put our model in training mode

        total_train_time += run_epoch_reg(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")
        
        results["epoch"].append( epoch )
        results["total time"].append( total_train_time )
        
      
        if val_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch_reg(model, optimizer, val_loader, loss_func, device, results, score_funcs, prefix="val", desc="Validating")
                
        #In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()
                
        if test_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch_reg(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
        
        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results' : results
                }, checkpoint_file)

    return pd.DataFrame.from_dict(results)
########################################################################
# Defining arctan function as an activation function in a network:
class atan(nn.Module):
  def __init__(self):
    super().__init__() 

  def forward(self, input):
    return torch.atan(input)

# n choose r function:

def nCr(n, r):
  f = math.factorial
  return f(n)/(f(r)*f(n-r))


class FC_Dataset(Dataset):
    def __init__(self, dataset, n=379):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        x = self.dataset[:, :self.n]
        y = self.dataset[:, self.n:]
        x = x.__getitem__(idx)
        y = y.__getitem__(idx)
        return x, y

# Defining a function for creating a fully connected layer accompanied by BatchNorm and the activation function:

def getLayer(in_size, out_size, activation='Sigmoid'):
    activation_dict = {'Sigmoid':nn.Sigmoid(), 'Tanh':nn.Tanh(), 'atan':atan(), 'ReLU':nn.ReLU(), 'LReLU':nn.LeakyReLU(0.1)}   
    return nn.Sequential( 
        nn.BatchNorm1d(in_size),
        nn.Linear(in_size,  out_size),
        activation_dict[activation]
    )


# Defining a constructor for building recurrent models. We combine 3 recurrent model types RNN, GRU, and LSTM in this constructor.
class recurrent_model(nn.Module):
  def __init__(self, T_length, input_dim, num_layers, num_neurons, output_dim, model_type, bidirectional=False):
    super().__init__()

    self.T_length = T_length  
    self.input_dim = input_dim
    self.num_layers = num_layers
    self.num_neurons = num_neurons
    self.output_dim = output_dim
    self.bidirectional = bidirectional
    self.model_type = model_type

    if self.bidirectional==False:
      self.last_num_neurons = self.num_neurons
    else:
      self.last_num_neurons = 2*self.num_neurons

    self.model_dict = {'RNN': nn.RNN(self.input_dim, self.num_neurons, self.num_layers, nonlinearity='relu', batch_first=True,   
                                     bidirectional=self.bidirectional),
                       'GRU': nn.GRU(self.input_dim, self.num_neurons, self.num_layers, batch_first=True, bidirectional=self.bidirectional),
                       'LSTM': nn.LSTM(self.input_dim, self.num_neurons, self.num_layers, batch_first=True, bidirectional=self.bidirectional)}

    self.layers = nn.ModuleList([nn.BatchNorm1d(self.T_length)] + [self.model_dict[self.model_type]] + 
                                [LastTimeStep(rnn_layers=self.num_layers)] +  
                                [nn.Linear(self.last_num_neurons, self.output_dim)])
    
  def forward(self, x):
    
    x = torch.transpose(x, 1, 2)
    for layer in self.layers:
      x = layer(x)
    y = x
    return y

########################################################################
########################################################################
# Constructing the combiner component of the architecture 
class Combiner(nn.Module):
    
    def __init__(self, featureExtraction, weightSelection):
        super(Combiner, self).__init__()
        self.featureExtraction = featureExtraction
        self.weightSelection = weightSelection
    
    def forward(self, input):
        features = self.featureExtraction(input) #(B, T, D) $\boldsymbol{h}_i = F(\boldsymbol{x}_i)$
        weights = self.weightSelection(features) #(B, T) or (B, T, 1) for $\boldsymbol{\alpha}$
        if len(weights.shape) == 2: #(B, T) shape
            weights.unsqueese(2) #now (B, T, 1) shape
        
        r = features*weights #(B, T, D), computes $\alpha_i \cdot \boldsymbol{h}_i$
        
        return torch.sum(r, dim=1) #sum over the T dimension, giving (B, D) final shape $\bar{\boldsymbol{x}}$

########################################################################
# Constructing the backbone component of the network
def backboneNetwork(T, D, neurons, activation):
  activation_dict = {'Sigmoid':nn.Sigmoid(), 'Tanh':nn.Tanh(), 'ReLU':nn.ReLU(), 'atan':atan(), 'LeakyReLU':nn.LeakyReLU(0.1)}
  return nn.Sequential(
    nn.BatchNorm1d(T),  
    nn.Linear(D, neurons), 
    activation_dict[activation],
    nn.BatchNorm1d(T),
    nn.Linear(neurons, neurons), 
    activation_dict[activation],
    nn.BatchNorm1d(T),
    nn.Linear(neurons, 2*neurons), 
    activation_dict[activation], 
    nn.Linear(2*neurons, 2*neurons), 
    activation_dict[activation], 
)
########################################################################
# Constructing the score component of the network
def attention_network(neurons, activation):
  activation_dict = {'Sigmoid':nn.Sigmoid(), 'Tanh':nn.Tanh(), 'ReLU':nn.ReLU(), 'atan':atan(), 'LeakyReLU':nn.LeakyReLU()}
  return nn.Sequential(
    nn.Linear(2*neurons, 2*neurons),
    activation_dict[activation],
    nn.Linear(2*neurons, 1), # (B, T, 1)
    nn.Softmax(dim=1),
)

# Creating a constructor for the attention model. Note one has the option to add extra layers on top of the attention network (desired number of 
# optional layers has to be specified through n_layers)

class attention_model(nn.Module):

  def __init__(self, T_length, input_dim, n_layers, num_neurons, output_dim, activation='LeakyReLU'):
    super().__init__()
    
    self.T_length = T_length
    self.input_dim = input_dim
    self.num_neurons = num_neurons
    self.n_layers = n_layers
    self.output_dim = output_dim
    self.activation = activation

    activation_dict = {'Sigmoid':nn.Sigmoid(), 'Tanh':nn.Tanh(), 'ReLU':nn.ReLU(), 'atan':atan(), 'LeakyReLU':nn.LeakyReLU()}

    self.layers = nn.ModuleList([Combiner(backboneNetwork(self.T_length, self.input_dim, self.num_neurons, self.activation), 
                                          attention_network(self.num_neurons, self.activation))] + 
                                [nn.Linear(2*self.num_neurons, 2*self.num_neurons) for _ in range(self.n_layers)] +
                                [nn.Linear(2*self.num_neurons, self.output_dim)])
    
    self.bns = nn.ModuleList([nn.BatchNorm1d(2*self.num_neurons) for _ in range(self.n_layers+1)])

    self.activ_layers = nn.ModuleList([activation_dict[self.activation] for _ in range(self.n_layers)])
    
  def forward(self, x):
    
    x = torch.transpose(x, 1, 2)
    x = self.layers[0](x)

    for bn, layer, active in zip(self.bns[:-1], self.layers[1:][:-1], self.activ_layers):
      x = active(layer(bn(x)))
    y = self.layers[-1](self.bns[-1](x))  
    return y

########################################################################
# Dot product score
class DotScore(nn.Module):
    def __init__(self, H):
        super(DotScore, self).__init__()
        self.H = H
    
    def forward(self, states, context):
        T = states.size(1)
        scores = torch.bmm(states, context.unsqueeze(2)) / np.sqrt(self.H) #(B, T, H) -> (B, T, 1)
        return scores
########################################################################
# General score 
class GeneralScore(nn.Module):

    def __init__(self, H):
        super(GeneralScore, self).__init__()
        self.w = nn.Bilinear(H, H, 1, bias=True) #stores $W$
    
    def forward(self, states, context):
        T = states.size(1) 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, H) -> (B, T, H)
        scores = self.w(states, context) #(B, T, H) -> (B, T, 1)
        return scores
########################################################################
# Additive attention score
class AdditiveAttentionScore(nn.Module):

    def __init__(self, H, activation='Tanh'):
        super(AdditiveAttentionScore, self).__init__()
        self.v = nn.Linear(H, 1) 
        self.w = nn.Linear(2*H, H)
        self.activation = activation
        self.activation_dict = {'Sigmoid':nn.Sigmoid(), 'Tanh':nn.Tanh(), 'ReLU':nn.ReLU(), 'atan':atan(), 'LeakyReLU':nn.LeakyReLU()}
    
    def forward(self, states, context):
        T = states.size(1) 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, H) -> (B, T, H)
        state_context_combined = torch.cat((states, context), dim=2) #(B, T, H) + (B, T, H)  -> (B, T, 2*H)
        scores = self.v(self.activation_dict[self.activation](self.w(state_context_combined))) # (B, T, 2*H) -> (B, T, 1)
        return scores

########################################################################    
# A constructor to apply attention score in a simpler way
class ApplyAttention(nn.Module):

    def __init__(self):
        super(ApplyAttention, self).__init__()
        
    def forward(self, states, attention_scores, mask=None):       
        if mask is not None:
            attention_scores[~mask] = -1000.0
        weights = F.softmax(attention_scores, dim=1) #(B, T, 1) still, but sum(T) = 1
        final_context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D)
        return final_context, weights

# A mask function to treat missing data in the T-sequence if any. For our dataset, there is no such missing value. Nonetheless, its inclusion 
# is harmless.

def getMaskByFill(x, time_dimension=1, fill=0):
    to_sum_over = list(range(1,len(x.shape))) #skip the first dimension 0 because that is the batch dimension
    
    if time_dimension in to_sum_over:
        to_sum_over.remove(time_dimension)
        
    with torch.no_grad():
        mask = torch.sum((x != fill), dim=to_sum_over) > 0
    return mask

########################################################################
# A constructor for the attention-based model with context vector
class SmarterAttentionNet(nn.Module):

    def __init__(self, T_length, input_dim, num_neurons, output_dim, activation='LeakyReLU', att_active='Tanh', score_net=None):
      
        super(SmarterAttentionNet, self).__init__()
        self.T = T_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.activation = activation
        self.att_active = att_active

        activation_dict = {'Sigmoid':nn.Sigmoid(), 'Tanh':nn.Tanh(), 'ReLU':nn.ReLU(), 'atan':atan(), 'LeakyReLU':nn.LeakyReLU()}
        
        self.backbone = backboneNetwork(self.T, self.input_dim, self.num_neurons, self.activation) #returns (B, T, neurons)       
        self.score_net = AdditiveAttentionScore(2*self.num_neurons, self.att_active) if (score_net is None) else score_net
        self.apply_attn = ApplyAttention()
        
        self.prediction_net = nn.Sequential(          #(B, H), 
            nn.BatchNorm1d(2*self.num_neurons),
            nn.Linear(2*self.num_neurons, 2*self.num_neurons),
            activation_dict[self.activation],
            nn.BatchNorm1d(2*self.num_neurons),
            nn.Linear(2*self.num_neurons, output_dim),  #(B, H)
            )        
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        mask = getMaskByFill(x)

        h = self.backbone(x) #(B, T, D) -> (B, T, H)
        h_context = (mask.unsqueeze(-1)*h).sum(dim=1)               #(B, T, H) -> (B, H)
        h_context = h_context/(mask.sum(dim=1).unsqueeze(-1)+1e-10)
        scores = self.score_net(h, h_context)                       # (B, T, H) , (B, H) -> (B, T, 1)
        final_context, _ = self.apply_attn(h, scores, mask=mask)
        y = self.prediction_net(final_context)
        return y
########################################################################
# Defining a constructor class which creates the correct tensor for the RNN type models. One can adjust (by choosing n) how to break the 
# data into feature and target subsets.
class RNN_Dataset(Dataset):
    # ngap determines the gap between training and test
    # datasets in terms for frequency (e.g f_gap = ngap*0.5 THz)
    def __init__(self, dataset, n=375, input_dim = 25, ngap=40):
        self.dataset = dataset
        self.n = n
        self.input_dim = input_dim        
        self.ngap = ngap        

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        x = self.dataset[:, :4]
        x = torch.stack([x for _ in range(self.input_dim)], dim=1)
        qc = self.dataset[:, 4:(self.n+4)].reshape(-1, self.input_dim, self.n//self.input_dim)
        x = torch.cat((x, qc), dim=2)
        y = self.dataset[:, 4+self.n+self.ngap:]       # Creating a frequency gap between feature and target
        x = x[idx, :, :]
        y = y[idx]
        return x, y

########################################################################
class RegressionDataset(Dataset):
    def __init__(self, dataset, reg_degree=1):
        self.dataset = dataset
        self.reg_degree = reg_degree

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)
        x = torch.tensor(monomials(x, self.reg_degree))
        return x, y
########################################################################    
def graph_results(model, test_data,device_num,freqs):

  sample_test_data = torch.tensor(np.array([np.array(test_data[device_num][0])]), dtype=torch.float32, device=device) 
  sample_test_dataT = torch.transpose(sample_test_data, 1, 2).to(device)
  y_true = np.array(test_data[device_num][1])

  with torch.no_grad():
    y_pred = model(sample_test_data).cpu().numpy()

  df = pd.DataFrame()
  df['Fr'] = pd.Series(freqs)
  df['y_true'] = pd.Series(y_true)
  df['y_pred'] = pd.Series(y_pred[0])
  df.plot(kind='line', x='Fr', y=['y_true', 'y_pred'], figsize=(8,4), title='Prediction vs Ground Truth for Device %d' %(device_num))
  plt.show()
  return
########################################################################
def return_true_pred(model, test_data,n_test,freqs):
    df_true = pd.DataFrame()
    df_true['Fr'] = pd.Series(freqs)
    df_pred = pd.DataFrame()
    df_pred['Fr'] = pd.Series(freqs)
    for device_num in range(n_test):
        sample_test_data = torch.tensor(np.array([np.array(test_data[device_num][0])]), dtype=torch.float32, device=device) 
        sample_test_dataT = torch.transpose(sample_test_data, 1, 2).to(device)
        y_true = np.array(test_data[device_num][1])
        with torch.no_grad():
            y_pred = model(sample_test_data).cpu().numpy()
        df_true[device_num] = pd.Series(y_true)
        df_pred[device_num] = pd.Series(y_pred[0])
    return df_true, df_pred
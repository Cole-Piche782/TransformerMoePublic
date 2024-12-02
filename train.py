"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import random

import testConfigs

import copy



# Relevant only when running the code on the OpenWebText dataset.
# If True, the program downloads the pre-trained embedding file and skips training the model.
shouldJustSaveGPTFile = False


# If True, the system selects cluster centroids only and does nothing else.
# Useful for debugging the cluster centroid selection function.
# Should not be set to True unless that's the sole intended action.
shouldJustHelpPickClusterCenters = False

# A variable that tells the system not to read the shift coefficients because they have already been read
# Never set the below to true in this script, it should only be set to true in the gpt config scripts
haveReadShiftCoefficients = False

# The below two build switches should only be set to true in the configuration files. 
# Never here. They help the system determine which embedding space to use etc.
shouldUseOpenWebText = False
shouldUseWordShakeSpeare = False

# If set to false, enables some extra logging of loss and percent accuracy
shouldOnlyShowEvalOut = False

# The below two build switches disable training and only perform evaluation if set to true.
shouldJustEstimateLoss = False
shouldEstimateMoeLoss = False

# Tells us whether to start from scratch, or from a checkpoint
shouldReloadRouter = False

# A variable to perform some spcial custom training I tried. It failed and so should be
# Set to false
shouldIgnoreStronglyCorrectRouterChoices = False

# If set to true, tests the experts under the assumption that the router is 100% accurate
shouldIdealizeMoeRouterForTests = False
# If set to true, just trains a regular transformer without my modifications
shouldUseUntouched = False

# If set to true makes the code print out how long various steps took
shouldPrintTimes = False

#The below should almost ALWAYS be false because we normally want top vs bottom division
shouldDoCloseToMidDivision = False


#do not change the below variable, it changes throughout the code and needs to start true
inTraining = False

# default values for how far along the principal axis the centroids of the two clusters
# are found
shiftCoefficient0 = 0.25
shiftCoefficient1 = 0.743

#Decide whether to import the modified model based on the build switch
if(shouldJustEstimateLoss):
    shouldUseUntouched = False
if(shouldUseUntouched):
    from model_untouched import GPTConfig, GPT
else:
    from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' or 'coleVal'
#init_from = 'resume'

expertNum = -1

if(shouldJustEstimateLoss):
    expertNum = -1

# wandb logging
wandb_log = False # disabled by default 
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'openwebtext' #this variable is usually overwritten from a config file
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024


n_layer = 6
n_head = 6
n_embd = 768

dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
binExtension = ""

# read in the expert number for the gpt model
if(sys.argv[1] == "config/train_gpt2.py"):
    if(len(sys.argv) > 2):
        try:
            expertNum = int(sys.argv[2])
            print("expertNum: " + str(expertNum))
        except:
            print("The second argument must be an integer")
            exit()
    binExtension = str(expertNum) + ".bin"
else:
    binExtension = ".bin"




# The config file is already loaded here, so if you assign any values to variables below
# here, they will override the config file.


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # Setup for dynamic parralel processing
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    print("ddp_local_rank: " + str(ddp_local_rank))
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



# poor man's data loader
data_dir = os.path.join('data', dataset)

# Setup to choose the centroids
maxMinSet = False
max_value = 0
min_value = 0
mid = 0
centroid0 = 0
centroid1 = 0
marginPoint0 = 0
marginPoint1 = 0
mins = 0
maxs = 0



# The below two variables are only useful for shakespeare char!
marginShiftCoefficient0 = 0.4
marginShiftCoefficient1 = 0.6

# This function finds the centroids of the clusters by performing a 
# simplified algorithm similar to principle component analysis, and then
# selecting points along the principal axis using the portions chosen
# in the configuration files.
def calculateClusterMeans(expertNum, model):
    global centroid0
    global centroid1
    global max_value
    global min_value
    global mid
    global mins
    global maxs
    global shiftCoefficient0
    global shiftCoefficient1
    global marginShiftCoefficient0
    global marginShiftCoefficient1
    global marginPoint0
    global marginPoint1
    global maxMinSet
    global binExtension
    fileName = "train" + binExtension
    data = np.memmap(os.path.join(data_dir, fileName), dtype=np.uint16, mode='r')
    
    if(not maxMinSet):
        maxMinSet = True
        # Initialize a variable to hold the maximum value
        max_value = 0
        min_value = 100000
        count = 0
        numValuesToCross = 1000 * 1000

        # Iterate through the data array, and look for the max, min,
        # and average word indeces.
        for value in data:
            mid += value
            count += 1
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value
            if(count % (100 * 1000) == 0):
                print("value count: " + str(count))
            if(count > numValuesToCross and shouldUseOpenWebText):
                break
        mid = int(mid / count)
        

        ckpt_path = os.path.join(out_dir, 'centroids.pt')

        # Make a list of all the word indeces from 0 up to max_value, and move 
        # it to the gpu with the model and its embeddings
        allTokens = torch.arange(0, max_value + 1)
        allTokens = allTokens.to(device)
        model = model.to(device)
        print("model is on device")
        tok_emb = getModelTransformer(model).wte(allTokens)
        print("tok_emb is calculated")
        tok_emb.to(device)
        centroids = torch.zeros(2, tok_emb.size(1))

        # Find the two points in embedding space storing the min and max possible 
        # values in the entire embedding space. The main line will run between these
        # points
        mins = torch.zeros(tok_emb.size(1))
        mins += 100000
        mins = mins.to(device)
        maxs = torch.zeros(tok_emb.size(1))
        maxs -= 100000
        maxs = maxs.to(device)
        tokenCount = 0

        mins = torch.min(tok_emb, dim=0)[0]
        maxs = torch.max(tok_emb, dim=0)[0]


        # Set the centroids of the clusters corresponding to the two experts
        # to two points on the main line such that the vocab is divided in half
        for i in range(tok_emb.size(1)):
            centroids[0][i] = mins[i] + (maxs[i] - mins[i])*shiftCoefficient0
            centroids[1][i] = mins[i] + (maxs[i] - mins[i])*shiftCoefficient1
        marginPoint0 = mins + (maxs - mins)*marginShiftCoefficient0
        marginPoint1 = mins + (maxs - mins)*marginShiftCoefficient1
        torch.save(centroids, ckpt_path)
        
        centroid0 = centroids[0]
        centroid1 = centroids[1]

        cluster0 = 0
        cluster1 = 0

        # Calculate the number of words that are closer to each centroid,
        # and if it isn't roughly half and half, print a message and exit
        centroid0Temp = centroids[0].unsqueeze(0).to(device)
        centroid1Temp = centroids[1].unsqueeze(0).to(device)
        dist0 = torch.norm(tok_emb - centroid0Temp, dim = -1)
        dist1 = torch.norm(tok_emb - centroid1Temp, dim = -1)
        cluster0 = torch.where(dist0 < dist1, 1, 0).sum()
        cluster1 = torch.where(dist1 < dist0, 1, 0).sum()

        absDiff = abs(cluster0 - cluster1)
        smaller = cluster0 if cluster0 < cluster1 else cluster1
        margin = 0.2*smaller


        mins = mins.to(device)
        maxs = maxs.to(device)
        centroid0 = centroid0.to(device)
        centroid1 = centroid1.to(device)

        if(absDiff > margin):
            print("cluster0: " + str(cluster0))
            print("cluster1: " + str(cluster1))
            print("absdiff: " + str(absDiff))
            print("margin: " + str(margin))
        
            print("Your cluster values are too far apart")
            print("You need to fix that before you can properly train a model")
            print("Shutting down")
            exit()
        
        if(shouldJustHelpPickClusterCenters):
            # Extra print statements 
            # Since this condition is only triggered when debugging
            print("cluster0: " + str(cluster0))
            print("cluster1: " + str(cluster1))
            exit()

# This function from nanogpt gets a batch to train on asynchronously
def get_batch(split):
    global maxMinSet
    global max_value
    global min_value
    global mid
    global clusterOne
    global clusterTwo
    global binExtension
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # Decide which file to get the data from
    if split == 'train':
        fileName = "train" + binExtension
        data = np.memmap(os.path.join(data_dir, fileName), dtype=np.uint16, mode='r')
    else:
        fileName = "val" + binExtension
        data = np.memmap(os.path.join(data_dir, fileName), dtype=np.uint16, mode='r')
    counts = []
    # count up how many of each type of value there are, and find the max, min, and average
    # word indeces from the training data
    for i in range(0, 256):
        counts.append(0)
    for(i, value) in enumerate(data):
        if(value > 0):
            break
    if(not maxMinSet):
        maxMinSet = True
        # Initialize a variable to hold the maximum value
        max_value = 0
        min_value = 100000
        count = 0
        numValuesToCross = 1000 * 1000
        # Iterate through the data array
        for value in data:
            mid += value
            count += 1
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value
            if(count > numValuesToCross and shouldUseOpenWebText):
                break
        mid = int(mid / count)
        #mid = int(max_value/2)

    # select random values to use to 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


import torch.nn as nn
import torch.nn.functional as F

# The below class can be used to convert embeddings back to the word index of the 
# word with the closest embedding of all the words in the vocabulary
class EmbeddingReverser:
    def __init__(self, embedding_layer):
        self.embedding_layer = embedding_layer
    
    def reverse(self, target_embeddings):
        # Flatten the target embeddings to a 2D tensor where each row is an embedding
        original_shape = target_embeddings.shape[:-1]
        flattened_target_embeddings = target_embeddings.view(-1, target_embeddings.shape[-1])
        
        # Compute the pairwise distances
        distances = torch.cdist(flattened_target_embeddings.unsqueeze(0), self.embedding_layer.weight.unsqueeze(0))
        
        # Find the indices of the minimum distances
        recovered_indices = torch.argmin(distances, dim=2).view(*original_shape)
        
        return recovered_indices
    

# Takes in the one digit values, does not take in logits
# if we are training an expert this function does nothing
# if we are training a router it transforms all the words
# into the word corresponding to the center of the cluster
# that they are a part of
def transformY(Y):
    global expertNum
    global embeddingModel
    global centroid1
    global centroid0
    global mins
    global maxs

    # get the embeddings from the word indeces,
    # and find their distances to the centroids
    tok_emb = getModelTransformer(embeddingModel).wte(Y)
    tok_emb.to(device)
    centroid0 = centroid0.to(device)
    centroid1 = centroid1.to(device)
    dis0 = torch.norm(tok_emb - centroid0, dim = -1)
    dis1 = torch.norm(tok_emb - centroid1, dim = -1)


    if(expertNum==-1):
        numDims = tok_emb.size()[-1]
        # transform every embedding into the centroid that it is closest to
        wherePutMin = torch.where(dis0 < dis1, 1, 0)
        wherePutMin = wherePutMin.unsqueeze(-1).repeat(1,1,numDims)
        transformedEmbeddings = torch.ones_like(tok_emb)
        transformedEmbeddings = torch.where(wherePutMin == 1, mins, maxs)

        # convert the transformed embeddings back into word indeces
        reverser = EmbeddingReverser(getModelTransformer(embeddingModel).wte)
        recovered_indeces = reverser.reverse(transformedEmbeddings)
        Y = recovered_indeces
    return Y

# Determines which indeces should not be trained on. For example,
# when expert 1 is training, if for a given word the answer is "dog"
# and dog is not in the vocabulary of expert 1, then expert 1 should
# not use this word to calculate the loss, since it doesn't need to 
# learn that.
def getZ(Y):
    global expertNum
    global max_value
    global mid
    global embeddingModel
    global centroid0
    global centroid1
    Z = torch.zeros_like(Y)
    Z.fill_(mid)

    
    if(not shouldDoCloseToMidDivision):
        Z = Y - Z
        #the below means we are training a non-expert model
        if(expertNum == -1):
            # make it just be all ones, since expert -1 (the router) needs to learn
            # to select the correct expert for every word
            Z = torch.ones_like(Y)
        else:
            startTime = time.time()
            # Calculate the distances to both centroids from each expert
            # and assign each word only to the expert who's centroid is
            # closest to said word
            tok_emb = getModelTransformer(embeddingModel).wte(Y)
            tok_emb.to(device)
            centroid0 = centroid0.to(device)
            centroid1 = centroid1.to(device)
            dis0 = torch.norm(tok_emb - centroid0, dim = -1)
            dis1 = torch.norm(tok_emb - centroid1, dim = -1)

            if(expertNum == 0):
                Z = dis0 < dis1
            #expert 1 handles the values that are greater than the mid value
            elif(expertNum == 1):
                Z = dis0 >= dis1

            else:
                print("invalid expert number: " +   str(expertNum))
                exit()
    elif(shouldDoCloseToMidDivision):
        # use word indeces instead of embeddings to determine which expert we assign each word to
        Z = Y - Z
        Z = Z.abs()
        allowedDis = (max_value - min_value)/4
        #the below means we are training a non-expert model, or a router
        if(expertNum == -1):
            # make it just be all ones
            Z = torch.ones_like(Y)
        elif(expertNum == 0):
            Z = torch.where(Z<allowedDis, 1, 0)
        elif(expertNum == 1):
            Z = torch.where(Z>=allowedDis, 1, 0)
    return Z


# The below is a custom training function I created when the router was not
# Working well to try to make it only focus on the words it wasn't already 
# Getting correct. The function didn't help much honestly
def getStronglyCorrectRouterChoices(logits, targets, embeddingModel):
    global mins
    global centroid0
    global centroid1

    # first, determine which centroid each output is closest to
    output = torch.argmax(logits, dim=2)

    outputEmbeddings = getModelTransformer(embeddingModel).wte(output)
    targetEmbeddings = getModelTransformer(embeddingModel).wte(targets)
    
    centroid0 = centroid0.to(device)
    centroid1 = centroid1.to(device)

    dis0 = torch.norm(targetEmbeddings - centroid0, dim = -1)
    dis1 = torch.norm(targetEmbeddings - centroid1, dim = -1)

    ShouldBeMin = torch.where(dis0 < dis1, 1, 0)

    # Calculate the distances of all output values to the centroids, and mark them
    # to be omitted from the loss function if they are already sufficiently close
    disToTarget = torch.norm(outputEmbeddings - targetEmbeddings, dim = -1)
    margin0ToTargetDistance = torch.norm(targetEmbeddings - marginPoint0, dim = -1)
    margin1ToTargetDistance = torch.norm(targetEmbeddings - marginPoint1, dim = -1)
    marginToTargetDistance = torch.where(ShouldBeMin == 1, margin0ToTargetDistance, margin1ToTargetDistance)
    closerThanMargin = torch.where(disToTarget < marginToTargetDistance, 1, 0)
    
    return closerThanMargin

isCalculatingValidationLoss = False
validationLossIterationCount = 0


# This function helps us calculate accuracy
def getCorrectRouterChoices(logits, targets, Z):
    # Set the output matrix to 1 at the locations where outputs matches targets
    output = torch.argmax(logits, dim=2)
    output = transformY(output)
    routerCorrectWords = torch.where(output == targets, 1, 0)
    routerCorrectWords = torch.where(Z == 1, routerCorrectWords, 0)
    return routerCorrectWords

# This function calculates accuracy for the router
def getPercentCorrectRouterChoices(logits, targets, Z, numTargetWords):
    #find which words were chosen correctly
    routerCorrectWords = getCorrectRouterChoices(logits, targets, Z)
    routerCorrectCountedWords = torch.where(routerCorrectWords == 1, 1, 0)
    #count correctly chosen words, and divide by the relevant number
    numRouterCorrectWords = routerCorrectCountedWords.sum()
    numRelevantTargetWords = torch.where(Z == 1, 1, 0).sum()
    routerPercentCorrect = float(numRouterCorrectWords)/numRelevantTargetWords
    return routerPercentCorrect

# Make sure expertNum is set correctly before calling this function
# as it just uses the global value, it doesn't take a parameter
def getPercentCorrectExpertWords(logits, targets, Z, numTargetWords):
    output = torch.argmax(logits, dim=2)
    # count the number of cases where output matches targets
    expertCorrectWords = torch.where(output == targets, 1, 0)
    expertCorrectWords = torch.where(Z == 1, expertCorrectWords, -1)
    expertCorrectCountedWords = torch.where(expertCorrectWords == 1, 1, 0)
    numExpertCorrectCountedWords = expertCorrectCountedWords.sum()
    # divide the number of correct words by the number of words assigned to the expert
    expertUncountedWords = torch.where(expertCorrectWords == -1, 1, 0)
    numExpertUncountedWords = expertUncountedWords.sum()
    expertPercentCorrect = float(numExpertCorrectCountedWords)/(numTargetWords - numExpertUncountedWords)
    return expertPercentCorrect

# The below function find % accuracy for a regular non moe model
def getPercentCorrect(y, logits):
    vals = logits.argmax(dim=-1)
    correct = torch.where(vals == y, 1, 0).sum()
    total = torch.ones_like(y).sum()
    percentCorrect = correct*100 / total
    return percentCorrect

# Just a function to print only if a certain build switch is True
def printTimeIfAppropriate(inStr):
    global shouldPrintTimes
    if(shouldPrintTimes):
        print(inStr)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_simple_loss():
    global validationLossIterationCount
    global inTraining
    inTraining = False
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # prepare to score for the split
        percentsCorrect = torch.zeros(eval_iters)
        startTime = 0
        hasPrinted = False
        startIter = 2
        tempStart = 0
        for k in range(eval_iters):
            # keep track of time
            if(k==startIter):
                startTime = time.time()
            tempStart2 = time.time()
            #retrieve a batch and print how long the synchronous part took
            X, Y = get_batch(split)
            printTimeIfAppropriate("time for one getting one batch: " + str(time.time() - tempStart2))
            # print how long the first few iterations take
            if(float(k-startIter)/eval_iters > 0.05 and not hasPrinted):
                print(f"percent done with {split} is: {float(k)/eval_iters}")
                diff = time.time() - startTime
                print("time gone by in seconds: ", diff)
                print("so estimated total time is ", (diff * eval_iters) / (float(k-startIter)) )
                print("split is: ", split)

                hasPrinted = True
            # print how long the previous iteration took
            if(not (tempStart == 0)):
                printTimeIfAppropriate("time for one iteration: " + str(time.time() - tempStart))
            tempStart = time.time()    
            tempStart2 = time.time()
            with ctx:
                logits = 0
                loss = 0
                # call the model and time how long this takes for comparison against the time
                # to retrieve a batch
                printTimeIfAppropriate("time before callModel\n", time.time())
                logits, loss = model(X, Y)

                printTimeIfAppropriate("time after callModel\n", time.time())

                printTimeIfAppropriate("time for callModel: " + str(time.time() - tempStart2))
                tempStart2 = time.time()
                # record the accuracy and add it and the loss to the appropriate lists
                percentCorrect = getPercentCorrect(Y, logits)
            percentsCorrect[k] = percentCorrect
            losses[k] = loss.item()
        out[split] = losses.mean()
        print("router percent correct for " + split + ": " + str(percentsCorrect.mean()))
    model.train()
    inTraining = True
    return out


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    global validationLossIterationCount
    global inTraining

    inTraining = False
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # prepare to score for the split
        percentsCorrect = torch.zeros(eval_iters)
        startTime = 0
        hasPrinted = False
        startIter = 2
        tempStart = 0

        for k in range(eval_iters):
            # keep track of time
            if(k==startIter):
                startTime = time.time()
            tempStart2 = time.time()
            #retrieve a batch and print how long the synchronous part took
            X, Y = get_batch(split)
            printTimeIfAppropriate("time for one getting one batch: " + str(time.time() - tempStart2))
            # print how long the first few iterations take
            if(float(k-startIter)/eval_iters > 0.05 and not hasPrinted):
                print(f"percent done with {split} is: {float(k)/eval_iters}")
                diff = time.time() - startTime
                print("time gone by in seconds: ", diff)
                print("so estimated total time is ", (diff * eval_iters) / (float(k-startIter)) )
                print("split is: ", split)
                hasPrinted = True

            numTargetWords = Y.size()[0] * Y.size()[1] #batch size * block size
            # Decide which words to mask out if any
            Z = getZ(Y)
            # print how long the previous iteration took
            if(not (tempStart == 0)):
                printTimeIfAppropriate("time for one iteration: " + str(time.time() - tempStart))
            tempStart = time.time()    
            tempStart2 = time.time()
            # transform Y to be just the two centroids if we are using the router, then print how long it took
            Y = transformY(Y)
            printTimeIfAppropriate("time for one transformation: " + str(time.time() - tempStart2))
            tempStart2 = time.time()
            # weight classes if using the router since they are imbalanced, and print how long it took
            if(expertNum == -1):
                Z = weightClasses(Y, Z)
            printTimeIfAppropriate("time for weightClasses: " + str(time.time() - tempStart2))
            with ctx:
                # call the model and time how long this takes for comparison against the time
                # to retrieve a batch
                tempStart2 = time.time()
                logits, loss = callModel(X, Y, Z, model)
                printTimeIfAppropriate("time for callModel: " + str(time.time() - tempStart2))
                tempStart2 = time.time()
                percentCorrect = 0
                # calculate the accuracy for the router or expert as appropriate
                if(not expertNum == -1):
                    totalNumWords = Z.size()[0] * Z.size()[1]
                    percentCorrect = getPercentCorrectExpertWords(logits, Y, Z, totalNumWords)
                routerPercentCorrect = 0
                if(expertNum == -1):
                    routerPercentCorrect = getPercentCorrectRouterChoices(logits, Y, Z, numTargetWords)
                    validationLossIterationCount += 1
                    percentCorrect = routerPercentCorrect
                
                printTimeIfAppropriate("time for getPercentCorrect: " + str(time.time() - tempStart2))
            # add the calculated loss and accuracy values to the lists
            losses[k] = loss.item()
            percentsCorrect[k] = percentCorrect
        # calculate the mean losses and accuracies
        out[split] = losses.mean()
        print("percent correct for " + split + ": " + str(percentsCorrect.mean()))
    model.train()
    inTraining = True
    return out

# estimates the loss for the whole system of all 3 models at once
@torch.no_grad()
def estimate_expert_loss():
    global ddp
    global expertNum
    global inTraining
    inTraining = False
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
        # load expert 0
        ckpt_path = os.path.join(out_dir, 'Expert0ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model0 = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

                
        model0.load_state_dict(state_dict)
        if block_size < model0.config.block_size:
            model0.crop_block_size(block_size)
            model_args['block_size'] = block_size # so that the checkpoint will have the right value
        model0.to(device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        # optimizer
        optimizer = model0.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        if init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None # free up memory

        # compile the model0
        if compile:
            print("compiling the model0... (takes a ~minute)")
            unoptimized_model0 = model0
            model0 = torch.compile(model0) # requires PyTorch 2.0

        # wrap model0 into DDP container
        if ddp:
            print("ddp is true")
            model0 = DDP(model0, device_ids=[ddp_local_rank])

        # load expert 1
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
        ckpt_path = os.path.join(out_dir, 'Expert1ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model1 = GPT(gptconf)
        state_dict = checkpoint['model']
                # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

                
        model1.load_state_dict(state_dict)
        if block_size < model1.config.block_size:
            model1.crop_block_size(block_size)
            model_args['block_size'] = block_size # so that the checkpoint will have the right value
        model1.to(device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        # optimizer
        optimizer = model1.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        if init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None # free up memory

        # compile the model1
        if compile:
            print("compiling the model1... (takes a ~minute)")
            unoptimized_model1 = model1
            model1 = torch.compile(model1) # requires PyTorch 2.0

        # wrap model1 into DDP container
        if ddp:
            print("ddp is true")
            model1 = DDP(model1, device_ids=[ddp_local_rank])
        

        for i in range(eval_iters):
            # set expertNum before getting and transforming data so it gets transformed correctly
            expertNum = -1
            X, Y = get_batch(split)
            Z = torch.ones_like(Y)

            targetsRouter = Y
            targetsRouter = transformY(targetsRouter)
            with ctx:
                # get the number of words, and the router's choices
                numTargetWords = Y.size()[0] * Y.size()[1] #batch size * block size
                logitsRouter, lossRouter = callModel(X, targetsRouter, Z, model)
                
                # mark the words where the router chose expert 0
                routerChoices = torch.argmax(logitsRouter, dim=2)
                routerChoicesEmbeddings = getModelTransformer(embeddingModel).wte(routerChoices)
                dis0 = torch.norm(routerChoicesEmbeddings - centroid0, dim = -1)
                dis1 = torch.norm(routerChoicesEmbeddings - centroid1, dim = -1)
                routerChose0 = torch.where(dis0 < dis1, 1, 0)
                
                #evaluate the accuracy of the router
                routerPercentCorrect = getPercentCorrectRouterChoices(logitsRouter, targetsRouter, Z, numTargetWords)
                
                #DO NOT UNCOMMENT FOLLOWING LINE EXCEPT FOR TESTS! It adjust the router output here to falsely be always right
                #routerOut = Y.clone()

                # copy targets and transform them into the correct domains for each expert
                targets0 = Y
                targets1 = Y
                expertNum = 0
                targets0 = transformY(targets0)
                expertNum = 1
                targets1 = transformY(targets1)
                
                # see which words to mask for each expert, then call each expert and find accuracy
                expertNum = 0
                Z0 = getZ(Y)
                Z0.to(device)
                logits0, loss0 = callModel(X, targets0, Z0, model0)
                expert0PercentCorrect = getPercentCorrectExpertWords(logits0, targets0, Z0, numTargetWords)
                expertNum = 1
                Z = getZ(Y)
                Z.to(device)
                logits1, loss1 = callModel(X, targets1, Z, model1)
                expert1PercentCorrect = getPercentCorrectExpertWords(logits1, targets1, Z, numTargetWords)

                # quickly check if there were any words passed to the model not found in either expert's vocab
                biggerZ = torch.where(Z>Z0, Z, Z0)
                isOne = torch.where(biggerZ<1, 1, 0).any()
                if(isOne):
                    print("there is an untrained word")
                    print("biggerZ: " + str(biggerZ))
                    exit()


                
                # Ensure routerOut has the same shape as logits0 and logits1
                routerChose0_expanded = routerChose0.unsqueeze(-1).repeat(1,1,logits0.size(-1))

                # Combine the logits tensors based on the router output
                logitsCombined = torch.where(routerChose0_expanded == 1, logits0, logits1)
                moePercentCorrect = getPercentCorrect(Y, logitsCombined)
                logits_flat = logitsCombined.view(-1, logitsCombined.size(-1))

                #transform targets into the expert domain so we can evaluate the output:
                targets = torch.where(Y <= mid, targets0, targets1)
                targets_flat = targets.view(-1)
                loss = model.get_cross_entropy(logits_flat, targets_flat, ignore_index=-1)
                if(i%10==0):
                    # print a detailed log of the score of the whole system, and each of its components
                    print("routerloss: " + str(lossRouter))
                    print("loss0: " + str(loss0))
                    print("loss1: " + str(loss1))
                    print("loss: " + str(loss))
                    print("routerPercentCorrect: " + str(routerPercentCorrect))
                    print("expert0PercentCorrect: " + str(expert0PercentCorrect))
                    print("expert1PercentCorrect: " + str(expert1PercentCorrect))
                    print("moePercentCorrect: " + str(moePercentCorrect))
                    print("have now completed: " + str(i) + " of " + str(eval_iters) + " iterations")
                    print("current split is: " + str(split))            
            # add the loss to the list of losses
            losses[i] = loss
        # print out the average loss
        out[split] = losses.mean()
        print("loss for current split is: " + str(out[split]))
    model.train()
    inTraining = True
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

#only where Z contains a 1 does the code actually pay attention to the value. 0 means mask out the value
def weightClasses(Y, Z):
    # make note of which indeces correspond to the max and min embeddings
    minEquivalent = 3
    maxEquivalent = 11

    # count up how many times the Y value corresponds to the min and max
    numMins = torch.where(Y == minEquivalent, 1, 0).sum()
    numMaxs = torch.where(Y == maxEquivalent, 1, 0).sum()
    # find the portion of min values and max values in Y
    totalNum = numMins + numMaxs
    portionMins = float(numMins)/totalNum
    portionMaxs = float(numMaxs)/totalNum
    # make note of which portion is less and which is more
    greaterPortion = portionMins
    lesserPortion = portionMaxs
    moreFrequentVal = minEquivalent
    if(portionMaxs > portionMins):
        greaterPortion = portionMaxs
        lesserPortion = portionMins 
        moreFrequentVal = maxEquivalent
    # make the portions the same by keeping only some of the values from the greater portion
    # decide how many of the values from the greater portion to keep
    portionToKeep = lesserPortion/greaterPortion
    portionToKeep = portionToKeep.to(device)
    # randomly select which of the values from the greater potion to drop
    randomNumbers = torch.zeros_like(Z)
    randomNumbers = torch.randint(1, 101, randomNumbers.shape)
    randomNumbers = randomNumbers.to(device)
    #make it so Z gets set to 0 only where Y == moreFrequentVal and randomNumbers > portionToKeep * 100
    zZeroes = torch.zeros_like(Z)
    zZeroes.to(device)
    #make zZeroes contain 1's where either condition is false
    zZeroes = torch.where(Y != moreFrequentVal, 1, zZeroes)
    zZeroes = torch.where(randomNumbers <= (portionToKeep * 100), 1, zZeroes)
    #make Z contain 0's where both conditions are true
    Z = torch.where(zZeroes == 0, 0, Z)    
    return Z

def callModel(X, Y, Z, model):
    global inTraining
    # call the model, and pass in the mask only if we are working with the modified model
    if(shouldUseUntouched):
        logits, loss = model(X, Y)
        return logits, loss
    else:
        logits, loss = model(X, Y, Z)
        return logits, loss

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
if(shouldJustEstimateLoss or shouldReloadRouter):
    init_from = 'resume'
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model = ""
checkpoint = ""
model_args = ""

def initGpt2ViaLibrary():
    global init_from
    tempInitFrom = init_from
    init_from = 'gpt2'
    model_args = {}
    model = ""
    if init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    #move the model the gpu, and set init_from back to its original value
    model.to(device)
    init_from = tempInitFrom
    print("model.wpe.weight.size() 3: " + str(model.transformer.wpe.weight.size()))
    return model

def initAnyModel(fileName):
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, fileName)

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    model_args = {}
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        print("checkpoint_model_args[" + k + "]: " + str(type(checkpoint_model_args[k]))+ " " + str(checkpoint_model_args[k]))
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    anyModel = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    anyModel.load_state_dict(state_dict)

    # crop down the model block size if desired, using model surgery
    if block_size < anyModel.config.block_size:
        anyModel.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    anyModel.to(device)
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        anyModel = torch.compile(anyModel) # requires PyTorch 2.0
    return anyModel

# get the transformer component of the model so we can work with it directly for purposes
# such as setting its embedding component to the pre-saved one
def getModelTransformer(model):
    transformer = ""
    # get the transformer, whether we are using ddp, or not
    if type(model) == torch.nn.parallel.DistributedDataParallel:
        transformer = model.module.transformer
    else:
        transformer = model.transformer
    return transformer

# This initializes a model for training or evaluation. Which model it initializes
# is determined by the global variable expertNum
def initModel():
    global model
    global checkpoint
    global model_args
    global dropout
    '''
    this function doesn't change the value of embeddingModel, 
    nor does it load from embedding model, it just makes sure the dimensions
    of the two models are alligned if initializing from scratch, that's all
    it does with embeddingModel
    '''
    global embeddingModel #READ THE ABOVE COMMENT
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to " + str(embeddingModel.config.vocab_size))
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else embeddingModel.config.vocab_size
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'coleVal':
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)

        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print("expertNum is: " + str(expertNum))
        
        print(f"Resuming training from {out_dir}")
        toResumeName = 'Expert' + str(expertNum) + 'ckpt.pt'
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, toResumeName)
        print("ckpt_path: " + ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        print("state dict type: " + str(type(state_dict)))
        print("model type: " + str(type(model)))
        for item in state_dict:
            print("item: " + str(item))



        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)


def reloadEmbeddingModelForTests(embeddingModel):
    # this function reloads the embedding model after saving it to check that it is working
    ckpt_path = os.path.join(out_dir, 'gpt2File.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model0 = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

            
    embeddingModel.load_state_dict(state_dict)
    print("embedding model post load wpe size: " + str(embeddingModel.transformer.wpe.weight.size()))
    
embeddingModel = ""
# handle the case where the embedding model comes from the internet 
# instead of from another model that we have pretrained
if((shouldUseWordShakeSpeare or shouldUseOpenWebText)):
    if(shouldJustSaveGPTFile):
        # download and save the gpt pre-trained embeddings to a file then exit
        embeddingModel = initGpt2ViaLibrary()
        print("embedding model wpe size: " + str(embeddingModel.transformer.wpe.weight.size()))
        model_args = {}
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(embeddingModel.config, k)
        checkpoint = {
            'model': embeddingModel.state_dict(),
            'model_args': model_args,
            'config': embeddingModel.config,
        }
        print(f"saving checkpoint to {out_dir}")
        checkPointName = 'gpt2File.pt'
        torch.save(checkpoint, os.path.join(out_dir, checkPointName))
        exit()
    else:
        # load the model
        embeddingModel = initAnyModel('gpt2File.pt')

# The below conditional statement just displays how many words are assigned to each cluster
# Useful for when manually setting the shift coefficients to split the words between experts
if(shouldJustHelpPickClusterCenters):
    calculateClusterMeans(-1, embeddingModel)
    exit()

initModel()

# retrieve the embeddings from the shakespeare character model that we trained
if(not shouldUseWordShakeSpeare and not shouldUseOpenWebText):
    embeddingModel = initAnyModel('embedding_only_otherwise_useless.pt')
    print("initialized embedding model for char")

embeddingModel.eval()
if((not shouldJustEstimateLoss) and (not shouldEstimateMoeLoss) and expertNum == -1):
    #no need to create reference to transformer here because have not yet created a DDP container
    if(init_from == 'scratch'):
        getModelTransformer(model).wte = copy.deepcopy(getModelTransformer(embeddingModel).wte)
    getModelTransformer(model).wte.requires_grad = True
    print("passed initModel")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    if(not shouldJustEstimateLoss):
        print("optimizer param list: " + str(optimizer.param_groups))
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model

    model = torch.compile(model) # requires PyTorch 2.0
    unoptimized_embeddingModel = embeddingModel

calculateClusterMeans(expertNum, embeddingModel)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    embeddingModel = DDP(embeddingModel, device_ids=[ddp_local_rank])


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
Z = getZ(Y)
Y = transformY(Y)

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# make note of the last percent accuracy we had so that we can quickly save the model if 
# it rapidly jumps over a certain acurracy threshold. Useful for debugging because 
# every time I found that it quickly learned above 95% accuracy there was a bug
lastPercentCorrect = 0
percentThreshold = 0.95

# estimate loss and exit without training if appropriate
if(shouldJustEstimateLoss):
    losses = ""
    if(shouldEstimateMoeLoss):
        print("estimating expert loss")
        losses = estimate_expert_loss()
    else:
        losses = estimate_loss()
    print(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    exit()

#make note of two start times, to be changed with different frequencies later
runStartTime = time.time()
tempStart = time.time()
while True:
    # the following print will be meaningless the first iteration
    print("time for one iteration: " + str(time.time() - tempStart))
    tempStart = time.time()
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    
    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0) and master_process and testConfigs.shouldEvaluateAtAll:
        isCalculatingValidationLoss = True
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        # save the checkpoint only if validation loss decreased, or if a build switch says
        # to always do so
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                # save to two checkpoints files so that if we get interrupted part way through
                # our save we don't end up corrupting our only copy of our model
                print(f"saving backup (called second) checkpoint to {out_dir}")
                checkPoint2Name = 'Expert' + str(expertNum) + 'Secondckpt.pt'
                torch.save(checkpoint, os.path.join(out_dir, checkPoint2Name))
                print("done saving backup (called second) checkpoint")
                secs = 2
                print("sleeping for " + str(secs) + " seconds")
                time.sleep(secs)
                print(f"saving checkpoint to {out_dir}")
                checkPointName = 'Expert' + str(expertNum) + 'ckpt.pt'
                torch.save(checkpoint, os.path.join(out_dir, checkPointName))
                print("done saving regular checkpoint")

        isCalculatingValidationLoss = False
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if(expertNum == -1):
                # weight the classes so that the router doesn't just become a dummy classifier
                Z = weightClasses(Y, Z)
                if(shouldIgnoreStronglyCorrectRouterChoices):
                    # decide which values the router is already so sure of that it shouldn't learn from them
                    print("shouldIgnoreStronglyCorrectRouterChoices is true")
                    with torch.no_grad():
                        logits, loss = callModel(X, Y, Z, model)
                        learnFromCorrectOptions = [0, 1]
                        rand = random.choice(learnFromCorrectOptions)
                        if(rand==0):
                            if(shouldIgnoreStronglyCorrectRouterChoices):
                                Ztemp = getStronglyCorrectRouterChoices(logits, Y, embeddingModel)
                                Z = torch.where(Ztemp==1, 0, Z)
            logits, loss = callModel(X, Y, Z, model)
            
            # now calculate the percent accuracy according to the model type (router or expert)
            numTargetWords = Y.size()[0] * Y.size()[1] #batch size * block size
            percentCorrect = 0
            if(expertNum == -1):
                percentCorrect = getPercentCorrectRouterChoices(logits, Y, Z, numTargetWords)
                lastPercentCorrect = percentCorrect
            else:
                percentCorrect = getPercentCorrectExpertWords(logits, Y, Z, numTargetWords)
                lastPercentCorrect = percentCorrect
            # divide loss by number of gradient accumulation steps if there are several
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        runEndtime = time.time()
        runTime = runEndtime - runStartTime
        prepStartTime = time.time()
        # determine which values to mask, and transform Y into the domain of the current model
        Z = getZ(Y)
        Y = transformY(Y)
        # time how long these transformations took
        prepEndTime = time.time()
        prepTime = prepEndTime - prepStartTime
        runStartTime = time.time()
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16    
    getModelTransformer(embeddingModel).wte.zero_grad()
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        if(not shouldOnlyShowEvalOut):
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            print("percentCorrect: " + str(percentCorrect))

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        exit()
        break

if ddp:
    destroy_process_group()

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




shouldJustSaveGPTFile = False

shouldJustHelpPickClusterCenters = False

#Never set the below to true in this script, it can only be set to true in the gpt config scripts
haveReadShiftCoefficients = False

#The below should NEVER be true if you want to actually run the router
shouldSimplifyForTesting = False
#the above runs a regular model in the routers place that approaches the router

shouldUseOpenWebText = False
shouldUseWordShakeSpeare = False

shouldOnlyShowEvalOut = False

shouldJustEstimateLoss = False
shouldEstimateMoeLoss = False

shouldReloadRouter = False
shouldIgnoreStronglyCorrectRouterChoices = False

shouldIdealizeMoeRouterForTests = False
shouldUseUntouched = False

shouldPrintTimes = False
#if(shouldSimplifyForTesting):
#    shouldUseUntouched = True


shouldUseEmbeddingDivision = True
#The below should almost ALWAYS be false because we normally want top vs bottom division
shouldDoCloseToMidDivision = False


#do not change the below variable, it changes throughout the code and needs to start true
inTraining = False

#default values for shakespeare char
shiftCoefficient0 = 0.25
shiftCoefficient1 = 0.743

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
#wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
#n_layer = 12 * 2
#n_head = 12
#n_embd = 768
n_layer = int(12/2 + 0.5)
n_head = int(12/2 + 0.5)
n_embd = int(768+0.5)

dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
#max_iters = 0 #THIS GETS OVERWRITTEN BY THE CONFIG FILE THEN RR_OVERWRITTEN LATER BY US!
#max_iters = 600000 # total number of training iterations
#max_iters = 1000 # total number of training iterations
#max_iters = 200 #this gets overridden from the config file and we need to re-overwrite it later
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




#Cole! this is where you assign the variables that you want to override the config file
if(expertNum == -1):
    dropout = 0.0

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
#ddp = False
#device = 'cpu'
if ddp:
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


def k_means_balanced(data, k=2, max_iter=100):
    # Randomly initialize centroids
    indices = torch.randperm(data.size(0))[:k]
    centroids = data[indices]

    for _ in range(max_iter):
        # Step 2: Assign clusters
        dists = torch.cdist(data, centroids)
        cluster_assignments = torch.argmin(dists, dim=1)

        # Step 3: Update centroids
        new_centroids = torch.stack([data[cluster_assignments == i].mean(dim=0) for i in range(k)])

        # Step 4: Balancing clusters
        if _ < max_iter - 1:  # Skip on last iteration
            for i in range(k):
                members = data[cluster_assignments == i]
                other_members = data[cluster_assignments != i]
                if members.size(0) > data.size(0) // k:
                    # More members than the balance allows
                    excess = members.size(0) - data.size(0) // k
                    distances_to_other_centroid = torch.cdist(members, new_centroids[1-i].unsqueeze(0))
                    # Indices of the farthest 'excess' points to reassign
                    indices_to_reassign = distances_to_other_centroid.squeeze().topk(excess, largest=True).indices
                    cluster_assignments[cluster_assignments == i][indices_to_reassign] = 1 - i

        # Check for convergence (if centroids do not change)
        if torch.allclose(new_centroids, centroids, atol=1e-5):
            break
        centroids = new_centroids

    return centroids, cluster_assignments

# poor man's data loader
data_dir = os.path.join('data', dataset)
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



#Warning! The below two variables are only useful for shakespeare char!
marginShiftCoefficient0 = 0.4
marginShiftCoefficient1 = 0.6

'''
if(shouldUseOpenWebText and not haveReadShiftCoefficients):
    print("Cole, you are now on openwebtext")
    print("So you need to fix your 2 shift coefficients")
    exit()
if(shouldUseWordShakeSpeare and not haveReadShiftCoefficients):
    print("Cole, you are now on word shakespeare")
    print("So you need to fix your 2 shift coefficients")
    exit()
'''

#the below two variables don't change later

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

        # Iterate through the data array
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

        # Load the centroids if the file exists
        print("mid: " + str(mid))
        print("max_value: " + str(max_value))
        print("min_value: " + str(min_value))
        allTokens = torch.arange(0, max_value + 1)
        allTokens = allTokens.to(device)
        model = model.to(device)
        print("model is on device")
        tok_emb = getModelTransformer(model).wte(allTokens)
        print("tok_emb is calculated")
        tok_emb.to(device)
        centroids = torch.zeros(2, tok_emb.size(1))

        mins = torch.zeros(tok_emb.size(1))
        mins += 100000
        mins = mins.to(device)
        print("mins created")
        maxs = torch.zeros(tok_emb.size(1))
        maxs -= 100000
        maxs = maxs.to(device)
        tokenCount = 0

        mins = torch.min(tok_emb, dim=0)[0]
        maxs = torch.max(tok_emb, dim=0)[0]
        print("mins.size: " + str(mins.size()))
        '''
        for token in tok_emb:
            tokenCount += 1
            if(tokenCount % 10 == 0):
                print("token count: " + str(tokenCount))
                print("token_emb_size: " + str(tok_emb.size()))
                print("size of token: " + str(token.size(0)))
            for i in range(token.size(0)):
                if token[i] < mins[i]:
                    mins[i] = token[i]
                if token[i] > maxs[i]:
                    maxs[i] = token[i]
        '''
                    
        for i in range(tok_emb.size(1)):
            print("i: " + str(i))
            print("centroids[0].size():" + str(centroids[0].size()))
            print("mins.size():" + str(mins.size()))
            print("maxs.size():" + str(maxs.size()))
            centroids[0][i] = mins[i] + (maxs[i] - mins[i])*shiftCoefficient0
            centroids[1][i] = mins[i] + (maxs[i] - mins[i])*shiftCoefficient1
        marginPoint0 = mins + (maxs - mins)*marginShiftCoefficient0
        marginPoint1 = mins + (maxs - mins)*marginShiftCoefficient1
        #centroids, assignments = k_means_balanced(tok_emb, k=2, max_iter=1000)
        torch.save(centroids, ckpt_path)
        
        centroid0 = centroids[0]
        centroid1 = centroids[1]
        #print("centroid0: " + str(centroid0))
        #print("centroid1: " + str(centroid1))

        #exit()
        #if os.path.exists(ckpt_path):
        #    loaded_centroids = torch.load(ckpt_path)
        cluster0 = 0
        cluster1 = 0

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
        dis0Diff = (mins - centroid0).norm()
        dis1Diff = (maxs - centroid1).norm()

        if(absDiff > margin):
            print("cluster0: " + str(cluster0))
            print("cluster1: " + str(cluster1))
            print("dis0Diff: " + str(dis0Diff))
            print("dis1Diff: " + str(dis1Diff))
            print("absdiff: " + str(absDiff))
            print("margin: " + str(margin))
        
            print("Cole, your cluster values are too far apart")
            print("You need to fix that before you can properly train an ai")
            print("Shutting down")
            exit()
        
        if(shouldJustHelpPickClusterCenters):
            print("cluster0: " + str(cluster0))
            print("cluster1: " + str(cluster1))
            print("dis0Diff: " + str(dis0Diff))
            print("dis1Diff: " + str(dis1Diff))
            exit()

# the expert num parameter is used only to determine whether we can use expert -1's embeddng space to calculate the cluster means
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
    if split == 'train':
        fileName = "train" + binExtension
        data = np.memmap(os.path.join(data_dir, fileName), dtype=np.uint16, mode='r')
    else:
        fileName = "val" + binExtension
        data = np.memmap(os.path.join(data_dir, fileName), dtype=np.uint16, mode='r')
    counts = []
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
        # Iterate through the data array
        for value in data:
            mid += value
            count += 1
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value
        mid = int(mid / count)
        #mid = int(max_value/2)

    ix = torch.randint(len(data) - block_size, (batch_size,))
    '''
    maxTrainIndex = 0
    maxIndex = len(data)
    if split == 'train':
        valData = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        maxIndex = len(valData)

    ix = torch.randint(maxIndex - block_size, (batch_size,))
    '''
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        #if ddp:
        #x.pin_memory().to('cuda:1', non_blocking=True), y.pin_memory().to('cuda:1', non_blocking=True)
        #x.pin_memory().to('cuda:2', non_blocking=True), y.pin_memory().to('cuda:2', non_blocking=True)
        #x.pin_memory().to('cuda:3', non_blocking=True), y.pin_memory().to('cuda:3', non_blocking=True)
        #x, y = x.pin_memory().to('cuda:0', non_blocking=True), y.pin_memory().to('cuda:0', non_blocking=True)

        #else:
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


import torch.nn as nn
import torch.nn.functional as F

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
    
def printIfPCR(input):
    global inPercentCorrectRouterChoices
    if(inPercentCorrectRouterChoices):
        #print(input)
        junk = 2

transformYCount = 0
#email address: sthompson@uvic.ca  press 3 then 1
#francisco: fcanjura@uvic.ca
#takes in the one digit values, does not take in logits
def transformY(Y):
    global transformYCount
    global expertNum
    global embeddingModel
    global centroid1
    global centroid0
    global mins
    global maxs
    global inPercentCorrectRouterChoices

    startTime = time.time()
    #print("Y dim: " + str(Y.size()))
    #exit()

    tok_emb = getModelTransformer(embeddingModel).wte(Y)
    tok_emb.to(device)
    centroid0 = centroid0.to(device)
    centroid1 = centroid1.to(device)
    dis0 = torch.norm(tok_emb - centroid0, dim = -1)
    dis1 = torch.norm(tok_emb - centroid1, dim = -1)


    if(expertNum==-1):
        numDims = tok_emb.size()[-1]
        #transformedEmbeddings = torch.zeroes_like(tok_emb)
        wherePutMin = torch.where(dis0 < dis1, 1, 0)
        printIfPCR("dis0: " + str(dis0))
        printIfPCR("dis1: " + str(dis1))
        printIfPCR("wherePutMin: " + str(wherePutMin))
        wherePutMin = wherePutMin.unsqueeze(-1).repeat(1,1,numDims)
        printIfPCR("wherePutMin post: " + str(wherePutMin))
        #printIfInPercentCorrectRouterChoices("wherePutMin shape: " + str(wherePutMin.size()))
        #printIfInPercentCorrectRouterChoices("mins shape: " + str(mins.size()))
        #printIfInPercentCorrectRouterChoices("maxs shape: " + str(maxs.size()))
        transformedEmbeddings = torch.ones_like(tok_emb)
        transformedEmbeddings = torch.where(wherePutMin == 1, mins, maxs)
        printIfPCR("tok_emb: " + str(tok_emb))
        printIfPCR("transformedEmbeddings: " + str(transformedEmbeddings))

        # Instantiate the reverser
        reverser = EmbeddingReverser(getModelTransformer(embeddingModel).wte)

        # Reverse the first embedding
        recovered_indeces = reverser.reverse(transformedEmbeddings)
        printIfPCR("recovered_indeces: " + str(recovered_indeces))
        Y = recovered_indeces
        transformYCount += 1


    '''
    centroid = centroid0
    if(expertNum == 0):
        centroid = centroid0
    
        #Y = Y * 2
    elif(expertNum == 1):
        centroid = centroid1

    tok_emb = getModelTransformer(baseModel).wte(Y)
    tok_emb.to(device)
    diffs = 
        #Y = (Y - mid) * 2
    '''
    return Y

def untransformY(Y):
    global expertNum
    '''
    if(expertNum == 0):
        Y = Y * mid/float(max_value)
    elif(expertNum == 1):
        Y = Y * mid/float(max_value) + mid
    '''
    return Y

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
            # make it just be all ones
            Z = torch.ones_like(Y)
        else:
            startTime = time.time()
            tok_emb = getModelTransformer(embeddingModel).wte(Y)
            tok_emb.to(device)
            centroid0 = centroid0.to(device)
            centroid1 = centroid1.to(device)
            dis0 = torch.norm(tok_emb - centroid0, dim = -1)
            dis1 = torch.norm(tok_emb - centroid1, dim = -1)
            #print("dis0: " + str(dis0))
            #print("shape of dis0: " + str(dis0))

            if(expertNum == 0):
                #print("shape of dis0: " + str(dis0.size()))
                #Z = (Z <= 0).int()
                Z = dis0 < dis1
                #Z = torch.where(torch.cdist(tok_emb, centroid0) < torch.cdist(tok_emb, centroid1), 1, 0)
                #print("z size: " + str(Z.size()))
                #exit()

                #print("time to calculate Z: " + str(time.time() - startTime))
                #print("start time: " + str(startTime))
                #exit()
            #expert 1 handles the values that are greater than the mid value
            elif(expertNum == 1):
                #Z = (Z > 0).int()
                Z = dis0 >= dis1
                #Z = torch.where(torch.cdist(tok_emb, centroid1) <= torch.cdist(tok_emb, centroid0), 1, 0)


            else:
                print("invalid expert number: " +   str(expertNum))
                exit()
    elif(shouldDoCloseToMidDivision):
        Z = Y - Z
        Z = Z.abs()
        allowedDis = (max_value - min_value)/4
        #the below means we are training a non-expert model
        if(expertNum == -1):
            # make it just be all ones
            Z = torch.ones_like(Y)
        elif(expertNum == 0):
            Z = torch.where(Z<allowedDis, 1, 0)
        elif(expertNum == 1):
            Z = torch.where(Z>=allowedDis, 1, 0)
    return Z

def printYandZ(Y, Z):
    print("mid: " + str(mid))
    print("Y:")
    print(Y.size())
    print(Y)
    print("Z:")
    print(Z.size())
    print(Z)
    exit()

def getStronglyCorrectRouterChoices(logits, targets, embeddingModel):
    global mins
    global centroid0
    global centroid1

    numDims = logits.size()[-1]
    output = torch.argmax(logits, dim=2)

    outputEmbeddings = getModelTransformer(embeddingModel).wte(output)
    targetEmbeddings = getModelTransformer(embeddingModel).wte(targets)
    
    centroid0 = centroid0.to(device)
    centroid1 = centroid1.to(device)
    dis0 = torch.norm(targetEmbeddings - centroid0, dim = -1)
    dis1 = torch.norm(targetEmbeddings - centroid1, dim = -1)

    ShouldBeMin = torch.where(dis0 < dis1, 1, 0)

    #print("ShouldBeMin: " + str(ShouldBeMin))
    #exit()
    #shouldBeMin = shouldBeMin.unsqueeze(-1).repeat(1,1,numDims)

    disToTarget = torch.norm(outputEmbeddings - targetEmbeddings, dim = -1)
    margin0ToTargetDistance = torch.norm(targetEmbeddings - marginPoint0, dim = -1)
    margin1ToTargetDistance = torch.norm(targetEmbeddings - marginPoint1, dim = -1)
    #print("should be min shape: " + str(ShouldBeMin.size()))
    #print("margin0ToTargetDistance shape: " + str(margin0ToTargetDistance.size()))
    #print("margin1ToTargetDistance shape: " + str(margin1ToTargetDistance.size()))
    marginToTargetDistance = torch.where(ShouldBeMin == 1, margin0ToTargetDistance, margin1ToTargetDistance)
    closerThanMargin = torch.where(disToTarget < marginToTargetDistance, 1, 0)
    
    #the below line effectively disables this function so we can test how well it learns without this
    #closerThanMargin = torch.zeros_like(closerThanMargin)
    return closerThanMargin

isCalculatingValidationLoss = False
validationLossIterationCount = 0
def printRouterOutputAndTargets(output, targets):
    global isCalculatingValidationLoss
    if(isCalculatingValidationLoss):
        thereAreFours = torch.where(output == 4, 1, 0).any()
        if(thereAreFours):
            breakpoint = 10
        else:
            breakpoint = 10
        #print("outputs: " + str(output))
        #print("targets: " + str(targets))

def getCorrectRouterChoices(logits, targets, Z):
    output = torch.argmax(logits, dim=2)
    #print("transformYCount: " + str(transformYCount))
    #exit()
    #print("output size: " + str(output.size()))
    #print("targets size: " + str(targets.size()))
    #exit()
    output = transformY(output)
    printRouterOutputAndTargets(output, targets)
    routerCorrectWords = torch.where(output == targets, 1, 0)
    routerCorrectWords = torch.where(Z == 1, routerCorrectWords, 0)
    return routerCorrectWords

inPercentCorrectRouterChoices = False
def getPercentCorrectRouterChoices(logits, targets, Z, numTargetWords):
    global inPercentCorrectRouterChoices
    inPercentCorrectRouterChoices = True
    routerCorrectWords = getCorrectRouterChoices(logits, targets, Z)
    routerCorrectCountedWords = torch.where(routerCorrectWords == 1, 1, 0)
    numRouterCorrectWords = routerCorrectCountedWords.sum()
    numRelevantTargetWords = torch.where(Z == 1, 1, 0).sum()
    routerPercentCorrect = float(numRouterCorrectWords)/numRelevantTargetWords
    inPercentCorrectRouterChoices = False
    return routerPercentCorrect

#make sure expertNum is set correctly before calling this function
def getPercentCorrectExpertWords(logits, targets, Z, numTargetWords):
    output = torch.argmax(logits, dim=2)
    untransformY(output)
    expertCorrectWords = torch.where(output == targets, 1, 0)
    expertCorrectWords = torch.where(Z == 1, expertCorrectWords, -1)
    expertCorrectCountedWords = torch.where(expertCorrectWords == 1, 1, 0)
    numExpertCorrectCountedWords = expertCorrectCountedWords.sum()
    expertUncountedWords = torch.where(expertCorrectWords == -1, 1, 0)
    numExpertUncountedWords = expertUncountedWords.sum()
    expertPercentCorrect = float(numExpertCorrectCountedWords)/(numTargetWords - numExpertUncountedWords)
    return expertPercentCorrect

def getPercentCorrect(y, logits):
    vals = logits.argmax(dim=-1)
    correct = torch.where(vals == y, 1, 0).sum()
    total = torch.ones_like(y).sum()
    percentCorrect = correct*100 / total
    return percentCorrect


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
        percentsCorrect = torch.zeros(eval_iters)
        startTime = 0
        hasPrinted = False
        startIter = 2
        tempStart = 0
        #Z = torch.zeros_like(Y)
        for k in range(eval_iters):
            if(k==startIter):
                startTime = time.time()
            tempStart2 = time.time()
            X, Y = get_batch(split)
            printTimeIfAppropriate("time for one getting one batch: " + str(time.time() - tempStart2))
            if(float(k-startIter)/eval_iters > 0.05 and not hasPrinted):
                print(f"percent done with {split} is: {float(k)/eval_iters}")
                diff = time.time() - startTime
                print("time gone by in seconds: ", diff)
                print("so estimated total time is ", (diff * eval_iters) / (float(k-startIter)) )
                print("split is: ", split)

                hasPrinted = True
            #numTargetWords = Y.size()[0] * Y.size()[1] #batch size * block size
            Z = getZ(Y)
            
            if(not (tempStart == 0)):
                printTimeIfAppropriate("time for one iteration: " + str(time.time() - tempStart))
            tempStart = time.time()    
            tempStart2 = time.time()
            #Y = transformY(Y)
            #print("time for one transformation: " + str(time.time() - tempStart2))
            tempStart2 = time.time()
            #if(expertNum == -1):
            #    Z = weightClasses(Y, Z)
            #print("time for weightClasses: " + str(time.time() - tempStart2))
            with ctx:
                tempStart2 = time.time()
                #logits, loss = callModel(X, Y, Z, model)
                logits = 0
                loss = 0
                printTimeIfAppropriate("time before callModel\n", time.time())
                #if(shouldUseUntouched):
                #    logits, loss = model(X, Y)
                #else:
                #    logits, loss = model(X, Y, Z)
                logits, loss = model(X, Y)

                printTimeIfAppropriate("time after callModel\n", time.time())

                printTimeIfAppropriate("time for callModel: " + str(time.time() - tempStart2))
                tempStart2 = time.time()
                percentCorrect = getPercentCorrect(Y, logits)
                #routerPercentCorrect = 0
                #if(expertNum == -1):
                #    routerPercentCorrect = getPercentCorrectRouterChoices(logits, Y, Z, numTargetWords)
                #    validationLossIterationCount += 1
                #print("time for getPercentCorrect: " + str(time.time() - tempStart2))
            #percentsCorrect[k] = routerPercentCorrect
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
        percentsCorrect = torch.zeros(eval_iters)
        startTime = 0
        hasPrinted = False
        startIter = 2
        tempStart = 0

        for k in range(eval_iters):
            if(k==startIter):
                startTime = time.time()
            tempStart2 = time.time()
            X, Y = get_batch(split)
            printTimeIfAppropriate("time for one getting one batch: " + str(time.time() - tempStart2))
            if(float(k-startIter)/eval_iters > 0.05 and not hasPrinted):
                print(f"percent done with {split} is: {float(k)/eval_iters}")
                diff = time.time() - startTime
                print("time gone by in seconds: ", diff)
                print("so estimated total time is ", (diff * eval_iters) / (float(k-startIter)) )
                print("split is: ", split)

                hasPrinted = True
            numTargetWords = Y.size()[0] * Y.size()[1] #batch size * block size
            Z = getZ(Y)
            
            if(not (tempStart == 0)):
                printTimeIfAppropriate("time for one iteration: " + str(time.time() - tempStart))
            tempStart = time.time()    
            tempStart2 = time.time()
            Y = transformY(Y)
            printTimeIfAppropriate("time for one transformation: " + str(time.time() - tempStart2))
            tempStart2 = time.time()
            if(expertNum == -1):
                Z = weightClasses(Y, Z)
            printTimeIfAppropriate("time for weightClasses: " + str(time.time() - tempStart2))
            #printYandZ(Y, Z)
            with ctx:
                tempStart2 = time.time()
                logits, loss = callModel(X, Y, Z, model)
                printTimeIfAppropriate("time for callModel: " + str(time.time() - tempStart2))
                tempStart2 = time.time()
                percentCorrect = 0
                if(not expertNum == -1):
                    totalNumWords = Z.size()[0] * Z.size()[1]
                    percentCorrect = getPercentCorrectExpertWords(logits, Y, Z, totalNumWords)
                    #percentCorrect = getPercentCorrect(Y, logits)
                routerPercentCorrect = 0
                if(expertNum == -1):
                    routerPercentCorrect = getPercentCorrectRouterChoices(logits, Y, Z, numTargetWords)
                    validationLossIterationCount += 1
                    percentCorrect = routerPercentCorrect
                
                printTimeIfAppropriate("time for getPercentCorrect: " + str(time.time() - tempStart2))
            losses[k] = loss.item()

            percentsCorrect[k] = percentCorrect
        out[split] = losses.mean()
        print("percent correct for " + split + ": " + str(percentsCorrect.mean()))
    model.train()
    inTraining = True
    return out

# helps estimate an arbitrarily accurate loss over either split using many batches
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
            expertNum = -1
            X, Y = get_batch(split)
            #Z = getZ(Y)

            # now I want to see how to router does just the words for expert 1
            #expertNum = 1
            Z = torch.ones_like(Y)
            expertNum = -1



            targetsRouter = Y
            targetsRouter = transformY(targetsRouter)
            #print("embedRouter:" + str(embedRouter[0][0]))
            #print("embed0:" + str(emed0[0][0]))
            #exit()
            with ctx:
                numTargetWords = Y.size()[0] * Y.size()[1] #batch size * block size
                #logitsRouter, lossRouter = callModel(X, Y, Z, model)
                logitsRouter, lossRouter = callModel(X, targetsRouter, Z, model)
                
                
                #print("logitsRouter: " + str(logitsRouter))
                routerChoices = torch.argmax(logitsRouter, dim=2)
                routerChoicesEmbeddings = getModelTransformer(embeddingModel).wte(routerChoices)
                dis0 = torch.norm(routerChoicesEmbeddings - centroid0, dim = -1)
                dis1 = torch.norm(routerChoicesEmbeddings - centroid1, dim = -1)
                routerChose0 = torch.where(dis0 < dis1, 1, 0)
                routerChose1 = torch.where(dis1 < dis0, 1, 0)
                #routerOut = transformY(routerChoices)
                #print("routerOut: " + str(routerOut))
                #exit()
                #routerOut = torch.argmax(logitsRouter, dim=2)

                #one where correct, zero where incorrect, -1 where not trained
                #routerCorrectWords = torch.where(routerOut == Y, 1, 0)
                #numRouterCorrectWords = routerCorrectWords.sum()
                #routerPercentCorrect = float(numRouterCorrectWords)/numTargetWords


                #routerPercentCorrect = getPercentCorrectExpertWords(logitsRouter, Y, Z, numTargetWords)
                routerPercentCorrect = getPercentCorrectRouterChoices(logitsRouter, targetsRouter, Z, numTargetWords)
                
                #DO NOT UNCOMMENT FOLLOWING LINE EXCEPT FOR TESTS! It adjust the router output here to falsely be always right
                #routerOut = Y.clone()

                targets0 = Y
                targets1 = Y
                expertNum = 0
                targets0 = transformY(targets0)
                expertNum = 1
                targets1 = transformY(targets1)

                #targets0 = Y * 2
                #targets1 = (Y - mid) * 2

                X.to(device)
                Y.to(device)
                
                expertNum = 0
                Z0 = getZ(Y)
                #THE BELOW LINE MUST BE COMMENTED! IT BREAKS THE CODE!
                #Z0 = getZ(targets0)
                Z0.to(device)
                #model0.to(device)
                #model0.eval()
                logits0, loss0 = callModel(X, targets0, Z0, model0)
                expert0PercentCorrect = getPercentCorrectExpertWords(logits0, targets0, Z0, numTargetWords)


                expertNum = 1
                Z = getZ(Y)
                Z.to(device)
                logits1, loss1 = callModel(X, targets1, Z, model1)
                expert1PercentCorrect = getPercentCorrectExpertWords(logits1, targets1, Z, numTargetWords)

                biggerZ = torch.where(Z>Z0, Z, Z0)
                isOne = torch.where(biggerZ<1, 1, 0).any()
                if(isOne):
                    print("there is an untrained word")
                    print("biggerZ: " + str(biggerZ))
                    exit()
                #else:
                    #print("all words are trained")

                
                # Ensure routerOut has the same shape as logits0 and logits1
                #routerOut_expanded = routerOut.unsqueeze(-1).expand_as(logits0)
                routerChose0_expanded = routerChose0.unsqueeze(-1).repeat(1,1,logits0.size(-1))

                #make it so routerout is artificially always right
                if(shouldIdealizeMoeRouterForTests):
                    routerOut_expanded = Y.unsqueeze(-1).expand_as(logits0)

                # Combine the logits tensors based on the condition
                logitsCombined = torch.where(routerChose0_expanded == 1, logits0, logits1)
                moePercentCorrect = getPercentCorrect(Y, logitsCombined)
                
                #Z0Expanded = Z0.unsqueeze(-1).expand_as(logits0)
                logits_flat = logitsCombined.view(-1, logitsCombined.size(-1))

                #transform targets into the expert domain so we can evaluate the output:

                targets = torch.where(Y <= mid, targets0, targets1)
                targets_flat = targets.view(-1)
                loss = model.get_cross_entropy(logits_flat, targets_flat, ignore_index=-1)
                if(i%10==0):
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
            losses[i] = loss
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
    
    minEquivalent = 3
    maxEquivalent = 11

    numMins = torch.where(Y == minEquivalent, 1, 0).sum()
    numMaxs = torch.where(Y == maxEquivalent, 1, 0).sum()
    totalNum = numMins + numMaxs
    portionMins = float(numMins)/totalNum
    portionMaxs = float(numMaxs)/totalNum
    greaterPortion = portionMins
    lesserPortion = portionMaxs
    moreFrequentVal = minEquivalent
    if(portionMaxs > portionMins):
        greaterPortion = portionMaxs
        lesserPortion = portionMins 
        moreFrequentVal = maxEquivalent
    portionToKeep = lesserPortion/greaterPortion
    portionToKeep = portionToKeep.to(device)
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
    
    #the below variable is only for tests
    '''yRelevant = torch.where(Z==1, Y, 0)
    numMins = torch.where(yRelevant == minEquivalent, 1, 0).sum()
    numMaxs = torch.where(yRelevant == maxEquivalent, 1, 0).sum()
    totalNum = numMins + numMaxs
    portionMins = float(numMins)/totalNum
    portionMaxs = float(numMaxs)/totalNum'''
    return Z

def callModel(X, Y, Z, model):
    global inTraining
    if(shouldUseUntouched):
        logits, loss = model(X, Y)
        return logits, loss
    else:
        logits, loss = model(X, Y, Z)
        allOnes = torch.ones_like(Z)
        total = allOnes.sum()
        numNumbers = torch.where(Z == 1, 1, 0).sum()
        #if(inTraining):
        #    print("total: " + str(total))
        #    print("numNumbers: " + str(numNumbers))
        #loss = loss * total/numNumbers
        return logits, loss

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
#init_from = 'resume'
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
    model.to(device)
    init_from = tempInitFrom
    print("model.wpe.weight.size() 3: " + str(model.transformer.wpe.weight.size()))
    return model

def initAnyModel(fileName):
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, fileName)
    #print(ckpt_path)
    #exit()
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    model_args = {}
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        #print("model_args[" + k + "]: " + type(model_args[k])+ " " + str(model_args[k]))
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
        #unoptimized_model = model
        anyModel = torch.compile(anyModel) # requires PyTorch 2.0
    return anyModel

def getModelTransformer(model):
    transformer = ""
    if type(model) == torch.nn.parallel.DistributedDataParallel:
        transformer = model.module.transformer
    else:
        transformer = model.transformer
    return transformer

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
    global embeddingModel #READ THE ABOVE COMMENT!
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            #print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
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
        
        #fileName = "train" + binExtension
        #print("data path: ", os.path.join(data_dir, fileName))
        #exit()

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
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
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
if((shouldUseWordShakeSpeare or shouldUseOpenWebText)):
    if(shouldJustSaveGPTFile):
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

        #reloadEmbeddingModelForTests(embeddingModel)
        exit()
    else:
        embeddingModel = initAnyModel('gpt2File.pt')
    #init_from = 'gpt2'

if(shouldJustHelpPickClusterCenters):
    calculateClusterMeans(-1, embeddingModel)
    exit()

#init_from = "resume"
#init_from = "scratch"
#if(expertNum == -1):
#    n_layer *= 2
initModel()

if(not shouldUseWordShakeSpeare and not shouldUseOpenWebText):
    embeddingModel = initAnyModel('embedding_only_otherwise_useless.pt')
    print("initialized embedding model for char")
    #print("embeddingModel: " + str(embeddingModel))

embeddingModel.eval()
if((not shouldJustEstimateLoss) and (not shouldEstimateMoeLoss) and expertNum == -1):
    #no need to create reference to transformer here because have not yet created a DDP container
    #getModelTransformer(embeddingModel).wte.requires_grad = False
    if(init_from == 'scratch'):
        getModelTransformer(model).wte = copy.deepcopy(getModelTransformer(embeddingModel).wte)
        #getModelTransformer(model).wte = getModelTransformer(embeddingModel).wte
    getModelTransformer(model).wte.requires_grad = True
    print("passed initModel")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    if(not shouldJustEstimateLoss):
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #if(expertNum==-1):
        #    getModelTransformer(embeddingModel).wte.requires_grad = False

        print("optimizer param list: " + str(optimizer.param_groups))
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model

    model = torch.compile(model) # requires PyTorch 2.0
    unoptimized_embeddingModel = embeddingModel
    #print("embeddingModel pre compile: " + str(embeddingModel))
    #embeddingModel = torch.compile(embeddingModel) # requires PyTorch 2.0
    #print("embeddingModel post compile: " + str(embeddingModel))

calculateClusterMeans(expertNum, embeddingModel)
'''            
i = 0
while i < len(Y):
    #print(Y[i])
    if Y[i] > mid:
        Z[i] = 1
    i += 1'''
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


lastPercentCorrect = 0
percentThreshold = 0.95


if(shouldJustEstimateLoss):
    losses = ""
    if(shouldEstimateMoeLoss):
        print("estimating expert loss")
        losses = estimate_expert_loss()
    else:
        losses = estimate_loss()
    print(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    exit()

runStartTime = time.time()
tempStart = time.time()
while expertNum < 2:
    if(not tempStart == 0):
        print("time for one iteration: " + str(time.time() - tempStart))
    tempStart = time.time()
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    
    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0) and master_process and testConfigs.shouldEvaluateAtAll:
        isCalculatingValidationLoss = True
        if(lastPercentCorrect > percentThreshold):
            breakpoint = 10
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

                #torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
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
            #printYandZ(Y, Z)
            if(expertNum == -1):
                Z = weightClasses(Y, Z)
                if(shouldIgnoreStronglyCorrectRouterChoices):
                    print("shouldIgnoreStronglyCorrectRouterChoices is true")
                    with torch.no_grad():
                        logits, loss = callModel(X, Y, Z, model)
                        #Z = getCorrectRouterChoices(logits, Y)
                        learnFromCorrectOptions = [0, 1]
                        rand = random.choice(learnFromCorrectOptions)
                        #rand = 0
                        if(rand==0):
                            #Z = getCorrectRouterChoices(logits, Y)
                            if(shouldIgnoreStronglyCorrectRouterChoices):
                                Ztemp = getStronglyCorrectRouterChoices(logits, Y, embeddingModel)
                                Ztemp = torch.where(Z == 1, 0, 1)
                                Z = torch.where(Ztemp==0, 0, Z)
            logits, loss = callModel(X, Y, Z, model)
            
            numTargetWords = Y.size()[0] * Y.size()[1] #batch size * block size
            percentCorrect = 0
            if(expertNum == -1):
                percentCorrect = getPercentCorrectRouterChoices(logits, Y, Z, numTargetWords)
                lastPercentCorrect = percentCorrect
            else:
                percentCorrect = getPercentCorrectExpertWords(logits, Y, Z, numTargetWords)
                lastPercentCorrect = percentCorrect
            if(expertNum == -1):
                loss = loss / gradient_accumulation_steps
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        runEndtime = time.time()
        runTime = runEndtime - runStartTime
        #if(not shouldOnlyShowEvalOut):
            #print("run time: " + str(runTime))
        prepStartTime = time.time()
        Z = getZ(Y)
        Y = transformY(Y)
        prepEndTime = time.time()
        prepTime = prepEndTime - prepStartTime
        #if(not shouldOnlyShowEvalOut):
            #print("prep time: " + str(prepTime))
        runStartTime = time.time()
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    #getModelTransformer(model).wte.requires_grad = False
    #getModelTransformer(model).wte.zero_grad()

    #getModelTransformer(embeddingModel).wte.requires_grad = False
    
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

            #if iter_num % (log_interval * 5) == 0:
            #    if(not ddp):
            #        print("model.wte 0 0: " + str(getModelTransformer(model).wte[0][0]))
            #        print("embeddingModel.wte 0 0: " + str(getModelTransformer(embeddingModel).wte[0][0]))

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        exit()
        break
        #if(expertNum == -1):
        #    break
        #expertNum += 1
        #iter_num = 0
        #print("now onto expert: " + str(expertNum))
if ddp:
    destroy_process_group()

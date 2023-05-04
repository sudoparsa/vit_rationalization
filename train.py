import torch
from tqdm import tqdm
from utils import *


def train_mnist(dataset, model, opt, scheduler, step, args, show_rationale=False, enable_continuity_loss=False):
    """
    Training invariant model on MNIST.
    """

    ### obtain the optimizer
    opt = opt

    ### average loss
    avg_inv_acc = 0
    avg_inv_loss = 0
    avg_var_loss = 0
    sparsity = 0
    count = 0

    selected_bias = 0
    num_sample = 0

    model.train()

    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataset)):
        count +=1

        inputs = inputs.to(device)
        labels = labels.to(device)
        envs = envs.to(device)


        opt.zero_grad()
        rationale, inv_logits = model(inputs)
        
        model_loss, var_loss = inv_rat_loss(inv_logits, envs, labels)

        
        p = 1
        var_lambda = 0
        if step > 5:
          var_lambda = (step-5)*100
          p = 1/var_lambda

        total_loss = p*(var_lambda * var_loss +  model_loss)

        total_loss.backward()

        # print (total_loss)
        # for n,p in model.named_parameters():
        #   print (n)
        #   print (p.grad)
        
        # if batch==0:
        #   break
          
        opt.step()


      
        sparsity += cal_sparsity_loss(rationale)
        avg_inv_loss += model_loss
        avg_var_loss += var_loss

        avg_inv_acc += torch.sum(torch.argmax(inv_logits, dim=1)==torch.argmax(labels, dim=1))
    
    # results
    avg_inv_acc = avg_inv_acc/(count*args.batch_size)
    avg_inv_loss = avg_inv_loss/(count*args.batch_size)
    avg_var_loss = avg_var_loss/(count*args.batch_size)
    sparsity = sparsity/count
  
    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f} {:s}{:.8f} {:s}{:4f}.".format(
        "----> [Train] Total iteration #", step, "inv acc: ",
        avg_inv_acc, "inv loss: ", avg_inv_loss, "var loss: ", avg_var_loss, "sparsity: ", sparsity),
          flush=True)
    
    scheduler.step()

    return step+1

def train_vit(dataset, model, opt, scheduler, step, args, var_lambda=0):
    """
    Training invariant model on MNIST.
    """

    ### obtain the optimizer
    opt = opt

    ### average loss
    avg_inv_acc = 0
    avg_inv_loss = 0
    avg_var_loss = 0
    count = 0

    num_sample = 0

    model.train()

    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataset)):
        count +=1

        inputs = inputs.to(device)
        labels = labels.to(device)
        envs = envs.to(device)

        opt.zero_grad()
        tokens = model.vit_encoder.forward_features(inputs)
        inv_logits = model.classifier_head(tokens[:, 0, :])
        
        model_loss, var_loss = inv_rat_loss(inv_logits, envs, labels)

        p = 1
        if step > 2:
          var_lambda = (step-2)*50
          p = 1/var_lambda

        total_loss = p*(args.var_lambda * var_loss +  model_loss)

        # print (total_loss)

        total_loss.backward()

        # for n,p in model.named_parameters():
        #   print (n)
        #   print (p.grad)
        
        # if batch==0:
        #   break
          
        opt.step()

        avg_inv_loss += model_loss
        avg_var_loss += var_loss

        avg_inv_acc += torch.sum(torch.argmax(inv_logits, dim=1)==torch.argmax(labels, dim=1))
      
    # results
    avg_inv_acc = avg_inv_acc/(count*args.batch_size)
    avg_inv_loss = avg_inv_loss/(count*args.batch_size)
    avg_var_loss = avg_var_loss/(count*args.batch_size)

    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f} {:s}{:.8f}.".format(
        "----> [Train] Total iteration #", step, "inv acc: ",
        avg_inv_acc, "inv loss: ", avg_inv_loss, "var loss: ", avg_var_loss),
          flush=True)
    
    # scheduler.step()

    return step+1

def test_mnist(args, dataset, model):
    """
    Conventional testing of a classifier.
    """
    avg_inv_acc = 0 
    count = 0

    model.eval()
    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataset)):
        count+=1

        inputs = inputs.to(device)
        labels = labels.to(device)
        envs = envs.to(device)

        rationale, inv_logits = model(inputs)

        avg_inv_acc += torch.sum(torch.argmax(inv_logits, dim=1)==torch.argmax(labels, dim=1))

    avg_inv_acc = avg_inv_acc/(count*args.batch_size)

    print("{:s}{:.4f}.".format(
        "----> [Eval] inv acc: ", avg_inv_acc))

    return avg_inv_acc


def test_vit(args, dataset, model):
    """
    Conventional testing of a classifier.
    """
    avg_inv_acc = 0 
    count = 0

    model.eval()
    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataset)):
        count+=1

        inputs = inputs.to(device)
        labels = labels.to(device)
        envs = envs.to(device)

        tokens = model.vit_encoder.forward_features(inputs)
        inv_logits = model.classifier_head(tokens[:, 0, :])

        avg_inv_acc += torch.sum(torch.argmax(inv_logits, dim=1)==torch.argmax(labels, dim=1))

    avg_inv_acc = avg_inv_acc/(count*args.batch_size)

    print("{:s}{:.4f}.".format(
        "----> [Eval] inv acc: ", avg_inv_acc))

    return avg_inv_acc

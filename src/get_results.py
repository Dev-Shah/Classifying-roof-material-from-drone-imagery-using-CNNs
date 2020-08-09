import itertools
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import log_loss
from load_data import *

def foo(model, dataloader, device):
    model.eval()
    output = {}
    for i,(inputs, labels, paths) in enumerate(dataloader):
        inputs = inputs.to(device)
        path=str(paths).split('/')[-1].strip(".png',)")
        out = model(inputs)
        output[path]=torch.exp(out)
        # labels = labels.to(device)
        if i%100==0:
            print(i)
            print(path)

    for k,v in output.items():
        output[k] = v[0].data.cpu().numpy()

    result = pd.DataFrame.from_dict(output).T

    result.columns=['concrete_cement','healthy_metal','incomplete','irregular_metal','other']

    return result

def foo_wa(model, dataloader, device):
    model.eval()
    final_output = np.array([])
    for i,(inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        out = model(inputs)
        output = torch.exp(out)
        try:
            final_output = np.concatenate((final_output,output.data.cpu().numpy()))
        except:
            final_output = output.data.cpu().numpy()

    return final_output


def get_results(dir_, models, device, type_, weighted_average = False):
    # EXAMPLE USAGE:
    # instantiate the dataset and dataloader
    dataloader = load_test_data(dir_)

    model.cuda() if device == 'cuda'

    if type_ == 'basic':
        result = foo(models,dataloader,device)
        result.to_csv('../results/' + models.__class__.__name__ + '.csv')

    else:
        if weighted_average:
            trainloader, valloader, valloader_2 = load_train_val_data(dir_,weighted_average = True)

            final_labels = np.array([])
            for i,(inputs, labels) in enumerate(valloader_2):
                final_labels = np.append(final_labels,labels.data.cpu().numpy())


            val_results = [(lambda x: foo_wa(model,valloader_2,device))(model) for model in models]
            weights = [1,2,3,4,5,6,7,8,9]
            weights_permutations = [p for p in itertools.product(weights, repeat=len(models))]
            weights_permutations = [x for x in weights_permutations if sum(x)==10]
            temp_indexes = np.linspace(0,len(weights_permutations),len(models)).astype(int)
            weights_permutations = [weights_permutations[x] for x in temp_indexes]
            weights_permutations.append(tuple([1/len(models)]*len(models)))


            best_loss = float('Inf')
            for weights_permutation in weights_permutations:
                for i,weight in enumerate(weights_permutation):
                    try:
                        temp_result+=results[i]*weight
                    except:
                        temp_result = results[i]*weight
                temp_loss = 0
                for i,label in enumerate(final_labels):
                    temp=np.array([0,0,0,0,0])
                    temp[label]=1
                    temp_loss += log_loss(temp,temp_result[i])
                if temp_loss<best_loss:
                    best_loss = temp_loss
                    best_weights = weights_permutation


            results = [(lambda x: foo(model,dataloader,device))(model) for model in models]

            for i,result in enumerate(results):
                try:
                    final_result += result*best_weights[i]
                except:
                    final_result = result*best_weights[i]


            final_result = pd.DataFrame.from_dict(output).T

            final_result.columns=['concrete_cement','healthy_metal','incomplete','irregular_metal','other']

            final_result.to_csv('../results/' + models[0].__class__.__name__ + '_snapshot_wa.csv')

            return final_result



        else:
            results = [(lambda x: foo(model,dataloader,device))(model) for model in models]

            final_result = results[0]
            for result in results[1:]:
                final_result+=result

            final_result/=len(results)

            final_result.to_csv('../results/' + models[0].__class__.__name__ + '_snapshot.csv')

            return final_result

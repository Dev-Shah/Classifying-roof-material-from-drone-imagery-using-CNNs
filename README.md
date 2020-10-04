# Classifying roof material from drone imagery using CNNs with snapshot ensembling (PyTorch)
• Predicted probability of type of roof material (concrete cement, healthy metal, irregular metal, incomplete, other) from
drone imagery to identify disaster prone areas using transfer learning with pre-trained ResNet 50 architecture. \
• Used snapshot ensembling (Train 1, get M) with cyclical cosine and linear learning rates to get six different CNN models
at the cost of training a single CNN. \
• Performed a comparative study on different methods to ensemble these models including a simple average, weighted
average and a gaussian similarity kernel. \
• Achieved a log loss of 0.7594 achieving a rank of 72 among 1425 participants (Top 5%) as opposed to the benchmark of 2.01 using
the pre-trained ResNet 50 architecture (challenge hosted on [drivendata.org](https://www.drivendata.org/competitions/58/disaster-response-roof-type/)). 

Instructions: \
run the shell script run.sh - ./run.sh snapshot (for snapshot ensembling) \
                              ./run.sh basic (for basic model) 
                              
user can edit the parameters for basic model in the basic.json file
and can edit the parameters for snapshot ensembling in the snapshot.json

running the script will train the model according to the parameters in the json file,
save the models in the models folder and save the results in the results folder

The dataset in this repo is just a random sample of the original dataset (as it was of approx. 4 GB)

For more information see the reports pdf or ppt

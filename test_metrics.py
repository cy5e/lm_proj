import numpy as np
import sys
import copy
probs_file = sys.argv[1]
gts_file = sys.argv[2]

probs = np.load(probs_file) # N*6151 matrix
gts = np.load(gts_file)  # N*6151 one-hot matrix
reals=  np.argmax(gts,axis=1)
classes = set()
for label in reals:
    classes.add(label)

cl_to_label = {} # dict having mapping from index number 0-6150 to actual class value
with open('retrained50_labels_ADAM0.0007_10000_512.txt','r') as f:
    for i,line in enumerate(f.readlines()):
        cl_to_label[i] = int(line.strip())
print(len(cl_to_label.keys()))
        
print("Num of classes",len(classes))
top1accu = sum(np.argmax(probs,1)==np.argmax(gts,1))/probs.shape[0]
print('Top-1 Accuracy is ',top1accu)

def cal_accu(inds, reals): # predict topk-accuracy given top-k indexes and ground truths
    N, k = inds.shape
    correct = 0
    for i,label in enumerate(reals):
        if label in inds[i]:
            correct += 1
    return float(correct)/N

for i in range(1,11):
    inds = np.argpartition(probs,-1*i,axis=1)[:,-1*i:]
    print('Top ',i,'Accuracy is',cal_accu(inds,reals))


preds = np.argmax(probs,1)
#inds5 = np.argpartition(probs,-5,axis=1)[:,-5:]
#inds10 = np.argpartition(probs,-10,axis=1)[:,-10:]


num_preds_per_class = {} # class : #predictions made in that class
num_corr_preds_pc = {} # class: #correct predictions
num_corr_10preds = {}
for cl in classes:
    num_preds_per_class[cl] = 0
    num_corr_preds_pc[cl] = 0
    num_corr_10preds[cl] = 0
    
for label in preds:
    num_preds_per_class[label] += 1
for i, label in enumerate(preds):
    if label == reals[i]:
        num_corr_preds_pc[label] += 1

prec_class = {}
recall_class = {}
f1_class  = {}
for cl in classes:
    if num_preds_per_class[cl] != 0:
        prec_class[cl] = num_corr_preds_pc[cl]/float(num_preds_per_class[cl])
    else:
        prec_class[cl] = 0.0
    recall_class[cl] = num_corr_preds_pc[cl]/2.0
    if recall_class[cl] + prec_class[cl] > 0:
        f1_class[cl] = 2*prec_class[cl]*recall_class[cl]/(recall_class[cl]+prec_class[cl])
    else:
        f1_class[cl] = 0.0
    
cum_prec, cum_recall, cum_f1 = 0.0,0.0,0.0
for cl in classes:
    cum_prec += prec_class[cl]
    cum_recall += recall_class[cl]
    cum_f1 += f1_class[cl]

C = len(classes)
print("Avg precision over all classes is",cum_prec/C)
print("Avg recall over all classes is",cum_recall/C)
print("Avg F1 score over all classes is",cum_f1/C)


mis_classes = [] # classes where both test examples were misclassified with Top-10 accuracy metric
inds10 = np.argpartition(probs,-10,axis=1)[:,-10:]
for i, label in enumerate(reals):
    if label in inds10[i]:
        num_corr_10preds[label] += 1
        
for cl in classes:
    if num_corr_10preds[cl] == 0:
        mis_classes.append(cl_to_label[cl])
        
print('List of classes where both test examples were misclassified with Top-10 accuracy metric')
print(mis_classes)

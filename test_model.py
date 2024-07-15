import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import math

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + 1e-06)
    npv = float(tn) / (tn + fn + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
    f1 = float(tp * 2) / (tp * 2 + fp + fn + 1e-06)
    return acc, precision, npv, sensitivity, specificity, mcc, f1

from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import math

# path = './HD3C/Subsets_results/ML-DL/'
#########################################################################################
data_t = pd.read_csv('data/X_test_rfe_EN.csv', header=None)
dataTest = np.array(data_t)
[m1, n1] = np.shape(dataTest)
# label1 = np.ones((int(m1 / 2), 1))
# label2 = np.zeros((int(m1 / 2), 1))
# label = np.append(label1, label2)
label = pd.read_csv('data/y_test_rfe_EN.csv',header=None).values
# shu = scale(dataTest)
Xt = dataTest
yt = label
###########################################################################################
[sample_num, input_dim] = np.shape(Xt)
Xt = np.reshape(Xt, (-1, 1, input_dim))

sepscores = []
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5

# load json and create model
json_file = open('model/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/CNN_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
#score = loaded_model.evaluate(Xt, yt, verbose=0)
#print(score)
y_score = loaded_model.predict(Xt)
y_class = categorical_probas_to_classes(y_score)

y_test = to_categorical(yt)
ytest = np.vstack((ytest, y_test))
y_test_tmp = yt
yscore = np.vstack((yscore, y_score))

acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(len(y_class), y_class,
                                                                                    y_test_tmp)
fpr, tpr, _ = roc_curve(y_test[:, 1], y_score[:, 1])
roc_auc = auc(fpr, tpr)
sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])
print('CNN:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))

scores = np.array(sepscores)
result1 = np.mean(scores, axis=0)
H1 = result1.tolist()
sepscores.append(H1)
result = sepscores

row = yscore.shape[0]
yscore = yscore[np.array(range(1, row)), :]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('shap_test_results/CNN_yscore_test.csv')

ytest = ytest[np.array(range(1, row)), :]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('shap_test_results/CNN_ytest_test.csv')

data_csv = pd.DataFrame(data=result)
data_csv.to_csv('shap_test_results/CNN_test_results.csv')
import numpy as np
import random
from feature import Feature
from tdnn import TDNN

def main():

    feature = Feature()
    x_temp = list()
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000001.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000002.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000003.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000004.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000005.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000006.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000007.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000008.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000009.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000010.png')]])

    X_train = np.array(x_temp)
    print (X_train.shape)
    y_train = np.array([random.uniform(0.8,1)]*10)
    X_test = np.array([[[feature.extractFeatureVector('../images/S502_001_00000001.png')]]])
    print(X_test.shape)
    y_test = np.array([1])

    tdnn = TDNN(verbose=True)
    tdnn.train(X_train, y_train, X_test, y_test)

main()
from tdnn import TDNN
from data import get_delayed_emotion_training_data

def main():
    X_train, y_train, X_test, y_test = get_delayed_emotion_training_data()
    print ('X_train: ' + str(X_train.shape))
    print ('y_train: ' + str(y_train.shape))
    print('X_test: ' + str(X_test.shape))
    print ('y_test: ' + str(y_test.shape))

    tdnn = TDNN(verbose=True)
    tdnn.train(X_train, y_train, X_test, y_test)


main()
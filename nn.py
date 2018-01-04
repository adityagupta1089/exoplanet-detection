import numpy as np
from keras.models import Sequential
from keras.layers import MaxPool1D, Conv1D, Dense, Flatten, Dropout
from random import shuffle

#%% validation split
def val_split(X, Y, train_split):
    #%% equal separation
    pos = sum(Y == 1)
    neg = sum(Y == 0)
    # we have same ratio of pos-neg in train and validation
    train_pos = pos * train_split
    train_neg = neg * train_split
    
    # define arrays for appending
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    
    # current count of positive and negative samples in training data
    curr_pos = 0
    curr_neg = 0
    for x, y in zip(X, Y):
        if (y == 1 and curr_pos < train_pos) or (y == 0 and curr_neg < train_neg):
            # only take train_pos positive and train_neg negative examples
            X_train.append(x)
            Y_train.append(y)
            if y == 1:
                curr_pos += 1
            else:
                curr_neg += 1
        else:
            X_val.append(x)
            Y_val.append(y)
            
    # convert to numpy arrays and return
    return (np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val))

#%% get data
def get_data(filename, test=False):
    #%% read data
    with open(filename, 'r') as f:
        content = f.readlines()

    #%% split at comma, skip the header row
    if test:
        content = np.array(list(list(map(float, line.split(',')[1:]))
            for line in content[1:]))
    else:
        content = np.array(list(list(map(float, line.split(',')))
            for line in content[1:]))
    #%% shuffling the data
    shuffle(content)

    #%% process the input
    X = content[:, 1:] if not test else content
    # standardize input
    mean = np.mean(X, axis=1)[:, np.newaxis] # adding dimension for row
    X = X - mean
    std = np.std(X, axis=1)[:, np.newaxis] # adding dimension for row
    X = X / std
    # number of time series
    items = X.shape[0]
    # we have (items, length), we want (items, length, 1) where last dimension
    # is for depth which contains the original series and various measures
    X = X.reshape(items, -1, 1)
    
    #%% process the output, 0(not exoplanet) or (exoplanet) from 1/2
    if not test:
        Y = content[:, 0] - 1    
    
    return (X, Y) if not test else X

#%% balanced generator
def balanced_generator(X, Y, batch_size):
    #%% Generate balanced batch
    # boolean vectors representing whether output is 1 or 0
    pos = Y == 1
    neg = Y == 0
    # count of elements in each class
    tot_pos = sum(pos)
    tot_neg = sum(neg)
    # indices, i.e. [0, 1, 2..]
    indices = np.arange(X.shape[0])
    # probabilities associated with each element
    # class 1 has p1, class 0 has p0 probability, a2 and a0 elements,
    # then expected number of elements from 
    # class 1 -> a1*p1/(a1*p1+a0*p0) * batch_size
    # we want it to be batch_size / 2 and denominator 1.
    # hence a1 * p1 = 1/2 or p1 = 1/(2*a1) and p0 = 1/(2*a0)
    probabs = [1/(2*tot_pos) if y == 1 else 1/(2*tot_neg) for y in Y]
    
    while 1:
        batch_indices = np.random.choice(indices, batch_size, p=probabs, 
                                         replace=True)    
        yield (X[batch_indices], Y[batch_indices])

#%% show stats    
def show_stats(model, X, Y):
    #%% predicting
    pred = model.predict(X).reshape(-1)
    # find elements of each type
    tp = sum((pred >= 0.5) * (Y == 1))
    fp = sum((pred >= 0.5) * (Y == 0))
    tn = sum((pred < 0.5) * (Y == 0))
    fn = sum((pred < 0.5) * (Y == 1))
    
    print('tp is %d, fp is %d, tn is %d, fn is %d\n' % (tp, fp, tn, fn))
    
    # calculate measusres
    prec = prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    # fmes = 2 / (1 / prec + 1 / rec)
    fmes = 2 * tp / (2 * tp + fp + fn)
    
    print('Accuracy is %2.2f%%, Precision is %2.2f%%, Recall is %2.2f%%, F-measure is %f\n' 
          % (acc * 100, prec * 100, rec * 100, fmes))
    
#%% main
if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = val_split(*get_data('ExoTrain.csv'), 0.7)
    X_test = get_data('Final Test.csv', test=True)
    
    
    #%% create model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu',
                     input_shape=X_train.shape[1:]))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    #%% compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy'])

    #%% fit the model
    batch_size = 64
    train_batches = X_train.shape[0] / batch_size
    val_batches = X_val.shape[0] / batch_size
    model.fit_generator(balanced_generator(X_train, Y_train, batch_size), 
        validation_data=balanced_generator(X_val, Y_val, batch_size),
        steps_per_epoch=train_batches, validation_steps=val_batches, 
        epochs=10, verbose=1)
    
    #%% load model
    model.load_weights('weights.hdf5')
    
    
    #%% show statistics on train & test
    print('Training Data:')
    show_stats(model, X_train, Y_train)
    print('Validation Data:')
    show_stats(model, X_val, Y_val)0
    
    #%% make predictions
    prediction = np.round(model.predict(X_test).reshape(-1) + 1)
    
    #%% save model weights
    #model.save_weights('weights.hdf5')
    
    #%% write results
    with open('results.txt', 'w') as f:
        for y in prediction:
            f.write(str(int(y)) + '\n')
            
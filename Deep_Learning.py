
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import  MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


with open(r"E:\MNIST\train-images.idx3-ubyte",'rb') as f:
    data = np.fromfile(f,dtype = np.uint8)

training = data[16:]
x_train = training.reshape((60000,28*28))
X_train = x_train/255


#Using the 2nd dataset label the digits in training sets ie split the 1st dataset for each digit
with open(r"E:\MNIST\train-labels.idx1-ubyte","rb") as f:
    train_label = np.fromfile(f,dtype = np.uint8)
    
Y_train = train_label[8:60008]


#Forming the testing images from 3rd dataset
with open(r"E:\MNIST\t10k-images.idx3-ubyte","rb") as f:
    test = np.fromfile(f,dtype = np.uint8)
    
test_set = test[16:7840017]
x_test = test_set.reshape((10000,28*28))
X_test = x_test/255

#Building confusion matrix for both the methods using the 4th dataset.
with open(r"E:\MNIST\t10k-labels.idx1-ubyte","rb") as f:
    test_label = np.fromfile(f,dtype = np.uint8)
    
Y_test = test_label[8:10009]


#creating the CNN model
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

nclasses = Y_train.max() - Y_train.min() + 1
Y_train = to_categorical(Y_train, num_classes = nclasses)
input_shape = (28,28,1)
X_input = Input(input_shape)


pca_100 = PCA(n_components=100)
pca_100.fit(X_train)
train_images_reduced = pca_100.transform(X_train)
test_images_reduced = pca_100.transform(X_train)

# get exact variability retained
print("\nVar retained (%):", 
      np.sum(pca_100.explained_variance_ratio_ * 100))

#Layer 1 of CNN
x = Conv2D(64,(3,3),strides=(1,1),name='layer_conv1',padding='same')(X_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool1')(x)


#Layer 2 of CNN
x = Conv2D(32,(3,3),strides=(1,1),name='layer_conv2',padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool2')(x)


#Layer 3 of CNN
x = Conv2D(32,(3,3),strides=(1,1),name='conv3',padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), name='maxPool3')(x)

#Construction
x = Flatten()(x)
x = Dense(64,activation ='relu',name='fc0')(x)
x = Dropout(0.25)(x)
x = Dense(32,activation ='relu',name='fc1')(x)
x = Dropout(0.25)(x)
x = Dense(10,activation ='softmax',name='fc2')(x)

conv_model = Model(inputs=X_input, outputs=x, name='Predict')
epochs = 1

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an image data generator with augmentation options
datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2)

datagen.fit(X_train)
augmented_train_data = datagen.flow(X_train, Y_train, batch_size = 1000)

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate = 0.02)
conv_model.compile(optimizer = optimizer, loss = "categorical_crossentropy",metrics = ['accuracy'])
conv_model.fit(augmented_train_data, epochs=epochs, validation_data = (X_train, Y_train))

predict = []
prediction = conv_model.predict(X_test)
for i in range(len(prediction)):
    max1 = max(prediction[i])
    for j in range(len(prediction[i])):
        if max1==prediction[i][j]:
            predict.append(j)

print("Accuracy Score is:",accuracy_score(predict,Y_test)*100)







train_labels = Y_train
test_labels = Y_test


# define network architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

MLP = Sequential()
MLP.add(InputLayer(input_shape=(100, ))) # input layer
MLP.add(Dense(64, activation='relu')) # hidden layer 1
MLP.add(Dense(32, activation='relu')) # hidden layer 2
MLP.add(Dense(10, activation='softmax')) # output layer

# optimization
MLP.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# train (fit)
history = MLP.fit(train_images_reduced, train_labels, 
                  epochs=20, batch_size=128, verbose=0,
                  validation_split=0.15)

# evaluate performance on test data
test_loss, test_acc = MLP.evaluate(test_images_reduced, test_labels,
                                         batch_size=128,
                                         verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


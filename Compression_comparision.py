'''Aim-1: To Compare the accuracy of image compression techniques-Eigenvalues of cov, PCA, FFT, LDA.'''
'''Aim-2: Using above techniques, find the most similar images.'''

#importing all the libraries
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score,confusion_matrix
import time
import matplotlib.pyplot as plt

#Forming the trainng images from 1st dataset
with open(r"D:\MNIST\train-images.idx3-ubyte",'rb') as f:
    data = np.fromfile(f,dtype = np.uint8)

training = data[16:]
train_output = training.reshape((60000,28*28))
X_train = training.reshape((60000,28*28))

#Using the 2nd dataset label the digits in training sets ie split the 1st dataset for each digit
with open(r"D:\MNIST\train-labels.idx1-ubyte","rb") as f:
    train_label = np.fromfile(f,dtype = np.uint8)
    
Y_train = train_label[8:60008]


#Forming the testing images from 3rd dataset
with open(r"D:\MNIST\t10k-images.idx3-ubyte","rb") as f:
    test = np.fromfile(f,dtype = np.uint8)
    
test_set = test[16:7840017]
X_test = test_set.reshape((10000,28*28))

#Building confusion matrix for both the methods using the 4th dataset.
with open(r"D:\MNIST\t10k-labels.idx1-ubyte","rb") as f:
    test_label = np.fromfile(f,dtype = np.uint8)
    
Y_test = test_label[8:10009]

pred_test = []
pred_test_PCA = []
pred_test_FFT = []

train0 = []
train1 = []
train2 = []
train3 = []
train4 = []
train5 = []
train6 = []
train7 = []
train8 = []
train9 = [] 

for i in range(len(Y_train)):
    if Y_train[i]==0:
        train0.append(X_train[i])
    elif Y_train[i]==1:
        train1.append(X_train[i])
    elif Y_train[i]==2:
        train2.append(X_train[i])
    elif Y_train[i]==3:
        train3.append(X_train[i])
    elif Y_train[i]==4:
        train4.append(X_train[i])
    elif Y_train[i]==5:
        train5.append(X_train[i])
    elif Y_train[i]==6:
        train6.append(X_train[i])
    elif Y_train[i]==7:
        train7.append(X_train[i])
    elif Y_train[i]==8:
        train8.append(X_train[i])
    elif Y_train[i]==9:
        train9.append(X_train[i])

mean_image0 = np.mean(train0,axis = 0)
mean_image1 = np.mean(train1,axis = 0)
mean_image2 = np.mean(train2,axis = 0)
mean_image4 = np.mean(train4,axis = 0)
mean_image3 = np.mean(train3,axis = 0)
mean_image5 = np.mean(train5,axis = 0)
mean_image6 = np.mean(train6,axis = 0)
mean_image7 = np.mean(train7,axis = 0)
mean_image8 = np.mean(train8,axis = 0)
mean_image9 = np.mean(train9,axis = 0)



#BUILT IN Function LDA method

n_LDA = 8
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

start = time.time()
lda = LDA(solver = "svd", n_components = n_LDA)
model = lda.fit(X_train,Y_train)
pred_test_LDA = model.predict(X_test)
end = time.time()

print(f"Time taken by LDA method {end-start}")
prob = []
for i in range(len(X_test)):
    prob.append(model._predict_proba_lr(X_test[i].reshape((1,-1))))

loss = 0
for i in range(len(prob)):
    for j in range(len(prob[i][0])):
        loss += -np.log(prob[i][0][j])
print("Total entropy = loss =", loss/10)
print("Accuracy score using LDA method is",accuracy_score(pred_test_LDA,Y_test)*100,"%")

Conf_matrix_LDA = confusion_matrix(pred_test_LDA,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_LDA))
print("Wrong predictions:", np.sum(Conf_matrix_LDA)-np.sum(np.trace(Conf_matrix_LDA)))
print()


#EIGENVALUES OF COVARIANCE method
n = 50
start = time.time()

data_center0 = train0 - mean_image0
#finding the covariance matrix
cov0 = np.cov(data_center0,rowvar = False)
eigenvalues0,eigenvectors0 = np.linalg.eigh(cov0)

sort_index0 = np.argsort(eigenvalues0)[::-1]
eigenvalues0 = eigenvalues0[sort_index0]
eigenvectors0 = eigenvectors0[:,sort_index0]
projected_data0 = np.dot(data_center0,eigenvectors0[:,:n])


data_center1 = train1 - mean_image1
#finding the covariance matrix
cov1 = np.cov(data_center1,rowvar = False)
eigenvalues1,eigenvectors1 = np.linalg.eigh(cov1)
sort_index1 = np.argsort(eigenvalues1)[::-1]
eigenvalues1 = eigenvalues1[sort_index1]
eigenvectors1 = eigenvectors1[:,sort_index1]
projected_data1 = np.dot(data_center1,eigenvectors1[:,:n])

data_center2 = train2 - mean_image2
#finding the covariance matrix
cov2 = np.cov(data_center2,rowvar = False)
eigenvalues2,eigenvectors2 = np.linalg.eigh(cov2)
sort_index2 = np.argsort(eigenvalues2)[::-1]
eigenvalues2 = eigenvalues2[sort_index2]
eigenvectors2 = eigenvectors2[:,sort_index2]
projected_data2 = np.dot(data_center2,eigenvectors2[:,:n])

data_center3 = train3 - mean_image3
#finding the covariance matrix
cov3 = np.cov(data_center3,rowvar = False)
eigenvalues3,eigenvectors3 = np.linalg.eigh(cov3)
sort_index3 = np.argsort(eigenvalues3)[::-1]
eigenvalues3 = eigenvalues3[sort_index3]
eigenvectors3 = eigenvectors3[:,sort_index3]
projected_data3 = np.dot(data_center3,eigenvectors3[:,:n])

data_center4 = train4 - mean_image4
#finding the covariance matrix
cov4 = np.cov(data_center4,rowvar = False)
eigenvalues4,eigenvectors4 = np.linalg.eigh(cov4)
sort_index4 = np.argsort(eigenvalues4)[::-1]
eigenvalues4 = eigenvalues4[sort_index4]
eigenvectors4 = eigenvectors4[:,sort_index4]
projected_data4 = np.dot(data_center4,eigenvectors4[:,:n])

data_center5 = train5 - mean_image5
#finding the covariance matrix
cov5 = np.cov(data_center5,rowvar = False)
eigenvalues5,eigenvectors5 = np.linalg.eigh(cov5)
sort_index5 = np.argsort(eigenvalues5)[::-1]
eigenvalues5 = eigenvalues5[sort_index5]
eigenvectors5 = eigenvectors5[:,sort_index5]
projected_data5 = np.dot(data_center5,eigenvectors5[:,:n])

data_center6 = train6 - mean_image6
#finding the covariance matrix
cov6 = np.cov(data_center6,rowvar = False)
eigenvalues6,eigenvectors6 = np.linalg.eigh(cov6)
sort_index6 = np.argsort(eigenvalues6)[::-1]
eigenvalues6 = eigenvalues6[sort_index6]
eigenvectors6 = eigenvectors6[:,sort_index6]
projected_data6 = np.dot(data_center6,eigenvectors6[:,:n])

data_center7 = train7 - mean_image7
#finding the covariance matrix
cov7 = np.cov(data_center7,rowvar = False)
eigenvalues7,eigenvectors7 = np.linalg.eigh(cov7)
sort_index7 = np.argsort(eigenvalues7)[::-1]
eigenvalues7 = eigenvalues7[sort_index7]
eigenvectors7 = eigenvectors7[:,sort_index7]
projected_data7 = np.dot(data_center7,eigenvectors7[:,:n])

data_center8 = train8 - mean_image8
#finding the covariance matrix
cov8 = np.cov(data_center8,rowvar = False)
eigenvalues8,eigenvectors8 = np.linalg.eigh(cov8)
sort_index8 = np.argsort(eigenvalues8)[::-1]
eigenvalues8 = eigenvalues8[sort_index8]
eigenvectors8 = eigenvectors8[:,sort_index8]
projected_data8 = np.dot(data_center8,eigenvectors8[:,:n])

data_center9 = train9 - mean_image9
#finding the covariance matrix
cov9 = np.cov(data_center9,rowvar = False)
eigenvalues9,eigenvectors9 = np.linalg.eigh(cov9)
sort_index9 = np.argsort(eigenvalues9)[::-1]
eigenvalues9 = eigenvalues9[sort_index9]
eigenvectors9 = eigenvectors9[:,sort_index9]
projected_data9 = np.dot(data_center9,eigenvectors9[:,:n])

#finding the new projection of data
for i in X_test:
    input_image = i
    new_center0 = input_image - mean_image0
    new_projection0 = np.dot(new_center0,eigenvectors0[:,:n])
    dist0 = int(min(min(cdist(new_projection0.reshape((1,-1)),projected_data0,'euclidean'))))
    
    new_center1 = input_image - mean_image1
    new_projection1 = np.dot(new_center1,eigenvectors1[:,:n])
    dist1 = int(min(min(cdist(new_projection1.reshape((1,-1)),projected_data1,'euclidean'))))
    
    new_center2 = input_image - mean_image2
    new_projection2 = np.dot(new_center2,eigenvectors2[:,:n])
    dist2 = int(min(min(cdist(new_projection2.reshape((1,-1)),projected_data2,'euclidean'))))
    
    new_center3 = input_image - mean_image3
    new_projection3 = np.dot(new_center3,eigenvectors3[:,:n])
    dist3 = int(min(min(cdist(new_projection3.reshape((1,-1)),projected_data3,'euclidean'))))
    
    new_center4 = input_image - mean_image4
    new_projection4 = np.dot(new_center4,eigenvectors4[:,:n])
    dist4 = int(min(min(cdist(new_projection4.reshape((1,-1)),projected_data4,'euclidean'))))
    
    new_center5 = input_image - mean_image5
    new_projection5 = np.dot(new_center5,eigenvectors5[:,:n])
    dist5 = int(min(min(cdist(new_projection5.reshape((1,-1)),projected_data5,'euclidean'))))
    
    new_center6 = input_image - mean_image6
    new_projection6 = np.dot(new_center6,eigenvectors6[:,:n])
    dist6 = int(min(min(cdist(new_projection6.reshape((1,-1)),projected_data6,'euclidean'))))
    
    new_center7 = input_image - mean_image7
    new_projection7 = np.dot(new_center7,eigenvectors7[:,:n])
    dist7 = int(min(min(cdist(new_projection7.reshape((1,-1)),projected_data7,'euclidean'))))
    
    new_center8 = input_image - mean_image8
    new_projection8 = np.dot(new_center8,eigenvectors8[:,:n])
    dist8 = int(min(min(cdist(new_projection8.reshape((1,-1)),projected_data8,'euclidean'))))
    
    new_center9 = input_image - mean_image9
    new_projection9 = np.dot(new_center9,eigenvectors9[:,:n])
    dist9 = int(min(min(cdist(new_projection9.reshape((1,-1)),projected_data9,'euclidean'))))
    dist = [dist0,dist1,dist2,dist3,dist4,dist5,dist6,dist7,dist8,dist9]
 
    index = dist.index(min(dist))
    pred_test.append(index)

end = time.time()

original_data0 = np.array(train0)
compressed_data0 = np.array(projected_data0)
data_loss0 = np.mean(np.square(original_data0[:,:n] - compressed_data0))

print("Total entropy = loss =",data_loss0/10)
print(f"Time taken by eigenvalue of covariance method {end-start}")
print("Accuracy Score of manual method is:", accuracy_score(pred_test,Y_test)*100,"%")

Conf_matrix_manual = np.array(confusion_matrix(pred_test,Y_test))
print("Correct predictions:", np.trace(Conf_matrix_manual))
print("Wrong predictions:", np.sum(Conf_matrix_manual)-np.sum(np.trace(Conf_matrix_manual)))
print()


#BUILT IN FUNCTION PCA method
n_PCA = 8
from sklearn.decomposition import PCA

p = PCA(n_components = n_PCA)
p.fit(train0)

#Applying compression
mean_image0 = np.mean(train0)
projected_data_PCA0 = p.transform(train0)
mean_image1 = np.mean(train1)
projected_data_PCA1 = p.transform(train1)
mean_image2 = np.mean(train2)
projected_data_PCA2 = p.transform(train2)
mean_image3 = np.mean(train3)
projected_data_PCA3 = p.transform(train3)
mean_image4 = np.mean(train4)
projected_data_PCA4 = p.transform(train4)
mean_image5 = np.mean(train5)
projected_data_PCA5 = p.transform(train5)
mean_image6 = np.mean(train6)
projected_data_PCA6 = p.transform(train6)
mean_image7 = np.mean(train7)
projected_data_PCA7 = p.transform(train7)
mean_image8 = np.mean(train8)
projected_data_PCA8 = p.transform(train8)
mean_image9 = np.mean(train9)
projected_data_PCA9 = p.transform(train9)

for i in X_test:
    input_image = i
    new_image_center_PCA0 = input_image - mean_image0
    new_projected_PCA0 = p.transform(new_image_center_PCA0.reshape((1,-1)))
    dist_PCA0 = int(min(min(cdist(new_projected_PCA0,projected_data_PCA0,'euclidean'))))
    
    new_image_center_PCA1 = input_image - mean_image1
    new_projected_PCA1 = p.transform(new_image_center_PCA1.reshape((1,-1)))
    dist_PCA1 = int(min(min(cdist(new_projected_PCA1,projected_data_PCA1,'euclidean'))))
    
    new_image_center_PCA2 = input_image - mean_image2
    new_projected_PCA2 = p.transform(new_image_center_PCA2.reshape((1,-1)))
    dist_PCA2 = int(min(min(cdist(new_projected_PCA2,projected_data_PCA2,'euclidean'))))
    
    new_image_center_PCA3 = input_image - mean_image3
    new_projected_PCA3 = p.transform(new_image_center_PCA3.reshape((1,-1)))
    dist_PCA3 = int(min(min(cdist(new_projected_PCA3,projected_data_PCA3,'euclidean'))))
    
    new_image_center_PCA4 = input_image - mean_image4
    new_projected_PCA4 = p.transform(new_image_center_PCA4.reshape((1,-1)))
    dist_PCA4 = int(min(min(cdist(new_projected_PCA4,projected_data_PCA4,'euclidean'))))
    
    new_image_center_PCA5 = input_image - mean_image5
    new_projected_PCA5 = p.transform(new_image_center_PCA5.reshape((1,-1)))
    dist_PCA5 = int(min(min(cdist(new_projected_PCA5,projected_data_PCA5,'euclidean'))))
    
    new_image_center_PCA6 = input_image - mean_image6
    new_projected_PCA6 = p.transform(new_image_center_PCA6.reshape((1,-1)))
    dist_PCA6 = int(min(min(cdist(new_projected_PCA6,projected_data_PCA6,'euclidean'))))
    
    new_image_center_PCA7 = input_image - mean_image7
    new_projected_PCA7 = p.transform(new_image_center_PCA7.reshape((1,-1)))
    dist_PCA7 = int(min(min(cdist(new_projected_PCA7,projected_data_PCA7,'euclidean'))))

    new_image_center_PCA8 = input_image - mean_image8
    new_projected_PCA8 = p.transform(new_image_center_PCA8.reshape((1,-1)))
    dist_PCA8 = int(min(min(cdist(new_projected_PCA8,projected_data_PCA8,'euclidean'))))
    
    new_image_center_PCA9 = input_image - mean_image9
    new_projected_PCA9 = p.transform(new_image_center_PCA9.reshape((1,-1)))
    dist_PCA9 = int(min(min(cdist(new_projected_PCA9,projected_data_PCA9,'euclidean')))) 
    
    dist_PCA = [dist_PCA0,dist_PCA1,dist_PCA2,dist_PCA3,dist_PCA4,dist_PCA5,dist_PCA6,dist_PCA7,dist_PCA8,dist_PCA9]
    
    #Finding the Least distance image from input_image
    index_PCA = dist_PCA.index(min(dist_PCA))
    pred_test_PCA.append(index_PCA)

original_data0 = np.array(train0)
compressed_data0 = np.array(projected_data_PCA0)
data_loss_PCA = np.mean(np.square(original_data0[:,:n_PCA] - compressed_data0))

print("Total entropy = loss =",data_loss_PCA/100)

print("Accuracy Score using PCA is:",accuracy_score(pred_test_PCA,Y_test)*100,"%")

Conf_matrix_PCA = confusion_matrix(pred_test_PCA,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_PCA))
print("Wrong predictions:", np.sum(Conf_matrix_PCA)-np.sum(np.trace(Conf_matrix_PCA)))
print()


#BUILT IN FUNCTION FFT method

from numpy.fft import fft2
from numpy.fft import ifft2
n_FFT = 1
start = time.time()

new_data_FFT0 = abs(ifft2(fft2(train0)))
new_data_FFT1 = abs(ifft2(fft2(train1)))
new_data_FFT2 = abs(ifft2(fft2(train2)))
new_data_FFT3 = abs(ifft2(fft2(train3)))
new_data_FFT4 = abs(ifft2(fft2(train4)))
new_data_FFT5 = abs(ifft2(fft2(train5)))
new_data_FFT6 = abs(ifft2(fft2(train6)))
new_data_FFT7 = abs(ifft2(fft2(train7)))
new_data_FFT8 = abs(ifft2(fft2(train8)))
new_data_FFT9 = abs(ifft2(fft2(train9)))

for i in X_test:
    input_image = i

    new_image_center_FFT0 = input_image - mean_image0
    new_projected_FFT0 = abs(ifft2(fft2(new_image_center_FFT0.reshape((1,-1)))))
    dist_FFT0 = int(min(min(cdist(new_projected_FFT0,new_data_FFT0,'euclidean'))))

    new_image_center_FFT1 = input_image - mean_image1
    new_projected_FFT1 = abs(ifft2(fft2(new_image_center_FFT1.reshape((1,-1)))))
    dist_FFT1 = int(min(min(cdist(new_projected_FFT1,new_data_FFT1,'euclidean'))))

    new_image_center_FFT2 = input_image - mean_image2
    new_projected_FFT2 = abs(ifft2(fft2(new_image_center_FFT2.reshape((1,-1)))))
    dist_FFT2 = int(min(min(cdist(new_projected_FFT2,new_data_FFT2,'euclidean'))))

    new_image_center_FFT3 = input_image - mean_image3
    new_projected_FFT3 = abs(ifft2(fft2(new_image_center_FFT3.reshape((1,-1)))))
    dist_FFT3 = int(min(min(cdist(new_projected_FFT3,new_data_FFT3,'euclidean'))))

    new_image_center_FFT4 = input_image - mean_image4
    new_projected_FFT4 = abs(ifft2(fft2(new_image_center_FFT4.reshape((1,-1)))))
    dist_FFT4 = int(min(min(cdist(new_projected_FFT4,new_data_FFT4,'euclidean'))))

    new_image_center_FFT5 = input_image - mean_image5
    new_projected_FFT5 = abs(ifft2(fft2(new_image_center_FFT5.reshape((1,-1)))))
    dist_FFT5 = int(min(min(cdist(new_projected_FFT5,new_data_FFT5,'euclidean'))))

    new_image_center_FFT6 = input_image - mean_image6
    new_projected_FFT6 = abs(ifft2(fft2(new_image_center_FFT6.reshape((1,-1)))))
    dist_FFT6 = int(min(min(cdist(new_projected_FFT6,new_data_FFT6,'euclidean'))))

    new_image_center_FFT7 = input_image - mean_image7
    new_projected_FFT7 = abs(ifft2(fft2(new_image_center_FFT7.reshape((1,-1)))))
    dist_FFT7 = int(min(min(cdist(new_projected_FFT7,new_data_FFT7,'euclidean'))))

    new_image_center_FFT8 = input_image - mean_image8
    new_projected_FFT8 = abs(ifft2(fft2(new_image_center_FFT8.reshape((1,-1)))))
    dist_FFT8 = int(min(min(cdist(new_projected_FFT8,new_data_FFT8,'euclidean'))))

    new_image_center_FFT9 = input_image - mean_image9
    new_projected_FFT9 = abs(ifft2(fft2(new_image_center_FFT9.reshape((1,-1)))))
    dist_FFT9 = int(min(min(cdist(new_projected_FFT9,new_data_FFT9,'euclidean'))))

    dist_FFT = [dist_FFT0,dist_FFT1,dist_FFT2,dist_FFT3,dist_FFT4,dist_FFT5,dist_FFT6,dist_FFT7,dist_FFT8,dist_FFT9]    
    
    #Finding the Least distance image from input_image
    index_FFT = dist_FFT.index(min(dist_FFT))
    pred_test_FFT.append(index_FFT)

end = time.time()
original_data0 = np.array(train0)
compressed_data0 = np.array(new_data_FFT0)
data_loss_FFT = np.mean(np.square(original_data0[:,:n_FFT] - compressed_data0))

print("Total entropy = loss =",data_loss_FFT/10)
print(f"Time taken by FFT method {end-start}")

print("Accuracy Score using FFT is:",accuracy_score(pred_test_FFT,Y_test)*100,"%")

Conf_matrix_FFT = confusion_matrix(pred_test_FFT,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_FFT))
print("Wrong predictions:", np.sum(Conf_matrix_FFT)-np.sum(np.trace(Conf_matrix_FFT)))
print()
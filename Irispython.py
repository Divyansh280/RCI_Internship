#Imprting the libraries
import pandas as pd
import numpy as np
from numpy.fft import fft2, ifft2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#Importing the dataset
df = pd.read_csv(r"C:\Users\amit\OneDrive\Desktop\FaceRec\Iris\IRIS.csv")

#Forming the training and testing set
train_set, test_set = train_test_split(df , test_size = 0.2)
train_set = df[:120]
test_set = df[120:]

#Splitting the dataset
X_test = test_set
X_train = train_set
Y_train = train_set.pop('species')
Y_test = test_set.pop('species')

#Seperating the data based on the Y-labels.
X = []
class_column = "species"
grouped_data = df.groupby(class_column)
for label, group in grouped_data:
    X.append(group.drop("species",axis = 1))


#Using LDA method
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 1)
model = lda.fit(X_train,Y_train)
pred_LDA = model.predict(X_test)


print("Accuracy score using LDA is:",accuracy_score(pred_LDA,Y_test)*100,"%")
Conf_matrix_LDA = confusion_matrix(pred_LDA,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_LDA))
print("Wrong predictions:", np.sum(Conf_matrix_LDA)-np.sum(np.trace(Conf_matrix_LDA)))
print("Cross entropy: ",cross_entropy(Y_test,pred_LDA))

'''
#Using PCA

pred_PCA = []
from sklearn.decomposition import PCA
model = PCA(n_components = 1)
model.fit(X_train)
transformed_data = model.transform(X_test)
projected_data_test = model.inverse_transform(transformed_data)
projected_data_PCA0 = model.inverse_transform(model.transform(X[0]))
projected_data_PCA1 = model.inverse_transform(model.transform(X[1]))
projected_data_PCA2 = model.inverse_transform(model.transform(X[2]))
dist0_PCA0, dist1_PCA1, dist2_PCA2 = [],[],[]    
for i in range(30):
    dist0_PCA0.append(np.linalg.norm(projected_data_test[i] - projected_data_PCA0[i]))
    dist1_PCA1.append(np.linalg.norm(projected_data_test[i] - projected_data_PCA1[i]))
    dist2_PCA2.append(np.linalg.norm(projected_data_test[i] - projected_data_PCA2[i]))
    dist0_PCAm = np.mean(dist0_PCA0)
    dist1_PCAm = np.mean(dist1_PCA1)
    dist2_PCAm = np.mean(dist2_PCA2)
    dist = [dist0_PCAm,dist1_PCAm,dist2_PCAm]
    index = dist.index(min(dist))
    #append the class label names
    if index==0:    
        pred_PCA.append("Iris-setosa")
    if index==1:    
        pred_PCA.append("Iris-versicolor")
    if index==2:    
        pred_PCA.append("Iris-virginica")

print("Accuracy score using PCA is:",accuracy_score(pred_PCA,Y_test)*100,"%")
Conf_matrix_PCA = confusion_matrix(pred_PCA,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_PCA))
print("Wrong predictions:", np.sum(Conf_matrix_PCA)-np.sum(np.trace(Conf_matrix_PCA)))

#Using manual method

mean0 = np.mean(X[0],axis = 0)
mean1 = np.mean(X[1],axis = 0)
mean2 = np.mean(X[2],axis = 0)

pred_manual = [] 
cov0 = np.cov(X[0].T)
eigenvalues0,eigenvectors0 = np.linalg.eigh(cov0)
sort_index0 = np.argsort(eigenvalues0)[::-1]
eigenvalues0 = eigenvalues0[sort_index0]
eigenvectors0 = eigenvectors0[:,sort_index0]
projected_data0 = X[0].dot(eigenvectors0[:,:30])

cov1 = np.cov(X[1].T)
eigenvalues1,eigenvectors1 = np.linalg.eigh(cov1)
sort_index1 = np.argsort(eigenvalues1)[::-1]
eigenvalues1 = eigenvalues1[sort_index1]
eigenvectors1 = eigenvectors1[:,sort_index1]
projected_data1 = X[1].dot(eigenvectors1[:,:30])

cov2 = np.cov(X[2].T)
eigenvalues2,eigenvectors2 = np.linalg.eigh(cov2)
sort_index2 = np.argsort(eigenvalues2)[::-1]
eigenvalues2 = eigenvalues2[sort_index2]
eigenvectors2 = eigenvectors2[:,sort_index2]
projected_data2 = X[2].dot(eigenvectors2[:,:30])

cov_test = np.cov(X_test)
eigenvalues_test,eigenvectors_test = np.linalg.eigh(cov_test)
sort_index_test = np.argsort(eigenvalues_test)[::-1]
eigenvalues_test = eigenvalues_test[sort_index_test]
eigenvectors_test = eigenvectors_test[:,sort_index_test]
projected_data_test_manual = X_test.dot(eigenvectors2[:,:30])

for j in range(120,150):
    dist0_PCA, dist1_PCA, dist2_PCA = [],[],[]    
    for i in range(4):
        dist0_PCA.append(np.linalg.norm(projected_data_test_manual[i][j] - projected_data0[i][j-120]))
        dist1_PCA.append(np.linalg.norm(projected_data_test_manual[i][j] - projected_data1[i][j-70]))
        dist2_PCA.append(np.linalg.norm(projected_data_test_manual[i][j] - projected_data2[i][j-20]))
        dist0_PCAm = np.mean(dist0_PCA)
        dist1_PCAm = np.mean(dist1_PCA)
        dist2_PCAm = np.mean(dist2_PCA)
    dist = [dist0_PCAm,dist1_PCAm,dist2_PCAm]
    index = dist.index(min(dist))
    #append the class label names
    if index==0:    
        pred_manual.append("Iris-setosa")
    if index==1:    
        pred_manual.append("Iris-versicolor")
    if index==2:    
        pred_manual.append("Iris-virginica")

#Measure the accuracy score
print("Accuracy score using manual method is:",accuracy_score(pred_manual,Y_test)*100,"%")
Conf_matrix_manual = confusion_matrix(pred_manual,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_manual))
print("Wrong predictions:", np.sum(Conf_matrix_manual)-np.sum(np.trace(Conf_matrix_manual)))

#Using FFT method

pred_FFT = []
new_data_FFT0 = abs(ifft2(fft2(X[0])))
new_data_FFT1 = abs(ifft2(fft2(X[1])))
new_data_FFT2 = abs(ifft2(fft2(X[2])))
projected_data_test_FFT = abs(ifft2(fft2(X_test)))
#Using the segregated classes, calculate the distance and hence the Label of the case

for i in range(30):
    dist0_PCA, dist1_PCA, dist2_PCA = [],[],[]    
    dist0_PCA.append(np.linalg.norm(projected_data_test_FFT[i] - new_data_FFT0[i]))
    dist1_PCA.append(np.linalg.norm(projected_data_test_FFT[i] - new_data_FFT1[i]))
    dist2_PCA.append(np.linalg.norm(projected_data_test_FFT[i] - new_data_FFT2[i]))
    dist0_PCAm = np.mean(dist0_PCA)
    dist1_PCAm = np.mean(dist1_PCA)
    dist2_PCAm = np.mean(dist2_PCA)
    dist = [dist0_PCAm,dist1_PCAm,dist2_PCAm]
    index = dist.index(min(dist))
    #append the class label names
    if index==0:    
        pred_FFT.append("Iris-setosa")
    if index==1:    
        pred_FFT.append("Iris-versicolor")
    if index==2:    
        pred_FFT.append("Iris-virginica")

#Measure accuracy score
print("Accuracy score using FFT method is:",accuracy_score(pred_FFT,Y_test)*100,"%")
Conf_matrix_FFT = confusion_matrix(pred_FFT,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_FFT))
print("Wrong predictions:", np.sum(Conf_matrix_FFT)-np.sum(np.trace(Conf_matrix_FFT)))
'''
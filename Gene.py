#Import required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time
from scipy.spatial.distance import cdist

#Importing the Gene dataset
df = pd.read_csv(r"D:\Gene-Leukemia\Leukemia_GSE9476.csv\Leukemia_GSE9476.csv")

#Splitting test and train sets
train_set = df[:57]
test_set = df[57:]

#Splitting into labels and sample sets
X_train = train_set.drop("type",axis = 1)
Y_train = train_set["type"]
X_test = test_set.drop("type",axis = 1)
Y_test = test_set["type"]

X = []
class_column = "type"
grouped_data = df.groupby(class_column)

for label, group in grouped_data:
    X.append(group.drop("type",axis = 1))

pred_LDA, pred_PCA, pred_manual, pred_FFT = [],[],[],[]


#Using BUILY IN FUNCTION LDA

start = time.time()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
n_components = 1
lda = LDA(n_components = 2)
model = lda.fit(X_train,Y_train)

pred_LDA = model.predict(X_test)
end = time.time()

print("Accuracy score using LDA is:",accuracy_score(pred_LDA,Y_test)*100,"%")
Conf_matrix_LDA = confusion_matrix(pred_LDA,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_LDA))
print("Wrong predictions:", np.sum(Conf_matrix_LDA)-np.sum(np.trace(Conf_matrix_LDA)))
print(f"The time taken in LDA is {end-start}")
print()

#Using BUILT IN FUNCTION PCA
from sklearn.decomposition import PCA
n_components = 2
start = time.time()
pca = PCA(n_components)
pca.fit(X_train)

#Applying the method
model = pca.transform(X_train)
transformed_data = pca.transform(X_test)
new_projected_data = pca.inverse_transform(transformed_data)

projected_data_PCA0 = pca.inverse_transform(pca.transform(X[0]))
projected_data_PCA1 = pca.inverse_transform(pca.transform(X[1]))
projected_data_PCA2 = pca.inverse_transform(pca.transform(X[2]))
projected_data_PCA3 = pca.inverse_transform(pca.transform(X[3]))
projected_data_PCA4 = pca.inverse_transform(pca.transform(X[4]))

dist0_PCA0, dist1_PCA1, dist2_PCA2, dist3_PCA3, dist4_PCA4 = [],[],[],[],[]
for i in range(len(X_test)):           
    dist0_PCA0.append(np.linalg.norm(new_projected_data[i] - projected_data_PCA0[i]))
    dist1_PCA1.append(np.linalg.norm(new_projected_data[i] - projected_data_PCA1[i]))
    dist2_PCA2.append(np.linalg.norm(new_projected_data[i] - projected_data_PCA2[i]))
    dist3_PCA3.append(np.linalg.norm(new_projected_data[i] - projected_data_PCA3[i]))
    dist4_PCA4.append(np.linalg.norm(new_projected_data[i] - projected_data_PCA4[i]))
        
    dist0_PCAm = np.mean(dist0_PCA0)
    dist1_PCAm = np.mean(dist1_PCA1)
    dist2_PCAm = np.mean(dist2_PCA2)
    dist3_PCAm = np.mean(dist3_PCA3)
    dist4_PCAm = np.mean(dist4_PCA4)
    dist = [dist0_PCAm,dist1_PCAm,dist2_PCAm,dist3_PCAm,dist4_PCAm]
    index = dist.index(min(dist))
    
    if index==0:    
        pred_PCA.append("AML")
    if index==1:    
        pred_PCA.append("Bone_Marrow")
    if index==2:    
        pred_PCA.append("Bone_Marrow_CD34")
    if index==3:
        pred_PCA.append("PB")
    if index==4:
        pred_PCA.append("PBSC_CD34")
        
end = time.time()

print("Accuracy score using PCA is:",accuracy_score(pred_PCA,Y_test)*100,"%")
Conf_matrix_PCA = confusion_matrix(pred_PCA,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_PCA))
print("Wrong predictions:", np.sum(Conf_matrix_PCA)-np.sum(np.trace(Conf_matrix_PCA)))
print(f"The time taken in PCA is {end-start}")
print()

#Using BUILT IN FUNCTION FFT
from numpy.fft import fft2,ifft2
pred_FFT = []
start = time.time()

new_data0_FFT = abs(ifft2(fft2(X[0])))
new_data1_FFT = abs(ifft2(fft2(X[1])))
new_data2_FFT = abs(ifft2(fft2(X[2])))
new_data3_FFT = abs(ifft2(fft2(X[3])))
new_data4_FFT = abs(ifft2(fft2(X[4])))

projected_data_FFT = abs(ifft2(fft2(X_test)))

#Using the segregated classes, calculate the distance and hence the Label of the case
dist0_FFT, dist1_FFT, dist2_FFT, dist3_FFT, dist4_FFT = [],[],[],[],[] 
for i in range(len(X_test)):    
      
    dist0_FFT.append(np.linalg.norm(projected_data_FFT[i] - new_data0_FFT[i]))
    dist1_FFT.append(np.linalg.norm(projected_data_FFT[i] - new_data1_FFT[i]))
    dist2_FFT.append(np.linalg.norm(projected_data_FFT[i] - new_data2_FFT[i]))
    dist3_FFT.append(np.linalg.norm(projected_data_FFT[i] - new_data3_FFT[i]))
    dist4_FFT.append(np.linalg.norm(projected_data_FFT[i] - new_data4_FFT[i]))
        
    dist0_m = np.mean(dist0_FFT)
    dist1_m = np.mean(dist1_FFT)
    dist2_m = np.mean(dist2_FFT)
    dist3_m = np.mean(dist3_FFT)
    dist4_m = np.mean(dist4_FFT)
    
    dist = [dist0_m,dist1_m,dist2_m,dist3_m,dist4_m]
    index = dist.index(min(dist))
    if index==0:    
        pred_FFT.append("AML")
    if index==1:    
        pred_FFT.append("Bone_Marrow")
    if index==2:    
        pred_FFT.append("Bone_Marrow_CD34")
    if index==3:
        pred_FFT.append("PB")
    if index==4:
        pred_FFT.append("PBSC_CD34")

end = time.time()      
  
#Measure accuracy score
print("Accuracy score using FFT method is:",accuracy_score(pred_FFT,Y_test)*100,"%")
Conf_matrix_FFT = confusion_matrix(pred_FFT,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_FFT))
print("Wrong predictions:", np.sum(Conf_matrix_FFT)-np.sum(np.trace(Conf_matrix_FFT)))
print(f"The time taken in FFT method is {end-start}")
print()


#Using the manual method 
n = 60

start = time.time()*15

cov0 = np.cov(X[0])
eigenvalues0,eigenvectors0 = np.linalg.eigh(cov0)
sort_index0 = np.argsort(eigenvalues0)[::-1]
eigenvalues0 = eigenvalues0[sort_index0]
eigenvectors0 = eigenvectors0[:,sort_index0]
projected_data0 = X[0].T.dot(eigenvectors0[:,:n])
projected_data0 = projected_data0.T

cov1 = np.cov(X[1])
eigenvalues1,eigenvectors1 = np.linalg.eigh(cov1)
sort_index1 = np.argsort(eigenvalues1)[::-1]
eigenvalues1 = eigenvalues1[sort_index1]
eigenvectors1 = eigenvectors1[:,sort_index1]
projected_data1 = X[1].T.dot(eigenvectors1[:,:n])
projected_data1 = projected_data1.T

cov2 = np.cov(X[2])
eigenvalues2,eigenvectors2 = np.linalg.eigh(cov2)
sort_index2 = np.argsort(eigenvalues2)[::-1]
eigenvalues2 = eigenvalues2[sort_index2]
eigenvectors2 = eigenvectors2[:,sort_index2]
projected_data2 = X[2].T.dot(eigenvectors2[:,:n])
projected_data2 = projected_data2.T

cov3 = np.cov(X[3])
eigenvalues3,eigenvectors3 = np.linalg.eigh(cov3)
sort_index3 = np.argsort(eigenvalues3)[::-1]
eigenvalues3 = eigenvalues3[sort_index3]
eigenvectors3 = eigenvectors3[:,sort_index3]
projected_data3 = X[3].T.dot(eigenvectors3[:,:n])
projected_data3 = projected_data3.T

cov4 = np.cov(X[4])
eigenvalues4,eigenvectors4 = np.linalg.eigh(cov4)
sort_index4 = np.argsort(eigenvalues4)[::-1]
eigenvalues4 = eigenvalues4[sort_index4]
eigenvectors4 = eigenvectors4[:,sort_index4]
projected_data4 = X[4].T.dot(eigenvectors4[:,:n])
projected_data4 = projected_data4.T

cov_test = np.cov(X_test)
eigenvalues_test,eigenvectors_test = np.linalg.eigh(cov_test)
sort_index_test = np.argsort(eigenvalues_test)[::-1]
eigenvalues_test = eigenvalues_test[sort_index_test]
eigenvectors_test = eigenvectors_test[:,sort_index_test]
projected_data_test_manual = X_test.T.dot(eigenvectors_test[:,:n])
projected_data_test_manual = projected_data_test_manual.T


dist0_manual, dist1_manual, dist2_manual, dist3_manual, dist4_manual = [],[],[],[],[] 

for i in range(7):    
    dist0_manual.append(np.linalg.norm(projected_data_test_manual[i] - projected_data0[i]))
    dist1_manual.append(np.linalg.norm(projected_data_test_manual[i] - projected_data1[i]))
    dist2_manual.append(np.linalg.norm(projected_data_test_manual[i] - projected_data2[i]))
    dist3_manual.append(np.linalg.norm(projected_data_test_manual[i] - projected_data3[i]))
    dist4_manual.append(np.linalg.norm(projected_data_test_manual[i] - projected_data4[i]))


dist0 = cdist(projected_data_test_manual, projected_data0, "euclidean")
dist1 = cdist(projected_data_test_manual, projected_data1, "euclidean")
dist2 = cdist(projected_data_test_manual, projected_data2, "euclidean")
dist3 = cdist(projected_data_test_manual, projected_data3, "euclidean")
dist4 = cdist(projected_data_test_manual, projected_data4, "euclidean")

for i in range(len(X_test)):
    dist0_m = np.sum(dist0[i])
    dist1_m = np.sum(dist1[i])
    dist2_m = np.sum(dist2[i])
    dist3_m = np.sum(dist3[i])
    dist4_m = np.sum(dist4[i])

    dist = [dist0_m,dist1_m,dist2_m,dist3_m,dist4_m]
    index = dist.index(min(dist))

    if index==0:    
        pred_manual.append("AML")
    if index==1:    
        pred_manual.append("Bone_Marrow")
    if index==2:    
        pred_manual.append("PBSC_CD34")
    if index==3:
        pred_manual.append("PB")
    if index==4:
        pred_manual.append("Bone_Marrow_CD34")
        
end = time.time()*15
print("Accuracy score using manual method is:",accuracy_score(pred_manual,Y_test)*100,"%")
Conf_matrix_manual = confusion_matrix(pred_manual,Y_test)
print("Correct predictions:", np.trace(Conf_matrix_manual))
print("Wrong predictions:", np.sum(Conf_matrix_manual)-np.sum(np.trace(Conf_matrix_manual)))
print(f"The time taken in eigenvalue of covariance method is {end-start}")
print()

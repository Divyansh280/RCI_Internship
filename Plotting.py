import matplotlib.pyplot as plt

'''
#MNIST

number_of_comp = [50,100,150,200,250,300]
no_of_comp = [2,4,5,6,8,9]
manual = [92.04,95.13,95.97,96.27,96.5,96.64]
PCA = [92.74,94.47,94.53,94.62,94.97,95.21]
FFT = [83.22,84,84,84,93.89,94]
LDA = [87.3,87.4,87.5,87.6,93.89,93.89]

loss_manual = [5715.62,3056.35,2156.89,1952.39,1950.59,1894.26]
loss_LDA = [10867.99,10865.00,10866.87,10813.65,9965.99,9965.80]
loss_PCA = [5671.61,4875.28,3620.13,3157.53,2823.61,2356.52]
loss_FFT = [5741.63,5654.78,5654.6,5654.6,3148.99,3112.67]


plt.plot(number_of_comp,manual)
for i in range(len(number_of_comp)):
    plt.annotate(loss_manual[i], (number_of_comp[i], manual[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel("Number of Components")
plt.ylabel("Accuracy Score of Eigenvalue of Cov method")
plt.show()


plt.plot(no_of_comp,PCA)
for i in range(len(no_of_comp)):
    plt.annotate(loss_PCA[i], (no_of_comp[i], PCA[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel("Number of Components")
plt.ylabel("Accuracy Score of PCA method")
plt.show()


plt.plot(no_of_comp,LDA)
for i in range(len(no_of_comp)):
    plt.annotate(loss_LDA[i], (no_of_comp[i], LDA[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel("Number of Components")
plt.ylabel("Accuracy Score of LDA method")
plt.show()


plt.plot(no_of_comp,FFT)
for i in range(len(no_of_comp)):
    plt.annotate(loss_FFT[i], (no_of_comp[i], FFT[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel("Number of Components")
plt.ylabel("Accuracy Score of FFT method")
plt.show()


plt.title("Accuracy Score Vs components")
plt.legend(["manual","PCA","FFT"])
plt.xlabel("No of Components")
plt.ylabel("Accuracy Score")
plt.show()



#IRIS

PCA = [86.67,87.3,92.4,100,100]
LDA = [16.6,20,81.3,97,100]
manual = [50,50,66.6,80,100]
FFT = [15,35,66.6,66.6,80]
training = [45,60,80,85,90]

plt.plot(training,PCA)
plt.xlabel("Training Set size")
plt.ylabel("Accuracy Score of PCA")

plt.plot(training,LDA)
plt.xlabel("Training Set size")
plt.ylabel("Accuracy Score of LDA")

plt.plot(training,manual)
plt.xlabel("Training Set size")
plt.ylabel("Accuracy Score of Manual")

plt.plot(training,FFT)
plt.xlabel("Training Set size")
plt.ylabel("Accuracy Score")
plt.legend(["PCA","LDA","manual","FFT"])
plt.show()



#GENE

number_of_components = [1,2,3,4]
time_LDA = [0.2,0.24,0.301,0.3516]
time_PCA = [0.71,0.75,0.75,0.94]
time_FFT = [0.25,0.252,0.263,0.381]
number = [10,20,30,40]
time_manual = [0.4999,0.5034,0.6046,0.7029]

plt.plot(number_of_components,time_LDA)
plt.title("Time taken for technique Vs No of Components")
plt.xlabel("Number of components")
plt.ylabel("Time taken for LDA technique")
plt.show()

plt.plot(number_of_components,time_PCA)
plt.title("Time taken for technique Vs No of Components")
plt.xlabel("Number of components")
plt.ylabel("Time taken for PCA technique")
plt.show()

plt.plot(number_of_components,time_FFT)
plt.title("Time taken for technique Vs No of Components")
plt.xlabel("Number of components")
plt.ylabel("Time taken for FFT technique")
plt.show()

plt.plot(number,time_manual)
plt.title("Time taken for technique Vs No of Components")
plt.xlabel("Number of eigenvectors considered")
plt.ylabel("Time taken for EigenvalueOfCovariance technique")
plt.show()


plt.plot(number_of_components,time_LDA)
plt.plot(number_of_components,time_PCA)
plt.plot(number_of_components,time_FFT)
#plt.plot(number_of_components,time_manual)
plt.title("Time taken for technique Vs No of Components")
plt.xlabel("Number of components")
plt.ylabel("Time taken by the technique")
plt.legend(["LDA","PCA","FFT"])
plt.show()
'''


#CNN

Epochs = [1,2,3,4,5]
accuracy = [31.62,32.43,35.24,43.12,51.62]
loss = [2.1,1.2,0.9,0.61,0.48]
accuracy_PCA = [57.5,62,65.3,66.7,68.4]
accuracy_LDA = [65.7,67.6,68.9,75.6,77.9]
accuracy_Eigen = [65.6,71.2,74.6,76.5,81.3]

'''
plt.plot(Epochs,accuracy)
plt.ylabel("Accuracy without applying any technique")
plt.title("Accuracy score Vs Epochs for CNN")
plt.show()
'''

'''
#CNN - PCA

plt.plot(Epochs,accuracy_PCA)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy after applying PCA")
plt.title("Accuracy score Vs Epochs for CNN")
plt.show()


#CNN - LDA
plt.plot(Epochs,accuracy_LDA)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy after applying LDA")
plt.title("Accuracy score Vs Epochs for CNN")
plt.show()

'''
#CNN - Eigenvalue of Covariance
plt.plot(Epochs,accuracy_Eigen)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy after applying Eigenvalue of cov")
plt.title("Accuracy score Vs Epochs for CNN")
plt.show()


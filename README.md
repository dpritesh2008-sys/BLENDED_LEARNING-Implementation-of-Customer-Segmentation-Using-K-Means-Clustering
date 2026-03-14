# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Dataset Loading – Load the customer dataset using pandas and select features such as Age, Annual Income, and Spending Score.

2.Feature Scaling – Standardize the selected features using StandardScaler to bring them to the same scale.

3.K-Means Clustering – Apply the K-Means algorithm with different numbers of clusters to group similar customers.

4.Elbow Method Visualization – Plot inertia values for different clusters to determine the optimal number of clusters.

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: Ritesh DP
RegisterNumber:  212225040339
*/
import os 
os.environ["OMPI NUM_THREADS"] = "1" 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn. cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
 
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

data = pd.read_csv('CustomerData.csv')

print(data.head()) 
print(data.columns)

features = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
X = data [features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10) # Explicit n init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
print("Name:Balasurya S")
print("Register No:212225100003")

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-') 
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
```

## Output:



## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 

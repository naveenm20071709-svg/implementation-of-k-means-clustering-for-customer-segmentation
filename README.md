# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Prepare the data by selecting relevant features for clustering.

2)Find the optimal number of clusters using the Elbow Method (WCSS vs. k).

3)Apply K-Means algorithm to assign data points to clusters.

4)Visualize and analyze the formed clusters with their centroids.

## Program:


```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Naveen M
RegisterNumber:  212225230198
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [19, 21, 20, 23, 31, 35, 40, 50, 55, 60],
    'AnnualIncome(k$)': [15, 16, 17, 18, 30, 40, 60, 80, 85, 90],
    'SpendingScore(1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 60]
}

df = pd.DataFrame(data)
print("Customer Data:")
print(df)

X = df[['AnnualIncome(k$)', 'SpendingScore(1-100)']]


wcss = []  
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Inertia)')
plt.show()


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)


df['Cluster'] = y_kmeans
print("\nCustomer Segmentation Results:")
print(df)


plt.figure(figsize=(8, 6))
plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1], s=80, c='red', label='Cluster 1')
plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1], s=80, c='blue', label='Cluster 2')
plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1], s=80, c='green', label='Cluster 3')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', marker='X', label='Centroids')

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.grid(True)
plt.show()

```

## Output:
<img width="607" height="272" alt="Screenshot 2026-05-16 093358" src="https://github.com/user-attachments/assets/2999b74e-0ef2-44fa-83fe-8ffacc4b8149" />
<img width="767" height="536" alt="Screenshot 2026-05-16 093407" src="https://github.com/user-attachments/assets/00891a74-3cf1-47c8-b32a-52e261f0aedc" />
<img width="847" height="354" alt="Screenshot 2026-05-16 093415" src="https://github.com/user-attachments/assets/37de2688-5351-4a07-9364-91d95e7cd0dc" />
<img width="900" height="639" alt="Screenshot 2026-05-16 093437" src="https://github.com/user-attachments/assets/f43f1afe-64a5-4658-8ecd-d2e5877518fa" />







## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

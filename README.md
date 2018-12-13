# CS6068 Parallel Computing Final Project
## Parallel K-Means


### Background  

K-Means Clustering is one of the popularly used unsupervised learning algorithms because of its simplicity and versatility. It's called simple as it has an iterative set of steps that is done until the results converge at an acceptable point.  

I am a machine learning enthusiast & have used this algorithm in a few of my projects before. I always thought that K-Means is a simple but versatile clustering algorithm with a decent time complexity.   

Given a dataset of n data points, the time complexity is O(nkt) where n is the number of data points, k is number of clusters formed and t is the number of iterations it takes for convergence.   

The problem is that it works slowly for a data set which is huge and/or has a relatively high number of clusters. To solve this issue, using data parallelism to calculate distances may result in better runtime.   


### Objectives

The sequential K-Means algorithms runs iterations of the same set of steps until it reaches and acceptable convergence. Initially k cluster centroids are chosen from the datapoints randomly/ using some criteria. Enhancing choosing methods for the centroids is not a part of this project.   

K-Means Algorithm Steps:
1) Find the distance of each data point to all centroids.   
2) Assign each data point to cluster with closest centroid.   
3) Update Centroid Values   

Repeat until there is no difference between the consecutive iteration results.  


Out of the above-mentioned steps, maximum time taken is to calculate the distance from each centroid. The same operation is done on each datapoint though at each iteration there is a variation in the centroids used.   
The main goal for this project is to improve run time for large datasets with more than 3 attributes by parallelizing the distance calculation for each datapoint.


### Files:

#### pkmeans.py   
Contains the code to sequential and parallel implementations of the K-Means algorithm.

#### Data    
Datasets used are stored in the data folder. Make sure download code with data for proper execution.

#### PK-Means - Report & PK-Means - Presentation   
Explaination about the project and how it was executed

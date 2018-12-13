import pandas as pd 
import numpy as np
from math import sqrt
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Randomly generating initial centroids
def choose_centroids(n,k):
    import random as rd
    centroids_idx = rd.sample(range(n),k)
    centroids_idx.sort()
    return centroids_idx

def distance(centroid,datapoint):
    return sqrt(np.square(centroid-datapoint).sum())

def Kmeans(df,cidx,n,k):
    iterations = 0
    time = []
    centroids = df.loc[cidx]
    clusters = np.zeros(n,dtype=int)
    
    while iterations < 100:
        startit = timer()
        
        iterations += 1
        old_clusters = clusters.copy()

        #Part to be Parallelized
        for i in range(n):
            dist = []
            for j in range(k):
                dist.append(distance(centroids.loc[centroids_idx[j]],df.loc[i]))
            clusters[i] = dist.index(min(dist))
        #Parallellizing ends here
        endit = timer() - startit
        time.append(endit)
        
        if np.array_equal(clusters,old_clusters):
            break
        
        # update Cluster centroids
        for j in range(k):
            centroids.loc[centroids_idx[j]] = df.loc[clusters == j].mean(axis=0)

    return np.mean(np.asarray(time)),clusters

import multiprocessing as mp
from scipy.spatial.distance import cdist
from itertools import repeat

def find_cluster(data,centroids):
    distances = cdist(data,centroids,'euclidean')
    return np.argmin(distances, axis = 1)

def mpKmeans(df,cidx,n,k,cpus):
    iterations = 0
    time = []
    centroids = df[cidx]
    clusters = np.zeros(n,dtype=int)
    
    split_data = np.array_split(df, cpus)
    
    while iterations < 100:
        startit = timer()
        
        iterations += 1
        old_clusters = clusters.copy()
        
        arg = zip(split_data, repeat(centroids))
        
        pool = mp.Pool(processes=cpus)
        split_clusters = pool.starmap(find_cluster,arg)
        pool.close()
        pool.join()
        
        clusters = np.concatenate(split_clusters)
        
        endit = timer() - startit
        time.append(endit)
        
        if np.array_equal(clusters,old_clusters):
            break
        
        # update Cluster centroids
        for j in range(k):
            centroids[j] = df[clusters == j].mean(axis=0)
        
    return np.mean(np.asarray(time)),clusters

if __name__ == "__main__":
    df = pd.read_csv('data\\Wholesale customers data.csv',sep=',')
#    df = pd.read_csv('data\\winequality-white.csv',sep=';')
#    df = df[df.columns[:8]]

    # Parameters for Kmeans
    k = [3,4,5,6,7,8,9] # Number of Clusters
    n = df.shape[0] # Number of datapoints
    att = df.shape[1] # Number of Attributes
    
    print("Number of Datapoints =",n)
    print("Number of Attributes=",att)
    
    #time taken to run different algorithms
    time_kmeans = np.zeros(len(k),dtype=float)
    time_mp = np.zeros(len(k),dtype=float)
    
    #iteration count in different algortihms
    iter_kmeans = np.zeros(len(k),dtype=float)
    iter_mp = np.zeros(len(k),dtype=float)
    
    
    #Differnces in the results
    diff_mp_km = np.zeros(len(k),dtype=int)
    
    # Initial centroids
    for i in range(len(k)):
        centroids_idx = choose_centroids(n,k[i])
        print("Number of Clusters to be formed:",k[i])
        print("\tTime\tIterations")
                
        #running normal kmeans algorithms
        start_k = timer()
        iter_kmeans[i],clusters_kmeans = Kmeans(df,centroids_idx,n,k[i])
        time_kmeans[i] = timer() - start_k
        print('Kmeans:\t%4.4f\t%4.4f'%(time_kmeans[i],iter_kmeans[i]))
        
        #running kmeans algorithm with parallel computing using Multiprocessing module
        start_m = timer()
        cpus = mp.cpu_count()
        iter_mp[i],clusters_mp = mpKmeans(df.values,centroids_idx,n,k[i],cpus)
        time_mp[i] = timer() - start_m
        print('MProcs:\t%4.4f\t%4.4f'%(time_mp[i],iter_mp[i]))
        

        #Calculating difference between results
        diff_mp_km[i] = sum(clusters_kmeans != clusters_mp)        

        if diff_mp_km[i] != 0:
            print("Cluster Differences: Kmeans Vs Multiprocessing :",diff_mp_km[i])
            
    #Plotting Total Exceution Time
    plt.plot(k,time_kmeans,'r-',k,time_mp,'g-')
    plt.title("Time Comparisons for Data of Size 4898 x 8")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Time Taken to Run Clustering Algorithm")
    plt.legend(("Sequencial K-Means","Parallel K-Means"),loc='best')
    plt.show()
    
    #Plottting Mean Iteration Time
    plt.plot(k,iter_kmeans,'r-',k,iter_mp,'g-')
    plt.title("Mean Iteration Time Comparisons for Data of Size 4898 x 8")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Mean Time Taken to Run 1 Iteration of KMeans Clustering Algorithm")
    plt.legend(("Sequencial K-Means","Parallel K-Means"),loc='best')
    plt.show()
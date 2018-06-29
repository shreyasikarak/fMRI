import numpy as np
from matplotlib import pyplot as plt
import pdb
import pandas as pd
import cv2

'''images = plt.imread('image.png')
#print(data.shape)
k = 2
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)
 
images = np.random.random_sample((10))
images = images.reshape((10,10))
images.shape(100,100)'''

images = plt.imread('cropa.png')
gray_image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)*255

'''segment to crop'''

row = gray_image.shape[0]
col = gray_image.shape[1]
resized_image = np.resize(gray_image,(row*col,1))

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
K = 5

ret,label,center=cv2.kmeans(resized_image, K, None, criteria, 10,
							cv2.KMEANS_RANDOM_CENTERS)

#first_cluster = np.where((label==0)==True)[0]
#label[first_cluster[0]] = 0.3

higher_index = np.argsort(np.resize(center,5))

second_cluster = np.where((label==higher_index[K-1])==True)[0]

val_to_assign = np.mean([center[higher_index[K-1]],1])
print(higher_index[K-1], val_to_assign)
resized_image[second_cluster] = val_to_assign

third_cluster = np.where((label==higher_index[K-2])==True)[0]
val_to_assign = np.mean([center[higher_index[K-1]],center[higher_index[K-2]]])
print(higher_index[K-2], val_to_assign)
resized_image[third_cluster] = val_to_assign

recreate_image = np.resize(resized_image,(row,col))
plt.imshow(recreate_image)
plt.show()
pdb.set_trace()

'''fourth_cluster = np.where((label==4)==True)[0]
resized_image[fourth_cluster] = np.mean([0.11079925,0.00443988])

fifth_cluster = np.where((label==5)==True)[0]
resized_image[fifth_cluster] = np.mean([0.00443988, 0.289724])'''





def euclidian(a, b):
	return np.linalg.norm(a-b)

def plot(dataset, history_centroids, belongs_to):
    colors = ['r', 'g']

    fig, ax = plt.subplots()

    for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        pdb.set_trace()
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    history_points = []
    pdb.set_trace()

    for index, centroids in enumerate(history_centroids):
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))

                plt.pause(0.8)

def kmeans(k, epsilon=0, distance='euclidian'):
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian
    
        images = plt.imread('cropa.png')
    
    num_instances, num_features = resized_image.shape
    #prototypes = resized_image[np.random.randint(0, num_instances - 1, size=k)]

    prototypes = np.array([[0. ], [0.2], [0.4], [0.6],[1]])

    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    belongs_to = np.zeros((num_instances, 1))
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(resized_image):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype, instance)

            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(resized_image[instances_close], axis=0)
            
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes

        history_centroids.append(tmp_prototypes)

  
    return prototypes, history_centroids, belongs_to


def execute():
    
	images = plt.imread('cropa.png')
	centroids, history_centroids, belongs_to = kmeans(5, 0,'euclidian')
	pdb.set_trace()
	
	# resized_image[np.where((resized_image>=0.6)==True)] = 1
	recreate_image = np.resize(resized_image,(row,col))
	plt.imshow(recreate_image, cmap='rgb')
	plt.savefig('Clustera.png')

	# plot(resized_image, history_centroids, belongs_to)

execute()

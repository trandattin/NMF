from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt

def get_lfw_people_with_100_images():
    """
    Take the LFW dataset and filter out the people with exactly 100 images.
    Return a tuple (data, target, names) with:
    - data: a 2D array of size (n_samples, n_features) containing feature vectors of the images.
    - target: a 1D array containing labels of each sample in the data.
    - names: a list of names of each person in the dataset.
    """
    lfw_dataset = fetch_lfw_people(min_faces_per_person=100)

    # Get the labels of each person in the dataset.
    unique_targets = np.unique(lfw_dataset.target)
    
    # Initialize a list to contain images of each person with exactly 100 images.
    data_list = []
    target_list = []
    names_list = []

    # Loop through the labels of each person.
    for target in unique_targets:
        # Takes the images of that person.
        images = lfw_dataset.data[lfw_dataset.target == target]
        
        # If the number of images of that person is greater than 100, only take the first 100 images.
        if len(images) > 100:
            images = images[:100]
        
        # Add the image to the list.
        data_list.append(images)
        target_list.append(np.full((len(images),), target))
        names_list.append(lfw_dataset.target_names[target])

    # Combine the list of images into a 2-dimensional array.
    data = np.concatenate(data_list)
    target = np.concatenate(target_list)
    names = np.array(names_list)
    image_shape = (lfw_dataset.images.shape[1], lfw_dataset.images.shape[2])
    
    return data, target, names, image_shape

def create_image_matrix(data):
    """
    Create the V matrix from the LFW image dataset.
    Return the V matrix with size (n_features, n_samples).
    """
    return np.reshape(data, (data.shape[0], -1)).T

def plot_50_images(V, image_shape):
    """
    Display 50 random images from matrix V in a 10x5 grid format.
    """
    # "Randomly select 50 indices of images
    idx = np.random.choice(V.shape[1], size=50, replace=False)
    
    # Create figure and axes
    fig, ax = plt.subplots(5, 10, figsize=(image_shape[0]/5, image_shape[1]/5))
    fig.subplots_adjust(hspace=-0.44, wspace=0.1)
    
    # Display the 50 images corresponding to the randomly selected indices.
    for i, axi in enumerate(ax.flat):
        if i < 50:
            axi.imshow(np.reshape(V[:, idx[i]], image_shape), cmap="gray")
            axi.axis("off")

def plot_5_people(data, target, names, image_shape):
    """
    Display 5 faces of 5 different people with corresponding names.
    """
    # Get the indices corresponding to each person in the dataset.
    people_indices = [np.where(target == i)[0] for i in range(len(names))]
        
    # Create figure and axes
    fig, ax = plt.subplots(1, 5, figsize=(image_shape[0]/5, image_shape[1]/5))
    fig.subplots_adjust(hspace=-0.3, wspace=0.1)
    
    # Display 5 faces of 5 different people along with their corresponding names.
    for i, axi in enumerate(ax.flat):
        person_name = names[i]
        image_idx = np.random.choice(people_indices[i])
        image = data[image_idx]
        axi.imshow(np.reshape(image, image_shape), cmap="gray")
        axi.axis("off")
        axi.set_title(person_name)
    
    plt.show()

def normalize(V,W,H):
    """
    how 50 images corresponding to randomly selected stats
    """
    r = W.shape[1]
    K = W @ H.T
    alpha = np.dot(V.ravel(), K.ravel()) / np.dot(K.ravel(), K.ravel())
    D = np.diag(np.sqrt(np.sum(H**2, axis=0) / np.sum(W**2, axis=0)))
    W = W @ D * np.sqrt(alpha)
    H = H @ np.linalg.inv(D) * np.sqrt(alpha)
    return W, H

def getMatrix(r):
    """
    Get V,W,H from lfw_dataset for NMF
    r: reduced rank
    """
    data, target, names, image_shape = get_lfw_people_with_100_images()
    V = create_image_matrix(data)
    m, n = V.shape[0], V.shape[1]
    W = np.random.uniform(low=0, high=1, size=(m, r))
    H = np.random.uniform(low=0, high=1, size=(n, r))
    W,H = normalize(V,W,H)  
    return V,W,H

def plotData():
    """
    Plot some image from lfw_data
    """
    data, target, names, image_shape = get_lfw_people_with_100_images()
    V = create_image_matrix(data)
    # Display the first 50 images in the V matrix as a 10x5 grid
    plot_50_images(V, image_shape)	
    # To display 5 people in the image dataset with their names
    plot_5_people(data, target, names, image_shape)

def plot_approxImages(V0, V1, V2, V3):
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(10, 8))
    fig.subplots_adjust(hspace= 0.05, wspace=-0.5)
    for i in range(3):
        for j in range(4):
            axs[i, j].imshow(eval(f'V{j}')[:, i].reshape((62, 47)), cmap='gray')
            axs[i, j].axis('off')
    plt.show()

def train_test_split_custom(X, y, test_size):
    
    permutation = np.random.permutation(len(y))
    shuffled_X = X[:, permutation]
    shuffled_y = y[permutation]
    # Calculate the number of columns to keep in the training set
    n_train = int((1 - test_size) * X.shape[1])

    # Split the rows of data into training and testing sets
    X_train = shuffled_X[:,:n_train]
    y_train = shuffled_y[:n_train]
    X_test = shuffled_X[:,:X.shape[1]-n_train]
    y_test = shuffled_y[:X.shape[1]-n_train]

    return X_train, X_test, y_train, y_test
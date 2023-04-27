# Import các thư viện cần thiết
from sklearn.datasets import fetch_lfw_people  # Cung cấp dataset LFW (Labeled Faces in the Wild)
from sklearn.model_selection import train_test_split  # Chia dữ liệu thành training set và testing set
from sklearn.decomposition import NMF  # Sử dụng thuật toán NMF để giảm chiều dữ liệu
from sklearn.neural_network import MLPClassifier  # Mô hình mạng neuron đa tầng để phân loại khuôn mặt
from sklearn.metrics import classification_report  # Báo cáo đánh giá mô hình
import matplotlib.pyplot as plt  # Thư viện trực quan hóa dữ liệu

# Load data
lfw_dataset = fetch_lfw_people(min_faces_per_person=100)  # Lấy dữ liệu từ dataset LFW, tối thiểu 100 khuôn mặt cho mỗi người

_, h, w = lfw_dataset.images.shape  # Lấy kích thước của hình ảnh khuôn mặt
X = lfw_dataset.data  # Dữ liệu hình ảnh khuôn mặt
y = lfw_dataset.target  # Nhãn tương ứng với mỗi hình ảnh
target_names = lfw_dataset.target_names  # Tên các người trong dataset

# split into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)  # Chia dữ liệu thành training set và testing set với tỷ lệ 7:3

n_components = 20  # Số chiều dữ liệu sau khi giảm chiều

# Compute a NMF
nmf = NMF(n_components = n_components, init = 'random', random_state = 0).fit(X_train)  # Sử dụng thuật toán NMF để giảm chiều dữ liệu

X_train_nmf = nmf.transform(X_train)  # Dữ liệu training set sau khi giảm chiều bằng NMF
X_test_nmf = nmf.transform(X_test)  # Dữ liệu testing set sau khi giảm chiều bằng NMF

# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes =(1024,), batch_size=256, verbose = True, early_stopping = True).fit(X_train_nmf, y_train)  # Sử dụng mô hình mạng neuron đa tầng để phân loại khuôn mặt

y_pred = clf.predict(X_test_nmf)  # Dự đoán nhãn cho testing set

print(classification_report(y_test, y_pred, target_names = target_names))  # Báo cáo đánh giá mô hình

#Visualization

def plot_gallery(images, titles, h, w, rows = 4, cols = 5):
    plt.figure(figsize = (1.8 * cols, 2.4 * rows))
    plt.subplots_adjust(bottom = 0, left = .01, right = .90, top = .90, hspace= .35)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i],size = 12)
        plt.xticks(())
        plt.yticks(())

def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)
eigenfaces = nmf.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface {0}".format(i) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()
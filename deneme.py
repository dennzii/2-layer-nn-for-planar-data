import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset,plot_decision_boundary

#Clipping miktarı
EPSILON = 1e-5

hidden_layer_activation_function = 'tanh'

#Dataset oluşturulur
X,Y = load_planar_dataset()

plt.scatter(X[0,:],X[1,:],c=Y,cmap=plt.cm.Spectral)
plt.show()

#Örnek sayısı
m = X.shape[1]

#İnput ve output layer unit sayıları
n_x = X.shape[0]
n_y = Y.shape[0]

#Ağırlıkların normal dağılım kullanılarak rastgele atanması
def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))#biasları 0 atamak yeterli
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2" : W2,
        "b2" : b2
    }
    return parameters

def forward_propagation(X,parameters,activation):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    #İlk katmanın yayılımı ve aktivasyonu
    Z1 = np.dot(W1,X) + b1
    A1 = relu(Z1) if activation == "relu" else np.tanh(Z1)

    #Output katmanı ve aktivasyonu
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)


    cache = {
        "A1":A1,
        "A2":A2,
        "Z1":Z1,
        "Z2":Z2,
    }
    return A2,cache

def activation_func(z,activation):
    if activation == "relu":
        return relu(z)
    elif activation == "tanh":
        return np.tanh(z)

    return relu(z)

#Rectified Linear Unit aktivasyonu
def relu(z):
    return np.maximum(0,z)

#ReLU türevi
def relu_der(z):
    return np.where(z>0,1,0)

def sigmoid(z):
    return 1/(1+np.exp(-z))


#Logistic loss fonksiyonu
def compute_cost(A2,Y):
    #logaritmaların içinin 0 olup tanımsız olmaması için değerler EPSILON kadar kırpılır.
    A2 = np.clip(A2,EPSILON,1-EPSILON)
    cost = (-1/m)*(np.sum(np.multiply(Y,np.log(A2)) + np.multiply((1-Y),np.log(1-A2)) ))

    #Çıktı boyutsuzlaştırılır.
    cost = float(np.squeeze(cost))
    return cost

def backward_propagation(cache,parameters,X,Y,activation):
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #Cost functionun model parametrelerine göre kısmi türevleri.

    dZ2 = A2 - Y#L'nin Z2ye göre kısmi türevi aradaki kısımlar atlanır.
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m

    #Kullanılmış aktivasyona göre backpropagation algoritması farklı işlemler yapar.
    dZ1 = np.dot(W2.T,dZ2) * (relu_der(Z1) if activation == "relu" else (1-np.power(A1,2)) )
    dW1 =np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m

    gradients = {
        "dW2": dW2,
        "db2": db2,
        "dW1" : dW1,
        "db1" : db1
    }
    return gradients

def update_parameters(gradients,parameters,learning_rate):
    dW1  = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #Parametreler güncellenir.
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    updated_parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return updated_parameters

def predict(X, parameters):
    A2,cache = forward_propagation(X,parameters,hidden_layer_activation_function)
    predictions = (A2 > 0.5)
    return predictions

def model(X,Y,hidden_units,hidden_unit_activation,learning_rate = 1.2,epoch = 10000):

    hidden_layer_activation_function = hidden_unit_activation

    #Parametreler oluşturulur.
    parameters = initialize_parameters(n_x, hidden_units, n_y )


    for i in range(epoch):

        yhat, cache  = forward_propagation(X,parameters,hidden_layer_activation_function)
        cost = compute_cost(yhat, Y)

        if i % 1000 == 0:
            print("epoch: " + str(i) + ", cost = " + str(cost))

        gradients = backward_propagation(cache,parameters, X,Y,hidden_layer_activation_function)

        parameters = update_parameters(gradients,parameters,learning_rate)

    #model parametre dictionarysi halinde geri döndürülür.
    return parameters

#Model oluşturulur hidden layer aktivasyon fonksiyonu seçilebilir.
parameters = model(X,Y,hidden_units = 50,hidden_unit_activation="tanh",epoch=10000)

#Decision boundary grafiği çizdirilir.
plot_decision_boundary(lambda x : predict(x.T,parameters),X,Y)
plt.title("Decision Boundary")
plt.show()







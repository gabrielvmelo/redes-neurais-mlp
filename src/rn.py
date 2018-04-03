import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net1 = network.Network([784, 30, 10])
net2 = network.Network([784, 15, 10])
net3 = network.Network([784, 30, 10, 10])
print "Starting net1 - (30 hidden layers)"
net1.SGD(training_data, 30, 10, 3.0, test_data=test_data)
print "Starting net2 - (15 hidden layers)"
net2.SGD(training_data, 30, 10, 3.0, test_data=test_data)
print "Starting net3 - (2 hidden layers (30, 10) and 4 outputs)"
net3.SGD(training_data, 30, 10, 3.0, test_data=test_data)

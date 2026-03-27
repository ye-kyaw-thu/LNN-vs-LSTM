import tonic.datasets as datasets
import tonic.transforms as transforms

# Define where to save the data (e.g., in a 'data' folder)
save_path = './data/N-MNIST'

# The dataset object handles the download automatically if files are not present
# Setting 'download=True' is generally handled within the library's logic.
# A transform might be needed to use the event-based data.
sensor_size = datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=1000)

trainset = datasets.NMNIST(save_path, train=True, transform=frame_transform)
testset = datasets.NMNIST(save_path, train=False, transform=frame_transform)


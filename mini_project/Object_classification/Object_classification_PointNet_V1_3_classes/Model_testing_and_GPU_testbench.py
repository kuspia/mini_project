"""
3D Perception on FPGA
      A modified LeNet model trained on point cloud capable of predicting 3 classes of objects "bottle","box","cup"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad


classes = ["bottle","box","cup"] # class of objects (that the model is capable of predicting
model_path = "C:\\Users\\grees\\OneDrive\\Desktop\\Intern_3D_perception_on_fpga_GreeshwarRS\\pointnet\\original_bot_cup_box_pointnet_3.pth"   # location of the .pth file
# text file containing the .pcd file location
test_text_file = "C:\\Users\\grees\\PycharmProjects\\intern\\dataset_new\\dataset_new_intern\\test_better.txt"

# for running in GPU to run on CPU change it  to "cpu"
# for GPU (for CPU comment this and un comment the line below it)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# the model architecture
class Transformer(nn.Module):

    def __init__(self, num_points=12000, K=3):
        # Call the super constructor
        super(Transformer, self).__init__()

        # Number of dimensions of the data
        self.K = K

        # Size of input
        self.N = num_points

        # Initialize identity matrix on the GPU (do this here so it only
        # happens once)
        self.identity = grad.Variable(
            torch.eye(self.K).double().view(-1))

        # First embedding block
        self.block1 = nn.Sequential(
            nn.Conv1d(K, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU())

        # Second embedding block
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU())

        # Third embedding block
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        # Multilayer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, K * K))

    # Take as input a B x K x N matrix of B batches of N points with K
    # dimensions
    def forward(self, x):
        # Compute the feature extractions
        # Output should ultimately be B x 1024 x N
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        self.N = x.shape[2]
        # Pool over the number of points
        # print("val")
        # Output should be B x 1024 x 1

        x = F.max_pool1d(x, self.N)

        x = x.view(-1, 1024)  # --> B x 1024 (after squeeze)
        # Run the pooled features through the multi-layer perceptron
        # Output should be B x K^2
        # print(x.shape,"before mlp")
        x = self.mlp(x)

        # Add identity matrix to transform
        # Output is still B x K^2 (broadcasting takes care of batch dimension)
        x += self.identity.to(device)

        # Reshape the output into B x K x K affine transformation matrices
        x = x.view(-1, self.K, self.K)

        return x

class PointNetBase(nn.Module):

    def __init__(self, num_points=12000, K=3):
        # Call the super constructor
        super(PointNetBase, self).__init__()
        # Size of input
        self.N = num_points
    # Input transformer for K-dimensional input
        # K should be 3 for XYZ coordinates, but can be larger if normals,
        # colors, etc are included
        self.input_transformer = Transformer(self.N, K)

        # Embedding transformer is always going to be 64 dimensional
        self.embedding_transformer = Transformer(self.N, 64)

        # Multilayer perceptrons with shared weights are implemented as
        # convolutions. This is because we are mapping from K inputs to 64
        # outputs, so we can just consider each of the 64 K-dim filters as
        # describing the weight matrix for each point dimension (X,Y,Z,...) to
        # each index of the 64 dimension embeddings
        self.mlp1 = nn.Sequential(
            nn.Conv1d(K, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU())


    # Take as input a B x K x N matrix of B batches of N points with K
    # dimensions
    def forward(self, x):

        # Number of points put into the network
        N = x.shape[2]

        # First compute the input data transform and transform the data
        # T1 is B x K x K and x is B x K x N, so output is B x K x N
        T1 = self.input_transformer(x)
        x = torch.bmm(T1, x)

        # Run the transformed inputs through the first embedding MLP
        # Output is B x 64 x N
        x = self.mlp1(x)

        # Transform the embeddings. This gives us the "local embedding"
        # referred to in the paper/slides
        # T2 is B x 64 x 64 and x is B x 64 x N, so output is B x 64 x N
        T2 = self.embedding_transformer(x)
        local_embedding = torch.bmm(T2, x)

        # Further embed the "local embeddings"
        # Output is B x 1024 x N
        global_feature = self.mlp2(local_embedding)

        # Pool over the number of points. This results in the "global feature"
        # referred to in the paper/slides
        # Output should be B x 1024 x 1 --> B x 1024 (after squeeze)
        global_feature = F.max_pool1d(global_feature, N).squeeze(2)

        return global_feature, local_embedding, T2


class PointNetClassifier(nn.Module):

    def __init__(self, num_points=1000, K=3):
        # Call the super constructor
        super(PointNetClassifier, self).__init__()

        self.N = num_points
        # Local and global feature extractor for PointNet

        self.base = PointNetBase(self.N, K)

        # Classifier for ShapeNet
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 3))  # 256,no. of classes to classify

    # Take as input a B x K x N matrix of B batches of N points with K
    # dimensions
    def forward(self, x):
        # Only need to keep the global feature descriptors for classification
        # Output should be B x 1024
        x, _, T2 = self.base(x)

        # Returns a B x 3
        # print(x.shape,"classifier")
        return self.classifier(x), T2

"""function for changing the range of coordinates
why ???
(as the points where in the range of eg. -0.33 - 0.66 hence there was little difference between each points.
hence model didnt give better results.After changing its range to 0-100 it gave better results.)
"""
def scale(val):

  input_start = min(val) # The lowest number of the range input.
  input_end = max(val) #  The largest number of the range input.
  output_start = 0# The lowest number of the range output.
  output_end = 100# The largest number of the range output.
  # print(input_end,input_start)
  val_out = []
  for i in val:
    output = output_start + ((output_end - output_start) / (input_end - input_start)) * (i - input_start)
    val_out.append(output)
  # if no. of points < 12000 append 0,0,0 to list
  if(len(val_out) < 12000):
    length = len(val_out)
    for i in range(12000 - length):
        val_out.append(0)

  return val_out
#function for reading the points from .pcd file and loading the coordinates in a list form
def load_points(file_name):

  id = None
  vertex_list = []  # list that contains the points
  # opening the .pcd file
  with open(file_name, "r") as p:  # reading .pcd file as text file and so skiping the first 11 unwanted line in .pcd file
    for i in range(0, 11):
      see = p.readline()
      if (i == 6):
        id = see.split(' ')[1]
    for i in range(0, int(id)):
      see = p.readline()
      a, b, c, d = see.split(' ')
      vertex_list.append([float(a),float(b),float(c)]) #appending the points

  no_of_points_in_pointcloud = len(vertex_list)   # total no. of points in that point cloud

  # changing (N,3) to (3,N) -- N- total no. of points in pointcloud
  vertex_list = [[vertex_list[j][i] for j in range(len(vertex_list))] for i in range(len(vertex_list[0]))]


  vertex_list_out = []   # list for appending the scaled points
  # for changing the range(scale function)
  for i in vertex_list:
    vertex_list_out.append(scale(i))   # changing the range for the coordinates
  #print(vertex_list_out)
  return vertex_list_out,no_of_points_in_pointcloud # array of points after scaling , total no. of points in that point cloud



# the code below is for loading the trained model
def load_checkpoint(filepath):


  checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

  model = checkpoint['model']
  model.load_state_dict(checkpoint['state_dict'])
  for parameter in model.parameters():
    parameter.requires_grad = False

  model.eval()

  return model

# loading the model
classifier = load_checkpoint(model_path).to(device)
print(classifier)

# testing model
def print_preds_test(file_n):

    # reading the .txt file containing the location of .pcd files for testing and its lables
    f = open(file_n, "r")
    # variables for calculating accuracy
    wrong_preds = 0
    correct_preds = 0
    wrong_preds_bottle = 0
    correct_preds_bottle = 0
    wrong_preds_cup = 0
    correct_preds_cup = 0
    wrong_preds_box = 0
    correct_preds_box = 0
    # running through each item in the .txt file
    for filename in f:
        filename,label = filename.split(" ")
        label = int(label)
        if(label == 0):
            # bottle
            filename = "C:\\Users\\grees\\PycharmProjects\\intern\\dataset_new\\dataset_new_intern\\bottle\\"+filename #+ "d"
        elif(label == 1):
            # box
            filename = "C:\\Users\\grees\\PycharmProjects\\intern\\dataset_new\\dataset_new_intern\\box\\" + filename# + "d"
        else:
            # cup
            filename = "C:\\Users\\grees\\PycharmProjects\\intern\\dataset_new\\dataset_new_intern\\cup\\" + filename# + "d"

        points, num_points = load_points(filename.rstrip("\n"))  # getting the list of points and total no. of points present in the point cloud
        # points = torch.from_numpy(points)
        points = torch.FloatTensor(points).to(device)  # converting the list of points to tensor
        points = points.unsqueeze(0)  # (3,N) to (1,3,N)

        # Forward pass
        outputs, _ = classifier(points)
        # prediction
        _, preds = torch.max(outputs, 1)  # 1 -- because it also returns the index of max value

        # Calculating accuracy for individual classes of objects
        if(label == 0):
            if (preds.data !=label):
                wrong_preds_bottle += 1
            else:
                correct_preds_bottle += 1
        if (label == 1):
            if (preds.data != label):
                wrong_preds_box+= 1
            else:
                correct_preds_box += 1
        if (label == 2):
            if (preds.data != label):
                wrong_preds_cup += 1
            else:
                correct_preds_cup += 1

        # Calculating overall accuracy
        if(preds.data != label):
            wrong_preds += 1
        else:
            correct_preds +=1
    print("percentage accuracy bottle: ", (correct_preds_bottle / (correct_preds_bottle + wrong_preds_bottle)) * 100)
    print("percentage accuracy box: ", (correct_preds_box / (correct_preds_box + wrong_preds_box)) * 100)
    print("percentage accuracy cup: ", (correct_preds_cup / (correct_preds_cup + wrong_preds_cup)) * 100)
    print("percentage accuracy: ",(correct_preds/(correct_preds+wrong_preds))*100)



import time
t = time.time()
# for testing the model
print_preds_test(test_text_file)

'''For GPU test bench uncomment the below lines'''
# uncomment this for GPU and change device to "cuda:0"
# print("Time taken to run 1573 pcd for classification (PointNet) using GPU = ",time.time()-t)

# uncomment this for CPU and change device to "cpu"
# print("Time taken to run 1573 pcd for classification (PointNet) using CPU = ",time.time()-t)

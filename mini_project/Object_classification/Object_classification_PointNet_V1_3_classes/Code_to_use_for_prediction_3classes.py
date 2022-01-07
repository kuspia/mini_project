"""
3D Perception on FPGA
      PointNet model trained on point cloud capable of predicting 3 classes of objects "bottle","box","cup"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad
import numpy as np # numpy and open3d is used only for visualization purpose
import open3d as o3d
from open3d import *

classes = ["bottle","box","cup"] # class of objects (that the model is capable of predicting
model_path = "C:\\Users\\grees\\OneDrive\\Desktop\\Intern_3D_perception_on_fpga_GreeshwarRS\\pointnet\\original_bot_cup_box_pointnet_3.pth"   # location of the .pth file
cluster_list = "D:\\plc_try1\\cluster_list_k.txt" # contains the location where the pointcloud clusters to predict is present
point_cloud_loaction = "D:\\plc_try1\\pcdss\\3bottle_light_2.pcd"  # location where the point cloud to predict is present
# font style- for visualizing prediction
font_style = 'C:\\Users\\grees\\PycharmProjects\\intern\\DejaVu Sans Mono for Powerline.ttf'

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

    def __init__(self, num_points=12000, K=3):
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
  with open(file_name, "r") as p:
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

  # finding min and max in all x,y,z of points for finding bounding box
  min_x = min(vertex_list[0])
  max_x = max(vertex_list[0])
  min_y = min(vertex_list[1])
  max_y = max(vertex_list[1])
  min_z = min(vertex_list[2])
  max_z = max(vertex_list[2])
  #print(min_x, min_y, min_z, max_x, max_y, max_z )
  bounding_box = [min_x, min_y, min_z, max_x, max_y, max_z]


  vertex_list_out = []   # list for appending the scaled points
  # for changing the range(scale function)
  for i in vertex_list:
    vertex_list_out.append(scale(i))   # changing the range for the coordinates
  #print(vertex_list_out)
  return vertex_list_out,no_of_points_in_pointcloud,bounding_box  # array of points after scaling , total no. of points in that point cloud



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

# for rendering the text to be shown above the bounding box (Visualization purpose only)
def text_3d(text, pos, direction=None, degree=0.0, density=10, font=font_style, font_size=10):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0, 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(0,255,0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=90)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, np.pi))
    pcd.rotate(R)
    return pcd


# generates bounding box (Visualization purpose only)
def generating_boundingbox(points):
    # for bounding box

    lines = [[0, 1], [0, 2], [1, 3], [2, 3],
             [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[0, 1, 0] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

# this function gets the extracted cluster location and predicts it, prints the prediction and also returns the bounding box for each clusters (for visualization)
def print_preds(file_n):
    obj_list = []
    point_cloud_cluster_list = [] # for visualization
    f = open(file_n, "r")
    for filename in f:
        #for visualization only------------------------------------------------------------------------------------
        clus_pcd = o3d.io.read_point_cloud(filename.rstrip("\n"), format="pcd")  # this will create an open3d object
        point_cloud_cluster_list.append(clus_pcd) # appending the clusters for visualization
        # --------------------------------------------------------------------------------------------------------
        points, num_points,bounding_box = load_points(filename.rstrip("\n"))  # getting the list of points and total no. of points present in the point cloud
        points = torch.FloatTensor(points).to(device)  # converting the list of points to tensor
        points = points.unsqueeze(0)  # (3,N) to (1,3,N)


        # Forward pass
        outputs, _ = classifier(points)
        # prediction
        _, preds = torch.max(outputs, 1)  # 1 -- because it also returns the index of max value
        # Softmax function for getting percent accuracy
        out_per = nn.Softmax(dim=0)
        percentage = torch.round(out_per(outputs.squeeze(0))*100)
        # print(outputs.squeeze(0))
        print(percentage)
        pred_flag = False # to decide at least one point cloud is being predicted above the threshold
        found = ""

        for i in range(len(percentage.tolist())):
            print("predicted as ",percentage.tolist()[i],"% as ",classes[i])

        if(classes[preds.data] == "bottle"):
          if(percentage.tolist()[preds.data] > 20):
                found = classes[preds.data] + " "+ str(percentage.tolist()[preds.data])+" % "
                pred_flag = True
        elif (classes[preds.data] == "box"):
            if (percentage.tolist()[preds.data] > 20):
                found = classes[preds.data] + " " + str(percentage.tolist()[preds.data]) + " % "
                pred_flag = True
        elif (classes[preds.data] == "cup"):
            if (percentage.tolist()[preds.data] > 20):
                found = classes[preds.data] + " " + str(percentage.tolist()[preds.data]) + " % "
                pred_flag = True



        if(pred_flag):
            print(found)
            min_x, min_y, min_z, max_x, max_y, max_z =  bounding_box
            # for printing what object is that
            obj_list.append(text_3d(found, pos=[max_x, min_y, min_z], font_size=20, density=1))
            points = [[min_x, min_y, min_z], [max_x, min_y, min_z], [min_x, max_y, min_z], [max_x, max_y, min_z],
                      [min_x, min_y, max_z], [max_x, min_y, max_z], [min_x, max_y, max_z], [max_x, max_y, max_z]]
            # generating and appending bounding box
            obj_list.append(generating_boundingbox(points))

    # o3d.visualization.draw_geometries(point_cloud_cluster_list)  # point cloud  with bounding box
    return obj_list , point_cloud_cluster_list # bounding boxes for the clusters and the clusters for visualizing them

# passing the cluster list to print_preds function it returns bounding boxes for the clusters and the clusters for visualizing them
point_cloud_list,point_cloud_clus_list = print_preds(cluster_list)
# print(point_cloud_list)

pcd = o3d.io.read_point_cloud(point_cloud_loaction,format="pcd") # The original point cloud
point_cloud_list.append(pcd)  # appending it to visualize


# visualizing the pointclouds with bounding boxes
o3d.visualization.draw_geometries([pcd])   # original point cloud
o3d.visualization.draw_geometries(point_cloud_clus_list)   # original point cloud
o3d.visualization.draw_geometries(point_cloud_list)  # point cloud  with bounding box



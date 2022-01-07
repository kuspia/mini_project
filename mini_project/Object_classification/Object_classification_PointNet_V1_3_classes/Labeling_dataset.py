# code for labeling the point clouds

import random
import os

def get_filepaths(directory,label):
    """
    This function will generate the file names in a directory
   
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        if(label == "0"):
            # bottle
            i = 1
            j = 1
            while(i <= 17):
                print(i,j)

                for filename in files:
                    # Join the two strings in order to form the full filepath.
                    filepath = os.path.join(root, filename)
                    if("bottle_"+str(i) in filepath):
                        file_paths.append(str(filepath.lstrip(directory+"\\")+" "+label))  # Add it to the list.
                        j+=1
                    if(j == 13):
                        i+=1
                        j = 1
                        break
            print("done 0")

        elif (label == "1"):
            # box
            i = 1
            j = 1
            while (i <= 6):
                print(i, j)
                for filename in files:
                    # Join the two strings in order to form the full filepath.
                    filepath = os.path.join(root, filename)
                    if ("box_" + str(i) in filepath):
                        file_paths.append(str(filepath.lstrip(directory + "\\") + " " + label))  # Add it to the list.
                        j += 1
                    if (j == 35):
                        i += 1
                        j = 1
                        break
            print("done 1")
        else:
            # cup
            i = 1
            j = 1
            while (i <= 4):
                print(i, j)
                for filename in files:
                    # Join the two strings in order to form the full filepath.
                    filepath = os.path.join(root, filename)
                    if ("cup_" + str(i) in filepath):
                        file_paths.append(str(filepath.lstrip(directory + "\\") + " " + label))  # Add it to the list.
                        j += 1
                    if (j == 51):
                        i += 1
                        j = 1
                        break
            print("done 2")

    return file_paths  # Self-explanatory.


bottle = get_filepaths("dataset_new\\dataset_new_intern\\bottle","0")  # ggg corresponds to the folder containing bottle dataset
box = get_filepaths("dataset_new\\dataset_new_intern\\box","1")    # uuu corresponds to the folder containing box dataset
cup = get_filepaths("dataset_new\\dataset_new_intern\\cup","2")    # ooo corresponds to the folder containing cup dataset
print(len(bottle),len(box),len(cup))   # length of each dataset
print(cup)
random.shuffle(bottle)  # shuffling the items in the list
random.shuffle(box)
random.shuffle(cup)
full = bottle+cup+box
test = bottle[0:41]+cup[0:41]+box[0:41]  # list of location of pointclouds for test dataset
train = bottle[41:]+cup[41:]+box[41:]    # list of location of pointclouds for train dataset

print(train)




print(len(train))
print(len(test))


# saving the .txt file containing the pointcloud file names and their corresponding label
textfile = open("dataset_new\\dataset_new_intern\\train_better.txt", "w")
for element in train:
    textfile.write(element + "\n")
textfile.close()

textfile = open("dataset_new\\dataset_new_intern\\test_better.txt", "w")
for element in test:
    textfile.write(element + "\n")
textfile.close()

# without test train split
textfile = open("dataset_new\\dataset_new_intern\\full.txt", "w")
for element in full:
    textfile.write(element + "\n")
textfile.close()


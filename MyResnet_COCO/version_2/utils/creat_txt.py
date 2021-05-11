"""Creat a txt file for the coco dataset
absolute_path \t label
"""
import os
img_path  = r"data\train\n01440764"
img_path2 = r"data\train\n04428191"
label_path= r"data\train\train.txt"
# img_list = os.listdir(img_path)
with open(label_path, "a") as f:
    for img_name in os.listdir(img_path):
        f.write(os.path.join( r"E:\data\data\train\n01440764", img_name) + "\t" + "0" + "\n")
    for img_name in os.listdir(img_path2):
        f.write(os.path.join( r"E:\data\data\train\n04428191", img_name) + "\t" + "1" + "\n")
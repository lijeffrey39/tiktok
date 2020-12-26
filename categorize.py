import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
from scipy.misc import imresize as imresize
import cv2
from PIL import Image
import glob
import csv


def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'whole_wideresnet18_places365_python36.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
    useGPU = 0
    if useGPU == 1:
        model = torch.load(model_file)
    else:
        model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()

# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0


def classifyImage(frame, frameNum):
    img = Image.fromarray(frame, 'RGB')
    input_img = V(tf(img).unsqueeze(0), volatile=True)

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output the IO prediction
    inOut = np.mean(labels_IO[idx[:10].numpy()]) # vote for the indoor or outdoor
    categories = []
    for i in range(0, 5):
        categories.append([probs[i], classes[idx[i]]])

    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[(frameNum * 2) + 1])
    idx_a = np.argsort(responses_attribute)
    attributes = [labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]
    return (inOut, categories, attributes)


def categorize(filePath):
    vid = cv2.VideoCapture(filePath)
    frameCount = 0
    totalInOut = 0
    categoryDict = {}
    attributesDict = {}
    frameNum = 0

    while(True):
        ret, frame = vid.read()
        if ret:
            frameCount += 1
            if (frameCount % 15 != 0):
                continue
            (inOut, categories, attributes) = classifyImage(frame, frameNum)
            totalInOut += inOut
            for cat in categories:
                category = cat[1]
                if (category not in categoryDict):
                    categoryDict[category] = cat[0]
                else:
                    categoryDict[category] += cat[0]

            for att in attributes:
                if (att not in attributesDict):
                    attributesDict[att] = 1
                else:
                    attributesDict[att] += 1

            frameNum += 1
        else:
            break

    sortCat = sorted(categoryDict.items(), key=lambda x: x[1], reverse=True)
    sortAtt = sorted(attributesDict.items(), key=lambda x: x[1], reverse=True)

    sortCat = list(map(lambda x: (x[0], round(x[1] / frameNum, 3)), sortCat))
    sortAtt = list(map(lambda x: (x[0], round(x[1] / frameNum, 3)), sortAtt))

    vid.release()
    cv2.destroyAllWindows()
    return (float(totalInOut / frameNum), sortCat[:5], sortAtt[:5])


def parseVideos(filePaths):
    allFoundIds = []
    with open("test.csv") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data_read = [row for row in reader]
        for r in data_read:
            allFoundIds.append(r[0])
    allFoundIds = set(allFoundIds)

    for filePath in files:
        vidId = filePath.split("/")[-1][:-4]
        if (vidId in allFoundIds):
            continue

        (inOut, sortCat, sortAtt) = categorize(filePath)
        print(inOut, sortCat, sortAtt)

        with open("test.csv", "a") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow([vidId, inOut, sortCat, sortAtt])

files = glob.glob("/Users/jeffreyli/Desktop/tiktok-scraper/jeffreyli6/*.mp4")
# print(len(files))
# parseVideos(files)

from ast import literal_eval

def findVideos():
    d = {}
    with open("videos.csv") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data_read = [row for row in reader]
        for r in data_read:
            arr = literal_eval(r[2])
            print(arr[0])
            vidId = r[0]

            for cat in arr:
                catName = cat[0]
                res = (vidId, cat[1])
                if (catName not in d):
                    d[catName] = [res]
                else:
                    d[catName].append(res)

    crevasse = d["crevasse"]
    crevasse.sort(key = lambda x: x[1])
    print(crevasse)
    for c in crevasse:
        print(c)

findVideos()
# COMP594
Individual Project : COMP594


# Background on Architecture

The architecture chosen for the initial product is a variation of encoder-decoder network called U-Net. 

U-Net was used first in medical image segmentation to classify cells' boundaries. 

See the following [arVix paper](https://arxiv.org/abs/1505.04597) 

Since then, U-Net has been adopted for image segmentation, due to the reasonable effectiveness.


# File Description

DrawingWithTensors: The main file for producing a satellite image as well as its corresponding ground-truth tensor.
The initial decision to simply save the tensor to disk was to save computation later. It would be very easily to load an already created tensor later on, instead of having to recompute or transform other memory.

The notebook is used for experiments while the .py version is used to call the actual functions.


CreateSyntheticDataset: The main file to produce the training set. This will DrawingWithTensors.py, and have options such as dimension size, how many files, etc. We note a generation to separate which type of images we are generating.


UNetExperiment: The main file to train the network. There are various parameters to set such as the batch size, epoch count, dimension size, which generation to use, etc. A noteable point of interest is that the outputs of the network are not transformed via sigmoid. The loss function should  automatically convert the inputs with sigmoid before comparison to a target tensor.

unet_models.py: The main file where the initial network definition resides. 
See the following GitHub work for an [demonstration](https://github.com/ternaus/TernausNet). 

# Notable Functions:

Load an image with matplotlib and interactivity enabled:
```
%matplotlib notebook

img = Image.open("270_5.35_b.png")
plt.imshow(img)
```

Evaluate performance on a single, new image:

```
inputs[0] = transforms.ToTensor()(img)
outputs = model(inputs)

DrawingWithTensors.showInferenceOnImage(img, torch.sigmoid(outputs[0]), class_label, threshold, classMap)

```
# Train an object detector for people and mixer trucks using Yolov5

This guide explains how to train your **Chronsite object detector**  with YOLOv5. Our implementation relies on the official Yolov5 implementation, which you can find at https://github.com/ultralytics/yolov5. 

# Default settings and methodology
## Preprocessing
We padded the images to square them and we resized them so that they are all of the same size. The chosen size was the maximum size of the available images (1920px width), so that no information is lost. This step was performed with the `Albumentations` package (https://github.com/albumentations-team/albumentations). We also implemented more in-depth augmentation strategies, still using `Albumentations`, with the following functions:
 - HorizontalFlip 
 - Rotate
 - RandomBrightness
 - RandomSnow
 - RandomFog

However, the augmentation technique provide marginal gains (1-2% mAP on our validation set) and tripled the training time, so we didn't retain it for the final solution.

## Training
We trained our model for 30 epochs, using the large version of YOLOv5 (you can chose the size of the architecture as a parameter in our model, ranging from small to extra-large). In order for the computer not to run out of memory (16Go RAM available on our machines), we used a batch size of 2. This settings takes approximately 4 hours to run on a P100 GPU. 


# Files and use
The `yolo` folder is composed of a script `yolo_utils.py` that contains all the necessary functions to train and detect objects on the images. It also contains a Jupyter notebook `eleven-yolo.ipynb` that is used to launch the script (which would also be possible from the command line). The weights of the trained model are available under the file `yolo-weights-best.pt`. 

In order to train the model, you just have to run the 2 first cells of the notebook, which will take care of : 

 - Package imports
 - Global variables definition (including paths to the data and the config options)
 - Cloning the yolo git repo
 - Creating the relevant arborescence for the data files
 - Training the model itself and saving the weights 

Then, you will be able to use the model to detect the objects over new images, by running the third cell containing the `detect.py` command. The output will be saved to the relevant folders as configured above. 

# Detection Example
This is the kind of visualisation you can get by detecting the objects over an unseen image:

![](https://files.slack.com/files-tmb/T019L75A5AA-F01CPF2S9H6-ed780a9ca6/image_720.png)

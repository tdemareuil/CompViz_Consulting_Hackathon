# compviz-consulting-hackathon

This repository holds files related to a 1-week hackathon organized by Eleven Consulting (Paris) in September 2020.

## Project

As a group of students from my MSc Data Science at Ecole Polytechnique, we conducted a mock consulting mission for Vinci Construction. The mission involved :
* **modeling**: we implemented a **YOLOv5 object detection model (PyTorch)** and a **UNet image segmentation model (Keras)** to detect workers, mixer trucks, concrete pumps and vertical formworks in CCTV images of various constrcution sites.
* **analysis**: we analyzed the images and detected objects in order to provide actionable KPIs to our client (task time, nb of workers involved, heatmap of worker's movement, etc.). Goal is to increase productivity on construction sites.
* **presentation**: we finally presented strategic recommendations to Eleven Consulting consultants. Deliverables include our final ppt presentation, the model and a small **webapp** built with Flask, which allows to interact with the segmentation model (see`modeling/UNet/app` folder).

## Demo

See video demo [here](/modeling/app_demo.mov).

![](/modeling/app_demo.mov)

# Analytics pipeline

You'll find in this directory *3 notebooks* corresponding to different types of analyses performent on construction site CCTV images:
* `analytics_image_to_csv.ipynb`: outputs a csv file with the number of workers performing a concreting task in each frame (one row per frame and per task)/
* `analytics_nb_labels.ipynb`: outputs dictionnary with statistics about the nb of occurences of each object in the images
* `heatmap.ipynb`: outputs a heatmap correcponding to the *movements of workers* (i.e. the most active areas) in the observed construction sites.

## For all files, make sure you specify the correct paths before running the cells:
* images_analytics = PATH_TO_ANALYTICS_IMG
* labels_analytics_poly = PATH_TO_ANALYTICS_JSON_POLY #shape: bitmap + origin point
* labels_analytics_people = PATH_TO_ANALYTICS_JSON_PEOPLE #shape: bounding box left upper and right lower point.

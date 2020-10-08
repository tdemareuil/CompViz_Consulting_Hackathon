# CREATE APP ENV WITH: conda env create -f environment.yml -n app_env
 
# app management imports
import flask
import requests
import glob
import sys
import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# image processing imports
import numpy as np
import skimage
from skimage import measure, io, img_as_ubyte
from skimage.color import rgb2gray
import shapely
from shapely import ops
from shapely.geometry import Polygon
from scipy.spatial import Delaunay
from descartes import PolygonPatch
import fiona
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import patches
plt.switch_backend('Agg')
import math
import random
import json
import pandas as pd
import colorsys

# modeling imports
import tensorflow as tf
import segmentation_models as sm
import albumentations as A


################################################################
#  Paths to dependencies
################################################################

# Root project directory
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)

# Path to trained weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, 'models', "best_weights.h5")
MODEL_PATH = os.path.join(ROOT_DIR, 'models', "best_model_0410.h5")
MODEL_NAME = os.path.basename(MODEL_PATH)


################################################################
#  Helper functions
################################################################

def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def threshold_image(image, threshold=128):
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]
    thresholded_image = image.copy()
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            thresholded_image[y, x] = 255 if image[y, x] >= threshold else 0     
    # return the thresholded image
    return thresholded_image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def alpha_shape(polygon, nb_points=500, alpha=500, plot=False):
    """
    Computes the alpha shape (concave hull smoothing) of a set of shapely Points, using
    a Delaunay triangulation algorithm.

    The idea is to build triangles between all points (i.e. computing a convex hull), and 
    to remove triangle lines longer than 1/alpha.
    Therefore, the higher the alpha, the higher the precision (you eliminate smaller 
    triangles and get closer to the original polygon shape). The nb of points inside of 
    the polygon is important as well: the higher the nb of points, the smaller the triangles, 
    and the higher alpha will have to be to augment the precision of the concave hull.

    => All in all, the value of alpha and the nb of points should depend on the 
    size of the original polygon! [TODO: automate the choice of nb_points and alpha]
    
    Arguments:
        nb_points: nb of points to generate inside of polygon
        alpha: alpha value to influence the gooeyness of the border
        plot: set to True to plot the Delaunay triangles and alpha shape polygon
    Returns:
        concave_hull_poly: new shapely Polygon object
        concave_hull_coords: list of point coordinates (as np array)
    """
    
    # generate points inside polygon for alpha shape / Delaunay triangulation
    points = generate_points_in_polygon(nb_points, polygon)
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    # Delaunay triangulation
    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - 
        triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - 
        triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - 
        triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)] # filter too large triangles

    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    
    # Recreate new edge point coordinates and new polygon (concave hull)
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis=0).tolist()
    m = shapely.geometry.MultiLineString(edge_points)
    triangles = list(shapely.ops.polygonize(m))
    concave_hull_poly = shapely.ops.cascaded_union(triangles)
    
    # get rounded coordinates of concave hull polygon
    if concave_hull_poly.geom_type == 'MultiPolygon':
        concave_hull_poly = concave_hull_poly[0]
    x,y = concave_hull_poly.exterior.coords.xy
    x = [round(i,0) for i in x]
    y = [round(i,0) for i in y]
    concave_hull_coords = list(zip(x,y))
    concave_hull_coords = list(dict.fromkeys(concave_hull_coords))

    # plot if requested
    if plot:
        lines = LineCollection(edge_points)
        plt.figure(figsize=(8,8))
        plt.title('Alpha={0} Delaunay triangulation'.format(alpha))
        plt.gca().add_collection(lines)
        delaunay_points = np.array([point.coords[0] for point in points])
        x_rand = [poly_r.coords[0][0] for poly_r in points]
        y_rand = [poly_r.coords[0][1] for poly_r in points]
        plt.plot(delaunay_points[:,0], delaunay_points[:,1], 'o', color='#f16824')
        _ = plot_polygon(concave_hull_poly)
        _ = plt.plot(x_rand,y_rand,'o', color='#f16824')

    return concave_hull_poly, concave_hull_coords

def generate_points_in_polygon(number, polygon):
    """
    Helper fuction for alpha_shape.
    Generates a chosen nb of random points inside of a random polygon.
    Returns: a list of shapely Point objects.
    """
    list_of_points = []
    minx, miny, maxx, maxy = polygon.bounds
    counter = 0
    while counter < number:
        pnt = shapely.geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            list_of_points.append(pnt)
            counter += 1
    return list_of_points

def plot_polygon(polygon):
    """
    Helper fuction for alpha_shape.
    Plots a shapely polygon in a matplotlib figure.
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    margin = 0

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig

json_template = {"categories":[{"Supercategory": "none",
                                  "name": "People",
                                  "id": 1},
                                 {"Supercategory": "none",
                                  "name": "Mixer_truck",
                                  "id": 2},
                                 {"Supercategory": "none",
                                  "name": "Vertical_formwork",
                                  "id": 3},
                                 {"Supercategory": "none",
                                  "name": "Concrete_pump_hose",
                                  "id": 4}],
                  "info": {"description": "Chronsite test dataset",
                           "year": 2020},
                  "licenses": [],
                  "images": [{"file_name": "xx.jpg",
                              "id": 33,
                              "height": 1024,
                              "width": 1280}],
                  'annotations': [{'area': 5040,
                                   'bbox': [980, 696, 60, 84],
                                   'category_id': 3,
                                   'id': 630,
                                   'image_id': 33,
                                   'iscrowd': 0,
                                   'segmentation': [[990, 700, 1007, 696]]
                                  }]}


################################################################
#  App definition
################################################################

app = flask.Flask(__name__)

# app.config['APP_ROOT'] = os.path.dirname(os.path.abspath(__file__))
app.config['APP_ROOT'] = app.root_path
app.config['APP_STATIC'] = os.path.join(app.config['APP_ROOT'], 'static')
app.config['INFERENCE_DIR'] = os.path.join(app.config['APP_ROOT'], 'inference')
app.config['MAX_CONTENT_LENGTH'] = 8 * 2048 * 2048
app.config['SECRET_KEY'] = "5c7b2ae79e54a6d5" # generated with: os.urandom(8).hex()
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # to avoid caching files in browser


# Define the home page
@app.route('/', methods=["POST", "GET"])
def home():
    return flask.render_template("index.html")


# Define the success page
@app.route('/success', methods=["POST", "GET"])
def predict():

    # Delete files from previous runs in inference folder
    files = glob.glob(app.config['INFERENCE_DIR']+'/*.*')
    for f in files:
        os.remove(f)
        
    # Option 1: Get user-uploaded image
    if flask.request.files.get('file'):
        image = flask.request.files['file']
        image.save(os.path.join(app.config['INFERENCE_DIR'], 'upload.png'))
        # If we had chosen to save the file under its original filename, for
        # security we would have run: filename = secure_filename(image.filename)
    
    # Option 2: Get chosen default image
    elif flask.request.form.get('test'):
        image = os.path.join(app.config['APP_STATIC'], flask.request.form.get('test'))

    # Pre-processing: read the image, pad/resize it and apply EfficientNet preprocessing
    image_id = os.path.basename(image)
    image = skimage.io.imread(image, plugin='matplotlib')
    if image.shape[-1] == 4: # remove alpha channel
        image = image[..., :3]
    HEIGHT = image.shape[0]
    WIDTH = image.shape[1]
    preprocess_input = sm.get_preprocessing(BACKBONE)
    resize_preprocess_transform = A.Compose([
        #A.PadIfNeeded(768, 1024, p=1.0, border_mode=0),
        A.Resize(height=768, width=1024, p=1.0),
        A.Lambda(image=preprocess_input)])
    resized_preprocessed = resize_preprocess_transform(image=image)
    image = resized_preprocessed["image"]
    print("--> Image loaded - going to detection...")

    # Run detection, return to original size and save predicted mask
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)
    pr_mask = pr_mask.squeeze()
    image = denormalize(image).squeeze()
    inverse_resize_transform = A.Compose([
        #A.PadIfNeeded(HEIGHT, WIDTH, p=1.0, border_mode=0),
        A.Resize(height=HEIGHT, width=WIDTH, p=1.0)])
    inverse_resized = inverse_resize_transform(image=image, mask=pr_mask)
    image, pr_mask = inverse_resized["image"], inverse_resized["mask"]
    skimage.io.imsave(os.path.join(app.config['INFERENCE_DIR'], "pr_mask.png"), pr_mask)
    print("--> Detection successful - now saving results for display...")

    # Post-processing: delineate contours to get polygons, delete smaller ones, save as JSON
    
    # T = 204
    # formworks = threshold_image(img_as_ubyte(pr_mask[...,0]), threshold = T)
    # pumps = threshold_image(img_as_ubyte(pr_mask[...,1]), threshold = T)
    # contours_formworks = measure.find_contours(formworks, 1)
    # contours_pumps = measure.find_contours(pumps, 1)
    # contours_formworks = [x.round(0).tolist() for x in np.array(contours_formworks)]
    # contours_pumps = [x.round(0).tolist() for x in np.array(contours_pumps)]

    # pr_polygons = json_template
    # try:
    #     for idx, contour in enumerate(contours_formworks.extend(contours_pumps)):
    #         poly = Polygon(contour)
    #         if poly.area < 300: # very few ground truth formworks & pumps are <300px area
    #             continue
    #         _, concave_hull_coords = alpha_shape(poly, alpha=0.01)
    #         details = 
    #         if contour is in contours_formworks:
    #            category_id
    #         pr_polygons['annotations'].append(details)
    #         pr_polygons[f'formwork_{idx+1}'] = concave_hull_coords
    # except:
    #     print('Post-processing failed - review alpha_shape function.')
    
    # with open(os.path.join(app.config['INFERENCE_DIR'],'pr_polygons.json'), 'w') as f:
    #     json.dump(pr_polygons, f)

    grayscale = rgb2gray(pr_mask)
    contours = measure.find_contours(grayscale, 0.1)
    pr_polygons = {}
    try:
        for idx, contour in enumerate(contours):
            poly = Polygon(contour)
            if poly.area < 300: # very few ground truth formworks & pumps are <300px area
                continue
            _, concave_hull_coords = alpha_shape(poly, alpha=0.01)
            pr_polygons[f'polygon_{idx}'] = concave_hull_coords
            #pr_polygons[f'polygon_{idx}'] = contour
    except:
        print('Post-processing failed - review alpha_shape function.')
    with open(os.path.join(app.config['INFERENCE_DIR'],'pr_polygons.json'), 'w') as f:
        json.dump(pr_polygons, f)

    # Overlay detected polygons on original image and save result
    fig, ax = plt.subplots()
    ax.imshow(image)
    colors = random_colors(len(contours))
    for idx, contour in enumerate(contours):
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        color = colors[idx]
        p = patches.Polygon(list(zip(contour[:,1],contour[:,0])), 
                            facecolor=color, edgecolor=color, alpha=0.5)
        ax.add_patch(p)
    ax.axis('off')
    plt.savefig(os.path.join(app.config['INFERENCE_DIR'], "result.png"), 
                bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # Save detection summary
    data = {}
    data["Successful detection"] = True
    try:
        #data["Number of detected formworks"] = len(contours_formworks)
        data["Number of detected formworks"] = len(pr_polygons)
        data["Number of detected pumps"] = 0
    except:
        data["Number of detected objects"] = len(contours)
    data_df = pd.DataFrame(data, index=[0]).T
    data_df.to_excel(os.path.join(app.config['INFERENCE_DIR'],
                                        "detection_summary.xlsx"), header=False)
    print("--> Saving done - showing success page.\n")

    return flask.render_template('success.html', data=data)


# This last route is just to make sure flask will accept displaying images 
# from the inference directory
@app.route('/inference/<path:filename>')
def display_file(filename):
    return flask.send_from_directory(app.config['INFERENCE_DIR'],
                                     filename, as_attachment=True)


################################################################
#  App launcher
################################################################

# Finally, the script that builds the model and launches the app
if __name__ == '__main__':
    # App will run only if called as main, i.e. not if imported in another code
    print("\n...Loading model and starting server...\n",
          "       ...please wait until server has fully started...\n")
     
    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Build the model architecture and load trained weights
    BACKBONE = 'efficientnetb3'
    CLASSES = ['formworks', 'pumps']
    preprocess_input = sm.get_preprocessing(BACKBONE)
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    #model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, weights=WEIGHTS_PATH)
    #model.load_weights(WEIGHTS_PATH, by_name=True)

    print("Loaded model:", MODEL_NAME)
    print("Model runs on http://0.0.0.0:5000/\n")

    # Run app
    app.run(host='0.0.0.0', port='5000', debug=True)

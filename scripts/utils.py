import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import cv2
import torch.functional as F
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from shapely.geometry import Polygon, MultiPolygon

from concurrent.futures import ThreadPoolExecutor, as_completed


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()

def get_score(target, out):
    intersection = np.logical_and(target, out)
    union = np.logical_or(target, out)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def iou(output, target):
    #doing interpolation and other stuff
    output, _ = output

    ### Classification
    output = output.max(dim=1)[1]
    output = output.float().unsqueeze(1)

    ### get and store the IOU metrics
    output = output.cpu().numpy()
    return get_score(output, target.cpu().numpy())


def evaluate(model, data_loader, neval_batches=10, to_cuda = True):
  model.eval();
  top1 = AverageMeter('Acc@1', ':6.2f')
  cnt = 0
  with torch.no_grad():
    for data in tqdm(data_loader):
      image = data['image']
      target = data['label']
      if to_cuda is True:
        model.to(torch.device('cuda'))
        input_tensor = Variable(image).cuda()
      else:
        model.to(torch.device('cpu'))
        input_tensor = Variable(image).cpu()
      output = model(input_tensor)
      #loss = criterion(output, target)
      cnt += 1
      acc = iou(output, target)
      print('.', end = '')
      top1.update(acc, image.size(0))
      if cnt >= neval_batches:
        return top1

  return top1

def debug_output_target(output, target):
  COLORS_DEBUG = [(255,0,0), (0,0,255)]
  clustered_out = get_clustered_output(output, True)
  target = target.cpu().numpy().astype(np.uint8)
  target = np.squeeze(target)
  debug_target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
  debug_target[np.where((debug_target == [1, 1, 1]).all(axis=2))] = COLORS_DEBUG[0]
  debug_target[np.where((debug_target == [2, 2, 2]).all(axis=2))] = COLORS_DEBUG[1]
  new_out = np.hstack((clustered_out, debug_target))
  cv2.imshow(new_out)

def debug_val_example(model, data_loader):
  with torch.no_grad():
    for data in tqdm(data_loader):
      image = data['image']
      target = data['label'].squeeze(1)
      target = torch.round(target*255) #converting to range 0-255
      target = target.type(torch.int64).cpu()
      model.to(torch.device('cpu'))
      input_tensor = Variable(image).cpu()
      output = model(input_tensor)
      break
  debug_output_target(output, target)
  
def get_clustered_output(output, multithreading):

  #doing interpolation and other stuff
  output, _ = output

  ### Classification
  resize_factor = 3
  output = output.max(dim=1)[1]
  output = output.float().unsqueeze(1)

  ### Resize to desired scale for easier clustering
  output = F.interpolate(output, size=(output.size(2) // resize_factor, output.size(3) // resize_factor) , mode='nearest')

  ### Obtaining actual output
  ego_lane_points = torch.nonzero(output.squeeze() == 1)
  other_lanes_points = torch.nonzero(output.squeeze() == 2)

  ego_lane_points = ego_lane_points.cpu().numpy()
  other_lanes_points = other_lanes_points.cpu().numpy()

  clusters = process_cnn_output([ego_lane_points, other_lanes_points], multithreading)
  all_lanes = get_lanes(clusters[0], clusters[1])
  final_output = np.zeros((120, 213, 3))
  COLORS = [(255,0,0), (0,255,0), (0,0,255)]

  if all_lanes[0] is not None:
      cv2.fillPoly(final_output, np.array([all_lanes[0]], dtype=np.int32), COLORS[0])
  if all_lanes[1] is not None:
      cv2.fillPoly(final_output, np.array([all_lanes[1]], dtype=np.int32), COLORS[1])
  if all_lanes[2] is not None:
      cv2.fillPoly(final_output, np.array([all_lanes[2]], dtype=np.int32), COLORS[2])

  final_output = cv2.resize(final_output, (640, 360))

  return final_output

def cluster(points):
  """
      Method used to cluster the points given by the CNN

      Args:
          points: points to cluster. They can be the points classified as egolane, the one classified
                  as other_lane, but NOT together

      Returns:
          pts: an array of arrays containing all the convex hulls (polygons) extracted for that group of points
  """
  pts = []
  eps = 1
  min_samples = 5 
  threshold_points = 700
  # Check added to handle when the network doesn't detect any point
  if len(points > 0):

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # This is an array of True values to ease class mask calculation
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[np.arange(len(db.labels_))] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    unique_labels = set(db.labels_)

    # Check if we have a cluster for hull extraction
    if n_clusters_ > 0:
      # Getting a boolean values array representing the pixels belonging to one cluster
      for index, k in enumerate(unique_labels):
        class_mask = points[core_samples_mask & (db.labels_ == k)]
        # Filtering clusters too small
        if class_mask.size > threshold_points:
          # If all the previous checks pass, then we get the hull curve to extract a polygon representing the lane
          hull = ConvexHull(class_mask)
          pts.append(np.vstack((class_mask[hull.vertices,1], class_mask[hull.vertices,0])).astype(np.int32).T)
  return pts

def process_cnn_output(to_cluster, multithreading = True):
  """
    Connector to handle the output of the CNN

    Args: 
        to_cluster: list of arrays of points. In to_cluster[0] there are the points classified by the CNN as egolane,
                    in to_cluster[1] the others. This have to be size 2
    Returns:
        clusters:     list of arrays of points. Here are saved the convex hull extracted. Again, in the first position
                    we find the hulls for the egolane, in the other for the other lanes
  """
  # Output containing the clusters
  clusters = [None] * 2

  ### Multithreading code 
  # We execute in a different thread the clustering for egolane and other lane, 
  # then synchronize the two thread using as_completed. Note that in the pool there are other
  # threads, not synchronized to the two used, ready to start the elaboration for another frame
  if(multithreading):
    threadpool = ThreadPoolExecutor(max_workers=5)
    
  if(multithreading):
    # Has to use a dict to avoid ambiguity between egolane and otherlanes. If a simple list is used, you can have
    # synchronization errors and put in clusters[0] the other lanes
    futures = {threadpool.submit(cluster, points) : index for index, points in enumerate(to_cluster)}
    for future in as_completed(futures):
      index = futures[future]
      clusters[index] = future.result()
  ### Single thread
  # If multithreading is disabled, we simply process the points sequentially
  else:
    clusters[0] = cluster(to_cluster[0])
    clusters[1] = cluster(to_cluster[1])

  return clusters

def get_lanes(egolane_clusters, other_lanes_clusters):
  """ 
      Method used to transform the hulls into polygons.
      The first thing to do is to select which of the clusters of the egolane is actually the egolane. 
      This is done selecting the cluster with the biggest area.
      Then, we subtract the intersections with the other lanes to avoid that one pixel is associated to both the egolane
      and another lane.
      Finally, we split the other lanes in left ones and right ones, basing the assumption on the centroid position.
      The biggest cluster on the right will be the right lane, the biggest on the left the left lane.

      args:
          egolane_clusters: set of points that represents the convex hull of the egolane. Can be more than one
          other_lanes_clusters: set of points that represents the convex hull of the other lanes. Can be more than one
  """ 
  ### Selecting the egolane
  egolane_polygons = [Polygon(x) for x in egolane_clusters]
  egolane = max(egolane_polygons, key=lambda p : p.area)

  egolane = Polygon(egolane)
  other_lanes_polygons = []

  ### Subtracting the intersecting pixels
  # note that this code gives priority to the detection of the other lanes; in this way we can minimize the risk of
  # getting on another lane
  for elem in other_lanes_clusters:
    elem = Polygon(elem)
    other_lanes_polygons.append(elem)
    egolane = egolane.difference(egolane.intersection(elem))

  ### Egolane refinement
  # The deletion of the intersecting regions can cause a split in the egolane in more polygons. In this case, 
  # the biggest polygon is selected as the new egolane
  if isinstance(egolane, MultiPolygon):
    polygons = list(egolane)
    egolane = max(polygons, key=lambda p : p.area)

  ### Splitting the other lanes in left and right ones
  left_lanes = [lane for lane in other_lanes_polygons if lane.centroid.x < egolane.centroid.x]
  right_lanes = [lane for lane in other_lanes_polygons if lane.centroid.x >= egolane.centroid.x]

  ### Selecting the right and the left lane
  left_lane = None if len(left_lanes) == 0 else max(left_lanes, key=lambda p : p.area)
  right_lane = None if len(right_lanes) == 0 else max(right_lanes, key=lambda p : p.area)

  ### Numpy conversion
  if egolane is not None: egolane = np.asarray(egolane.exterior.coords.xy).T
  if left_lane is not None: left_lane = np.asarray(left_lane.exterior.coords.xy).T
  if right_lane is not None: right_lane = np.asarray(right_lane.exterior.coords.xy).T
  
  return egolane, left_lane, right_lane
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from rospy.numpy_msg import numpy_msg
from ld_lsi.msg import CnnOutput, ClusteringOutput

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union

from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import time
import numpy as np

### Colors used for visualization
# Ego, left, right
COLORS = [(255,0,0), (0,255,0), (0,0,255)]


class ClusterExecutor:
    """
        Class used to extract clusters and to handle threading.
    """

    def __init__(self, eps, min_samples, threshold_points, multithreading=False, max_workers=5):
        """
            Class constructor

            Args:
                eps: epsilon used for DBSCAN
                min_samples: number of points required to be a cluster for DBSCAN
                threshold_points: number of points required to be a lane
                multithreading: enable multithreading via ThreadPoolExecutor
                max_workers: number of threads in the pool
        """
        # Clustering parameters
        self.eps = eps
        self.min_samples = min_samples
        self.threshold_points = threshold_points

        # Multithreading parameters
        self.multithreading = multithreading
        if(self.multithreading):
            self.threadpool = ThreadPoolExecutor(max_workers=max_workers)


    def cluster(self, points):
        """
            Method used to cluster the points given by the CNN

            Args:
                points: points to cluster. They can be the points classified as egolane, the one classified
                        as other_lane, but NOT together

            Returns:
                pts: an array of arrays containing all the convex hulls (polygons) extracted for that group of points
        """
        pts = []
        # Check added to handle when the network doesn't detect any point
        if len(points > 0):

            # DBSCAN clustering
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)

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
                    if class_mask.size > self.threshold_points:
                        # If all the previous checks pass, then we get the hull curve to extract a polygon representing the lane
                        hull = ConvexHull(class_mask)
                        pts.append(np.vstack((class_mask[hull.vertices,1], class_mask[hull.vertices,0])).astype(np.int32).T)
        return pts

    def process_cnn_output(self, to_cluster):
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
        if(self.multithreading):
            # Has to use a dict to avoid ambiguity between egolane and otherlanes. If a simple list is used, you can have
            # synchronization errors and put in clusters[0] the other lanes
            futures = {self.threadpool.submit(self.cluster, points) : index for index, points in enumerate(to_cluster)}
            for future in as_completed(futures):
                index = futures[future]
                clusters[index] = future.result()
        ### Single thread
        # If multithreading is disabled, we simply process the points sequentially
        else:
            clusters.append(self.cluster(to_cluster[0]))
            clusters.append(self.cluster(to_cluster[1]))

        return clusters        


class LaneExtractor:
    """
        Class used to transform a group of convex hulls of different classes in egolane, right lane and left lane
    """

    def __init__(self):
        """
        Class constructor
        """
        pass


    def get_lanes(self, egolane_clusters, other_lanes_clusters):
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


class LDClusteringNode:
    """
        Ros Node definition handling messages and errors.
    """
    
    def points_received_callback(self, points_data):
        """
            Callback called when a message from the CNN node is received. Used to
            elaborate the data using an instance of ClusterExecutor and publish a message
            on a topic to tell another node where the lanes are

            Args:
                points_data:    message of type CnnOutput containing the coordinates of egolane and otherlane points
        """

        start_t = time.time()

        try:
            # To start the loop, we log that we received the cnn output and begin measuring time
            rospy.loginfo("Received cnn output")

            # The lanes are passed as float32 arrays, so we have to get the 2d coordinates and process them
            egolane_points = np.reshape(points_data.egolane, (-1, 2))
            otherlanes_points = np.reshape(points_data.otherlanes, (-1, 2))
            clusters = self.cexe.process_cnn_output([egolane_points, otherlanes_points])


        except Exception as e:
            rospy.logerr("Couldn't cluster the data passed by CNN. Try to disable " \
                + " multithreading if it is enabled. Exception: %s" % e)

        msg = ClusteringOutput()

        ### Getting the actual lanes
        try:
            msg.ego_lane, msg.left_lane, msg.right_lane = self.le.get_lanes(clusters[0], clusters[1])
        except Exception as e:
            rospy.logwarn("No egolane detected. Sending None for everyone. Exception %s" % e)
            msg.ego_lane = None
            msg.left_lane = None
            msg.right_lane = None

        # Resending the road type
        msg.road_type = points_data.road_type

        ### Publishing the messages
        try:
            self.pub.publish(msg)
            self.time.append(time.time() - start_t)
            # Logging speed and extracted data
            fps = float(len(self.time)) / sum(self.time)

            rospy.loginfo("Sent message containing lanes. Working at %s fps" % fps)

        except Exception as e:
            rospy.logerr("Exception: %s" % e)

        ### Visualization
        try:
            if self.debug:
                lanes_image = np.zeros((90,160,3))

                if msg.ego_lane is not None:
                    cv2.fillPoly(lanes_image, np.array([msg.ego_lane], dtype=np.int32), COLORS[0])
                if msg.left_lane is not None:
                    cv2.fillPoly(lanes_image, np.array([msg.left_lane], dtype=np.int32), COLORS[1])
                if msg.right_lane is not None:
                    cv2.fillPoly(lanes_image, np.array([msg.right_lane], dtype=np.int32), COLORS[2])

                cv2.imshow("Lanes", cv2.resize(lanes_image, (320,240), cv2.INTER_NEAREST))
                cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Visualization error: %s " % e)


    def __init__(self):
        """
            Class constructor
        """
        try:
            ### Retrieving parameters for DBSCAN clustering from the command line
            # Epsilon is used to define the boundaries to search for other points in DBSCAN
            self.eps = rospy.get_param('eps', 1)
            # Minimum number of points between the boundaries to be a central point in DBSCAN
            self.min_samples = rospy.get_param('min_samples', 5)
            # Cluster filtering threshold, only consider clusters with > 700 points
            self.threshold_points = rospy.get_param('threshold_points', 700)
            # If enabled, it visualizes output
            self.debug = rospy.get_param('debug', True)
            # Enable multithreading using ThreadPoolExecutor. A pool of thread is made available for elaboration 
            self.multithreading = rospy.get_param('multithreading', True)
            # Number of threads in the pool
            self.max_workers = rospy.get_param('max_workers', 4)
            # ROS messages queue size
            queue_size = rospy.get_param('queue_size', 10)
        except Exception as e:
            print("Cannot read ros params. Maybe your roscore is down?")

        ### Declaring components
        # ClusterExecutor for managing clustering and threading
        try:
            self.cexe = ClusterExecutor(self.eps, self.min_samples, self.threshold_points, self.multithreading, self.max_workers)
            self.le = LaneExtractor()
        except Exception as e:
            rospy.logerr("Something went wrong initializing the ClusterExecutor. Exception: %s" % e)

        try:        
            # Time array used for FPS measurement
            self.time = []
            # Creating publisher to send lane info
            self.pub = rospy.Publisher('ld_lanes', numpy_msg(ClusteringOutput), queue_size=queue_size)

            ### ROS node setup
            rospy.init_node('ld_clustering_node', anonymous=True)
            rospy.Subscriber('ld_class_input', numpy_msg(CnnOutput), self.points_received_callback)
            rospy.spin()
        except Exception as e:
            rospy.logerr("Impossible to initialize the node and the other ROS components. Exception: %s" % e)

# Node activation
if __name__=='__main__':
    try:
        LDClusteringNode()
        rospy.loginfo("Shutting down node")
    except Exception as e:
        rospy.logerr("Something went very wrong. %s " % e)

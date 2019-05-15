#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from ld_lsi.msg import CnnOutput
import rospkg
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge

import torch
import importlib
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import cv2

import sys
import os
import time

### Colors for visualization
# Ego: red, other: blue
COLORS_DEBUG = [(255,0,0), (0,0,255)]

# Road name map
ROAD_MAP = ['Residential', 'Highway', 'City Street', 'Other']

class LDCNNNode:
    """
        CNN Node. It takes an image as input and process it using the neural network. Then it resizes the output
        and publish it on a topic.
    """
    def img_received_callback(self, image):
        '''
            Callback for image processing
            It submits the image to the CNN, extract the output, then resize it for clustering
            and publishes it on a topic
        
              Args:
                  image: image published on topic by the camera
        '''
        try:
            ### Pytorch conversion
            rospy.loginfo("Received image")
            start_t = time.time()
            image = self.bridge.imgmsg_to_cv2(image)
            input_tensor = torch.from_numpy(image)
            input_tensor = torch.div(input_tensor.float(), 255)
            input_tensor = input_tensor.permute(2,0,1).unsqueeze(0)
        except Exception as e:
            rospy.logerr("Cannot convert image to pytorch. Exception: %s" % e)

        try:
            ### PyTorch 0.4.0 compatibility inference code
            if torch.__version__ < "0.4.0":
                input_tensor = Variable(input_tensor, volatile=True).cuda()
                output = self.cnn(input_tensor)
            else:
                with torch.no_grad():
                    input_tensor = Variable(input_tensor).cuda()
                    output = self.cnn(input_tensor)

            if self.with_road:
                output, output_road = output
                road_type = output_road.max(dim=1)[1][0]
            ### Classification
            output = output.max(dim=1)[1]
            output = output.float().unsqueeze(0)

            ### Resize to desired scale for easier clustering
            output = F.interpolate(output, size=(output.size(2) / self.resize_factor, output.size(3) / self.resize_factor) , mode='nearest')

            ### Obtaining actual output
            ego_lane_points = torch.nonzero(output.squeeze() == 1)
            other_lanes_points = torch.nonzero(output.squeeze() == 2)

            ego_lane_points = ego_lane_points.view(-1).cpu().numpy()
            other_lanes_points = other_lanes_points.view(-1).cpu().numpy()

        except Exception as e:
            rospy.logerr("Cannot obtain output. Exception: %s" % e)

        try:
            ### Construction of message used for clustering

            msg = CnnOutput()
            msg.egolane = ego_lane_points
            msg.otherlanes = other_lanes_points
            msg.road_type = -1 if not self.with_road else road_type
            
            self.pub.publish(msg)

            ### Logging and fps measurement
            self.time.append(time.time() - start_t)
            rospy.loginfo("Sent lanes information to clustering node with " \
                + " %s ego lane points and %s other lanes points. %s fps" % (len(ego_lane_points), len(other_lanes_points), len(self.time) / sum(self.time)))
        except Exception as e:
            rospy.logerr("Cannot publish message. Exception: %s" % e)
        ### Debug visualization options
        if self.debug:
            try:
                # Convert the image and substitute the colors for egolane and other lane
                output = output.squeeze().unsqueeze(2).data.cpu().numpy()
                output = output.astype(np.uint8)

                output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
                output[np.where((output == [1, 1, 1]).all(axis=2))] = COLORS_DEBUG[0]
                output[np.where((output == [2, 2, 2]).all(axis=2))] = COLORS_DEBUG[1]

                # Blend the original image and the output of the CNN
                output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                image = cv2.addWeighted(image, 1, output, 0.4, 0)
                if self.with_road:
                    cv2.putText(image, ROAD_MAP[road_type], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Visualization
                rospy.loginfo("Visualizing output")
                cv2.imshow("CNN Output", cv2.resize(image, (320, 240), cv2.INTER_NEAREST))
                cv2.waitKey(1)
            except Exception as e:
                rospy.logerr("Visualization error. Exception: %s" % e)

    def __init__(self):
        """
            Class constructor.
        """
        try:
            # Adding models path to PYTHONPATH to import modules
            rospack = rospkg.RosPack()
            sys.path.insert(0, os.path.join(rospack.get_path('ld_lsi'),'res','models'))

            # Initialize CNN parameters with defaults
            model_name = rospy.get_param('model_name', 'erfnet_road')
            weights_name = rospy.get_param('weights_name', 'weights_erfnet_road.pth')
            self.resize_factor = rospy.get_param('resize_factor', 5)
            self.debug = rospy.get_param('debug', True)
            self.with_road = rospy.get_param('with_road', True)
            queue_size = rospy.get_param('queue_size', 10)
        except Exception as e:
            rospy.logerr("Cannot load parameters. Check your roscore. %s" % e)

        try:
            weights_path = os.path.join(rospack.get_path('ld_lsi'), 'res', 'weights', weights_name)

            # Assuming the main constructor is method Net()
            self.cnn = importlib.import_module(model_name).Net()

            # GPU only mode, setting up
            self.cnn = torch.nn.DataParallel(self.cnn).cuda()
            self.cnn.load_state_dict(torch.load(weights_path))
            self.cnn.eval()

            rospy.loginfo("Initialized CNN %s", model_name)
        except Exception as e:
            rospy.logerr("Cannot load neural network. Exception: %s" % e)

        # opencv Bridge for image translation
        self.bridge = CvBridge()
        self.time = []

        try:            
            # Publisher to send messages to the clustering node
            self.pub = rospy.Publisher('ld_class_input', numpy_msg(CnnOutput), queue_size=queue_size)

            # ROS node setup
            rospy.init_node('ld_cnn_node', anonymous=True)
            rospy.Subscriber('ld_image_input', Image, self.img_received_callback)
            rospy.spin()
        except Exception as e:
            rospy.logerr("Cannot initialize ros node. Exception: %s" % e)

# Node initialization
if __name__ == '__main__':
    node = LDCNNNode()
    rospy.loginfo("Shutting down node")

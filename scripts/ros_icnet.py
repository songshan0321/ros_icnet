#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from inference import ICNetInference

class RosInference:

    def __init__(self):
        self.m_rate = rospy.get_param("~m_rate", 30.0)
        self.m_input_image_topic = rospy.get_param("~m_input_image_topic", "/camera/image_raw")
        self.m_output_image_topic = rospy.get_param("~m_output_image_topic", "/camera/segmentation")
        self.m_weight_path = rospy.get_param("~m_weight_path", "/proj/tf_ws/src/ros_icnet/model/weight/model_best.ckpt")
        
        self.inference = ICNetInference(self.m_weight_path)
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(self.m_input_image_topic, Image, self.image_callback)
        self.image_pub = rospy.Publisher(self.m_output_image_topic, Image, queue_size=1)

        self.rate = rospy.Rate(self.m_rate)

        self.image = None

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def spin(self):

        while not rospy.is_shutdown():

            if self.image is not None:
                result = self.inference.infer(self.image)
                output_image, boundary = self.inference.process(self.image, result[0])
                #TODO: pointcloud = self.create_pointcloud(boundary)

                output_image = self.bridge.cv2_to_imgmsg(output_image)
                self.image_pub.publish(output_image)

            self.rate.sleep()

if __name__ == '__main__':

    rospy.init_node("icnet_inference_node", anonymous = False)
    ros_inference = RosInference()
    ros_inference.spin()
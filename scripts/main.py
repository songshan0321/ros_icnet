#!/usr/bin/env python
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge

from inference import ICNetInference

class RosInference:

    def __init__(self):
        self.m_rate = rospy.get_param("~m_rate", 30.0)
        self.m_input_camera_info_topic = rospy.get_param("~m_input_camera_info_topic", "/camera/camera_info")
        self.m_input_image_topic = rospy.get_param("~m_input_image_topic", "/camera/image_raw")
        self.m_output_image_topic = rospy.get_param("~m_output_image_topic", "/camera/segmentation")
        self.m_output_pointcloud_topic = rospy.get_param("~m_output_pointcloud_topic", "/camera/pointcloud")
        self.m_weight_path = rospy.get_param("~m_weight_path", "/proj/tf_ws/src/ros_icnet/model/weight/model_best.ckpt")
        
        self.inference = ICNetInference(self.m_weight_path)
        self.bridge = CvBridge()

        self.cam_info_sub = rospy.Subscriber(self.m_input_camera_info_topic, CameraInfo, self.info_callback, queue_size = 1)
        self.image_sub = rospy.Subscriber(self.m_input_image_topic, Image, self.image_callback, queue_size = 1)

        self.image_pub = rospy.Publisher(self.m_output_image_topic, Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(self.m_output_pointcloud_topic, PointCloud, queue_size=1)

        self.rate = rospy.Rate(self.m_rate)
        self.image = None

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def info_callback(self, msg):
        self.inference.camera_info = msg

    def spin(self):
        while not rospy.is_shutdown():

            if self.image is not None:
                result, duration = self.inference.infer(self.image)
                output_image, points = self.inference.process(self.image, result[0], duration)

                pointcloud = PointCloud()
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'camera_link'
                pointcloud.header = header

                for point in points:
                    pointcloud.points.append(Point32(point[0], point[1], point[2]))

                output_image = self.bridge.cv2_to_imgmsg(output_image)
                self.image_pub.publish(output_image)

                self.pointcloud_pub.publish(pointcloud)

                # self.image = None

            self.rate.sleep()

if __name__ == '__main__':

    rospy.init_node("icnet_inference_node", anonymous = False)
    ros_inference = RosInference()
    ros_inference.spin()
import rospy
import numpy as np
import random
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Range
from std_msgs.msg import ColorRGBA
from thymio_msgs.msg import SystemSound
import cv2
import datetime
import keyboard

LINEAR_VELOCITY = 0.04
ANGULAR_VELOCITY = 0.4
MAX_SENSOR_RANGE = 0.11
COLOR_ROOM = {1: [0.0, 0.0, 1.0], 2: [1.0, 0.0, 0.0]}
THRESHOLD_PREDICTION_IMAGES = 20
THRESHOLD_ACCURACY = 0.8
IMAGE_LIMIT = 30


MAX_LINEAR_VELOCITY = 0.13
MIN_LINEAR_VELOCITY = -0.13
MAX_ANGULAR_VELOCITY = 2.0
MIN_ANGULAR_VELOCITY = -2.0


# Author Alessio Della Libera and Andrea Bennati

def log(line):
    text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(text + " " + line)


class Timer:
    def __init__(self, delay):
        self.delay = delay
        self.timestamp = None

    def can_execute(self):
        if self.timestamp is None:
            return True

        if self.timestamp + self.delay < time.time():
            return True

        return False

    def update_time(self):
        self.timestamp = time.time()


class KidnappedThymio:
    def __init__(self, thymio_name, room, cnn, save_image, predict, render, record):
        rospy.init_node('thymio_controller', anonymous=True)
        rospy.sleep(1)

        # NEURAL NETWORK used for prediction
        self.cnn = cnn
        self.finished = False

        # PARAMETER CONFIGURATION
        self.thymio_name = thymio_name
        self.proximity_data = {}
        self.obstacle_detected = False
        self.image = None

        self.path = "{}/".format(room)
        self.save_image = save_image
        self.predict = predict
        self.render = render
        self.record = record
        self.counter_images = 0

        self.predict_timer = Timer(delay=1)
        self.save_image_timer = Timer(delay=0.5)

        # USED FOR PREDICTIONS
        self.predictions = {1: 0, 2: 0}  # 1 -> room1 , 2 -> room2
        self.number_predicted_images = 0.0
        self.counter_images = 0
        self.predicted_room = None

        # PUBLISHERS
        self.velocity_publisher = rospy.Publisher('/{}/cmd_vel'.format(thymio_name), Twist, queue_size=10)
        self.led_publisher_top = rospy.Publisher('{}/led/body/top'.format(thymio_name), ColorRGBA, queue_size=1)
        self.led_publisher_left = rospy.Publisher('{}/led/body/bottom_left'.format(thymio_name), ColorRGBA, queue_size=1)
        self.led_publisher_right = rospy.Publisher('{}/led/body/bottom_right'.format(thymio_name), ColorRGBA, queue_size=1)
        self.sound = rospy.Publisher('{}/sound/play/system'.format(thymio_name), SystemSound, queue_size=1)

        # CAMERA SENSORS
        rospy.Subscriber('/{}/camera/image_raw/compressed'.format(thymio_name), CompressedImage, self.read_raw_image, queue_size=1)

        # FRONT PROXIMITY SENSORS
        rospy.Subscriber('/{}/proximity/left'.format(thymio_name), Range, self.check_obstacles)
        rospy.Subscriber('/{}/proximity/center_left'.format(thymio_name), Range, self.check_obstacles)
        rospy.Subscriber('/{}/proximity/center'.format(thymio_name), Range, self.check_obstacles)
        rospy.Subscriber('/{}/proximity/center_right'.format(thymio_name), Range, self.check_obstacles)
        rospy.Subscriber('/{}/proximity/right'.format(thymio_name), Range, self.check_obstacles)

        self.rate = rospy.Rate(20)
        self.current_pose = Odometry().pose.pose
        self.message_velocity = Twist()

        self.controlled_by_human = False

        if self.record:
            fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # cv2.VideoWriter_fourcc(*'MP4V')
            self.video_writer = cv2.VideoWriter(
                "{}/output{}.mp4".format(self.path, str(datetime.datetime.fromtimestamp(time.time()))), fourcc, 20.0,
                (640, 480))

    def reset(self):
        self.predictions = {1: 0, 2: 0}  # 1 -> room1 , 2 -> room2
        self.number_predicted_images = 0.0

    def publish_msg(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.velocity_publisher.publish(msg)

    def check_obstacles(self, data):
        sensor_id = self.get_sensor_id(data)
        self.proximity_data[sensor_id] = data.range

        old_state = self.obstacle_detected
        self.obstacle_detected = not all(value >= MAX_SENSOR_RANGE for value in self.proximity_data.values())

        if old_state != self.obstacle_detected:
            log("State changed: " + ("obstacle found" if self.obstacle_detected else "free"))

    def go_back(self, linear_velocity, seconds):
        log("Going back...")
        t_end = time.time() + seconds
        while time.time() < t_end:
            self.publish_msg(-linear_velocity, 0)  # random angle
            self.rate.sleep()
        log("Going back done!!!!")

    def rotate_random(self, ang_vel=ANGULAR_VELOCITY):
        seconds = random.uniform(3, 7)
        t_end = time.time() + seconds
        angular_velocity = ang_vel if int(seconds * 10) % 2 == 0 else -ang_vel

        log("Rotate for {} seconds with angular velocity {} (at least)".format(seconds, angular_velocity))

        self.publish_msg(0, angular_velocity)  # random angle

        while time.time() < t_end or not all(value > MAX_SENSOR_RANGE for value in self.proximity_data.values()):
            self.rate.sleep()
        log("Rotation done!!!!")

    def get_sensor_id(self, sensor_data):
        sensor_id = sensor_data.header.frame_id
        sensor_id = sensor_id.split('/')[-1]
        return sensor_id

    def read_raw_image(self, img_data):
        pixels = np.fromstring(img_data.data, dtype=np.dtype(np.uint8))
        image = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
        self.image = image

    def turn_on_led(self, color):
        rgba = list(color) + [1.0]
        self.led_publisher_top.publish(*rgba)
        self.led_publisher_left.publish(*rgba)
        self.led_publisher_right.publish(*rgba)

    def save_images(self, image):
        cv2.imwrite("{}img{}.png".format(self.path, self.counter_images), image)
        self.counter_images += 1

    def render_image(self, image):
        cv2.imshow("Thymio Camera", image)
        cv2.waitKey(30)

    def record_video(self, image):
        if not self.video_writer:
            print("Error creating video")
        self.video_writer.write(image)

    def handle_image(self, image):
        if self.save_image:
            if self.save_image_timer.can_execute():  # save the image every n seconds
                self.save_images(image)
                self.save_image_timer.update_time()

        if self.render:
            self.render_image(image)

        if self.record:
            self.record_video(image)

        if self.predict:
            if self.predict_timer.can_execute():
                self.predict_room(image)
                self.predict_timer.update_time()

    def predict_room(self, image):
        image = cv2.resize(image, (128, 96))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        room_predicted, prob = self.cnn.predict(image)

        log("===> Predicted room {} with probability={}".format(room_predicted, prob))
        if prob <= 0.8:  # exclude the predictions with less than 80% probability
            log("**********> Prediction discarded :( - not so accurate")
            return

        self.predictions[room_predicted] += 1
        self.number_predicted_images += 1

        log("===> Total valid images predicted={} (room1={}, room2={})".format(self.number_predicted_images,
                                                                    (self.predictions[1] / self.number_predicted_images),
                                                                    (self.predictions[2] / self.number_predicted_images)))

        if self.number_predicted_images > THRESHOLD_PREDICTION_IMAGES:

            for room, count_room in self.predictions.items():
                if (count_room / self.number_predicted_images) > THRESHOLD_ACCURACY:
                    log("====> Predicted Room {}".format(room))

                    if self.predicted_room is None or self.predicted_room != room:
                        self.sound.publish(sound=SystemSound.central_button)
                        self.predicted_room = room

                    self.turn_on_led(COLOR_ROOM[room])

                    if not self.controlled_by_human:
                        self.stop()
                        self.finished = True
                    break

        if self.number_predicted_images > IMAGE_LIMIT:
            self.reset()

    def stop(self):
        log("Thymio stopped")
        self.publish_msg(0, 0)

    def random_walking(self):
        try:
            old_state = self.obstacle_detected
            while not rospy.is_shutdown():
                obstacle = self.obstacle_detected

                if old_state != obstacle:
                    log("Main loop ====> State changed " + ("obstacle found" if obstacle else "free"))
                    old_state = obstacle

                    if obstacle:
                        self.stop()
                        self.go_back(linear_velocity=LINEAR_VELOCITY, seconds=1.5)
                        self.rotate_random(ang_vel=ANGULAR_VELOCITY)

                self.publish_msg(LINEAR_VELOCITY, 0.0)

                if self.image is not None:
                    self.handle_image(self.image)
                    self.image = None

                if self.finished:
                    break
                self. rate.sleep()

        except (rospy.ROSInterruptException, KeyboardInterrupt) as e:
            self.publish_msg(0, 0)
            print("Int errupted: {}".format(e))

    def human_control(self):
        self.actual_linear_velocity = 0.0
        self.actual_angular_velocity = 0.0
        self.controlled_by_human = True

        log("Start human controlled Thymio")
        keyboard.on_press_key('down', self.slow_down_event, suppress=False)
        keyboard.on_press_key('up', self.speed_up_event, suppress=False)
        keyboard.on_press_key('left', self.turn_left_event, suppress=False)
        keyboard.on_press_key('right', self.turn_right_event, suppress=False)
        keyboard.on_press_key(' ', self.stop_event, suppress=False)

        try:
            while not rospy.is_shutdown():
                if self.image is not None:
                    self.handle_image(self.image)
                    self.image = None
                self.rate.sleep()

        except (rospy.ROSInterruptException, KeyboardInterrupt) as e:
            self.stop()
            print("Interrupted: {}".format(e))

    def bound_linear_velocity(self, value):
        return max(MIN_LINEAR_VELOCITY, min(MAX_LINEAR_VELOCITY, value))

    def bound_angluar_velocity(self, value):
        return max(MIN_ANGULAR_VELOCITY, min(MAX_ANGULAR_VELOCITY, value))

    def speed_up_event(self, data):
        self.actual_linear_velocity += 0.01
        self.actual_linear_velocity = self.bound_linear_velocity(self.actual_linear_velocity)
        log("speed up")
        self.publish_msg(self.actual_linear_velocity, 0.0)

    def slow_down_event(self, data):
        self.actual_linear_velocity -= 0.01
        self.actual_linear_velocity = self.bound_linear_velocity(self.actual_linear_velocity)
        log("slow down")
        self.publish_msg(self.actual_linear_velocity, 0.0)

    def turn_left_event(self, data):
        self.actual_angular_velocity += 0.1
        self.actual_angular_velocity = self.bound_angluar_velocity(self.actual_angular_velocity)
        log("turn left")
        self.publish_msg(self.actual_linear_velocity, self.actual_angular_velocity)

    def turn_right_event(self, data):
        self.actual_angular_velocity -= 0.1
        self.actual_angular_velocity = self.bound_angluar_velocity(self.actual_angular_velocity)
        log("turn right")
        self.publish_msg(self.actual_linear_velocity, self.actual_angular_velocity)

    def stop_event(self, data):
        self.actual_linear_velocity = 0.0
        self.actual_angular_velocity = 0.0
        self.stop()

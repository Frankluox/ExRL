"""ROS image/feedback input and raw-joint output for a Piper control node.

The control node owns enabling, joint limits, and reset motions. This adapter
publishes policy commands only while an episode is running.
"""

from __future__ import annotations

from collections import deque
import threading
import time
import numpy as np

from real_robot.observations import action_to_raw, state_from_raw


class PiperRobot:
    def __init__(self, config):
        import rospy
        from sensor_msgs.msg import Image, JointState
        from cv_bridge import CvBridge

        self.ros, self.joint_type, self.bridge = rospy, JointState, CvBridge()
        limits = np.asarray(config.get("joint_limits"), dtype=float)
        if (
            limits.shape != (6, 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError(
                "Set joint_limits to the six raw-unit limits for your Piper controller"
            )
        self.config = config
        self.condition = threading.Condition()
        self.samples = {k: deque(maxlen=128) for k in ("image", "wrist_image", "state")}
        self.offset = config.get("joint_offset", [0] * 6)
        if not rospy.core.is_initialized():
            rospy.init_node("exrl", anonymous=True, disable_signals=True)
        self.publisher = rospy.Publisher(config["action_topic"], JointState, queue_size=1)
        self.subscribers = [
            rospy.Subscriber(
                config["image_topic"],
                Image,
                lambda m: self._receive("image", m),
                queue_size=1,
                buff_size=2**24,
            ),
            rospy.Subscriber(
                config["wrist_topic"],
                Image,
                lambda m: self._receive("wrist_image", m),
                queue_size=1,
                buff_size=2**24,
            ),
            rospy.Subscriber(
                config["state_topic"], JointState, lambda m: self._receive("state", m), queue_size=1
            ),
        ]

    def _receive(self, name, msg):
        received, wall = time.monotonic(), time.time()
        stamp = msg.header.stamp.to_sec()
        if stamp <= 0 or not -0.01 <= wall - stamp <= self.config.get("max_age", 0.5):
            return
        if name == "state":
            value = state_from_raw(msg.position[:7], self.offset)
        else:
            value = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8").copy()
            crop = self.config.get(name + "_crop")
            if crop:
                x, y, w, h = map(int, crop)
                if x < 0 or y < 0 or x + w > value.shape[1] or y + h > value.shape[0]:
                    raise ValueError(f"Invalid {name} crop")
                value = value[y : y + h, x : x + w].copy()
        with self.condition:
            self.samples[name].append((received, stamp, value))
            self.condition.notify_all()

    def observe(self, after=None):
        """Return aligned camera/feedback samples newer than an action boundary."""
        after = time.monotonic() if after is None else float(after)
        wall_boundary = time.time() + after - time.monotonic()
        deadline = time.monotonic() + self.config.get("observation_timeout", 5)
        with self.condition:
            while time.monotonic() < deadline:
                rows = {
                    k: [v for v in values if v[0] >= after and v[1] >= wall_boundary]
                    for k, values in self.samples.items()
                }
                if all(rows.values()):
                    for image in rows["image"]:
                        wrist = min(rows["wrist_image"], key=lambda x: abs(x[1] - image[1]))
                        joint = min(rows["state"], key=lambda x: abs(x[1] - image[1]))
                        stamps = [image[1], wrist[1], joint[1]]
                        if max(stamps) - min(stamps) <= self.config.get("max_skew", 0.1):
                            return {
                                "image": image[2].copy(),
                                "wrist_image": wrist[2].copy(),
                                "state": joint[2].copy(),
                                "time": max(image[0], wrist[0], joint[0]),
                            }
                self.condition.wait(0.01)
        raise TimeoutError("No fresh, synchronized camera and joint observation")

    def command(self, action):
        if self.publisher.get_num_connections() == 0:
            raise RuntimeError("No Piper control node subscribes to the action topic")
        raw = action_to_raw(action, self.offset)
        limits = np.asarray(self.config["joint_limits"], dtype=float)
        if (
            limits.shape != (6, 2)
            or np.any(raw[:6] < limits[:, 0])
            or np.any(raw[:6] > limits[:, 1])
        ):
            raise ValueError("Policy command exceeds the configured Piper joint limits")
        msg = self.joint_type()
        msg.header.stamp = self.ros.Time.now()
        msg.name = [f"joint{i + 1}" for i in range(7)]
        msg.position = raw.astype(float).tolist()
        self.publisher.publish(msg)

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.publisher.unregister()

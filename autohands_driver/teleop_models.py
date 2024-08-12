import numpy as np


class AnalyticalTeleopModel:
    def __init__(self, motor_names):
        # set up params
        self.coeff = 1.0

        self.min_angle = np.ones([20, 1])*(0.0)
        self.max_angle = np.ones([20, 1])*90.0
        # import pdb
        # pdb.set_trace()
        self.max_angle[2] = 50.0
        # use numpy put
        np.put(self.max_angle, [6, 10, 14, 18], 110.0)
        # self.max_angle[6,10,14,18] = 110.0
        np.put(self.max_angle, [3, 7, 11, 15, 19], 70.0)
        # self.max_angle[3,7,11,15,19] = 70.0
        # set up the min and max angles first joint

        # thumb flex
        self.min_angle[0] = -20.0
        self.max_angle[0] = 50.0
        # thumb add
        self.min_angle[1] = -40.0
        self.max_angle[1] = -5.0

        # index add
        self.min_angle[4] = -13.0
        self.max_angle[4] = 7.0

        self.motor_names = motor_names

    def predict(self, x):
        x = np.reshape(x, (20, 1))
        # map from manus to motor cmds

        # 0 thumb flex dual tendon
        # 1 thumb add
        # 2 thumb flex_2
        # 3 thumb flex_3

        # 4 index add dual tendon
        # 5 index hinge
        # 6 index flex_2
        # 7 index flex_3

        # 8 middle add
        # 9 middle hinge
        # 10 middle flex_2
        # 11 middle flex_3

        # 12 ring add
        # 13 ring hinge
        # 14 ring flex_2
        # 15 ring flex_3

        # 16 pinky add
        # 17 pinky hinge
        # 18 pinky flex_2
        # 19 pinky flex_3

        motor_cmds = np.zeros([16, 1])

        # finger hinge and curl

        id_hinge_index = [i for i, name in enumerate(
            self.motor_names) if 'index_1' in name]
        id_hinge_middle = [i for i, name in enumerate(
            self.motor_names) if 'middle_1' in name]
        id_hinge_ring = [i for i, name in enumerate(
            self.motor_names) if 'ring_1' in name]
        id_hinge_pinky = [i for i, name in enumerate(
            self.motor_names) if 'pinky_1' in name]
        ids_hinge = id_hinge_index + id_hinge_middle + id_hinge_ring + id_hinge_pinky

        id_curl_index = [i for i, name in enumerate(
            self.motor_names) if 'index_23' in name]
        id_curl_middle = [i for i, name in enumerate(
            self.motor_names) if 'middle_23' in name]
        id_curl_ring = [i for i, name in enumerate(
            self.motor_names) if 'ring_23' in name]
        id_curl_pinky = [i for i, name in enumerate(
            self.motor_names) if 'pinky_23' in name]
        ids_curl = id_curl_index + id_curl_middle + id_curl_ring + id_curl_pinky

        ids_manus_hinge = [5, 9, 13, 17]
        ids_manus_curl_1 = [6, 10, 14, 18]
        ids_manus_curl_2 = [7, 11, 15, 19]
        # map from manus to motor cmds for the four fingers excluding thumb
        # import pdb
        # pdb.set_trace()
        hinge_cmds = (x[ids_manus_hinge]-self.min_angle[ids_manus_hinge]) / \
            (self.max_angle[ids_manus_hinge]-self.min_angle[ids_manus_hinge])
        motor_cmds[ids_hinge] = hinge_cmds

        # average the curl cmds for the two joints
        curl_cmds_1 = (x[ids_manus_curl_1]-self.min_angle[ids_manus_curl_1]) / \
            (self.max_angle[ids_manus_curl_1]-self.min_angle[ids_manus_curl_1])
        curl_cmds_2 = (x[ids_manus_curl_2]-self.min_angle[ids_manus_curl_2]) / \
            (self.max_angle[ids_manus_curl_2]-self.min_angle[ids_manus_curl_2])
        curl_cmds = (curl_cmds_1 + curl_cmds_2)/2.0
        motor_cmds[ids_curl] = curl_cmds

        # thumb joints
        idx_thumb_flex = [i for i, name in enumerate(
            self.motor_names) if 'thumb_flex' in name]
        idx_thumb_flex_manus = [0]
        # map from manus to motor cmds for the thumb

        motor_cmds[idx_thumb_flex] = (x[idx_thumb_flex_manus]-self.min_angle[idx_thumb_flex_manus])/(
            self.max_angle[idx_thumb_flex_manus]-self.min_angle[idx_thumb_flex_manus])
        motor_cmds[idx_thumb_flex] = -1 + 2*motor_cmds[idx_thumb_flex]

        idx_thumb2 = [i for i, name in enumerate(
            self.motor_names) if 'thumb_2' in name]
        idx_thumb2_manus = [2, 3]
        thumb2_cmds = (x[idx_thumb2_manus]-self.min_angle[idx_thumb2_manus]) / \
            (self.max_angle[idx_thumb2_manus]-self.min_angle[idx_thumb2_manus])
        thumb2_cmds = (thumb2_cmds[0] + thumb2_cmds[1])/2.0
        motor_cmds[idx_thumb2] = thumb2_cmds

        # thumb add
        idx_thumb_add = [i for i, name in enumerate(
            self.motor_names) if 'thumb_add' in name]
        idx_thumb_add_manus = [1]
        motor_cmds[idx_thumb_add] = (-x[idx_thumb_add_manus]-self.min_angle[idx_thumb_add_manus])/(
            self.max_angle[idx_thumb_add_manus]-self.min_angle[idx_thumb_add_manus])

        # index add
        idx_index_add = [i for i, name in enumerate(
            self.motor_names) if 'index_add' in name]
        idx_index_add_manus = [4]

        motor_cmds[idx_index_add] = (x[idx_index_add_manus] -
                                     self.min_angle[idx_index_add_manus]) / (self.max_angle[idx_index_add_manus] - self.min_angle[idx_index_add_manus])
        motor_cmds[idx_index_add] = -1*(-1 + 2*motor_cmds[idx_index_add])

        return self.coeff*motor_cmds


class AnalyticalTeleopModelThumbRotation:
    def __init__(self, motor_names):
        # set up params
        self.coeff = 1.0
        # self.alpha = np.pi/4.0
        self.alpha = 0.0

        self.min_angle = np.ones([20, 1])*(0.0)
        self.max_angle = np.ones([20, 1])*90.0
        # import pdb
        # pdb.set_trace()
        self.max_angle[2] = 50.0
        # use numpy put
        np.put(self.max_angle, [6, 10, 14, 18], 110.0)
        # self.max_angle[6,10,14,18] = 110.0
        np.put(self.max_angle, [3, 7, 11, 15, 19], 70.0)
        # self.max_angle[3,7,11,15,19] = 70.0
        # set up the min and max angles first joint

        # thumb flex
        self.min_angle[0] = -15.0
        self.max_angle[0] = 40.0
        # thumb add
        # inverted
        self.min_angle[1] = -55.0
        self.max_angle[1] = -10.0

        # index add
        self.min_angle[4] = -13.0
        self.max_angle[4] = 7.0

        self.motor_names = motor_names

    def predict(self, x):
        x = np.reshape(x, (20, 1))
        # map from manus to motor cmds

        # 0 thumb flex dual tendon
        # 1 thumb add
        # 2 thumb flex_2
        # 3 thumb flex_3

        # 4 index add dual tendon
        # 5 index hinge
        # 6 index flex_2
        # 7 index flex_3

        # 8 middle add
        # 9 middle hinge
        # 10 middle flex_2
        # 11 middle flex_3

        # 12 ring add
        # 13 ring hinge
        # 14 ring flex_2
        # 15 ring flex_3

        # 16 pinky add
        # 17 pinky hinge
        # 18 pinky flex_2
        # 19 pinky flex_3

        motor_cmds = np.zeros([16, 1])

        # finger hinge and curl

        id_hinge_index = [i for i, name in enumerate(
            self.motor_names) if 'index_1' in name]
        id_hinge_middle = [i for i, name in enumerate(
            self.motor_names) if 'middle_1' in name]
        id_hinge_ring = [i for i, name in enumerate(
            self.motor_names) if 'ring_1' in name]
        id_hinge_pinky = [i for i, name in enumerate(
            self.motor_names) if 'pinky_1' in name]
        ids_hinge = id_hinge_index + id_hinge_middle + id_hinge_ring + id_hinge_pinky

        id_curl_index = [i for i, name in enumerate(
            self.motor_names) if 'index_23' in name]
        id_curl_middle = [i for i, name in enumerate(
            self.motor_names) if 'middle_23' in name]
        id_curl_ring = [i for i, name in enumerate(
            self.motor_names) if 'ring_23' in name]
        id_curl_pinky = [i for i, name in enumerate(
            self.motor_names) if 'pinky_23' in name]
        ids_curl = id_curl_index + id_curl_middle + id_curl_ring + id_curl_pinky

        ids_manus_hinge = [5, 9, 13, 17]
        ids_manus_curl_1 = [6, 10, 14, 18]
        ids_manus_curl_2 = [7, 11, 15, 19]
        # map from manus to motor cmds for the four fingers excluding thumb
        # import pdb
        # pdb.set_trace()
        hinge_cmds = (x[ids_manus_hinge]-self.min_angle[ids_manus_hinge]) / \
            (self.max_angle[ids_manus_hinge]-self.min_angle[ids_manus_hinge])
        motor_cmds[ids_hinge] = hinge_cmds

        # average the curl cmds for the two joints
        curl_cmds_1 = (x[ids_manus_curl_1]-self.min_angle[ids_manus_curl_1]) / \
            (self.max_angle[ids_manus_curl_1]-self.min_angle[ids_manus_curl_1])
        curl_cmds_2 = (x[ids_manus_curl_2]-self.min_angle[ids_manus_curl_2]) / \
            (self.max_angle[ids_manus_curl_2]-self.min_angle[ids_manus_curl_2])
        curl_cmds = (curl_cmds_1 + curl_cmds_2)/2.0
        motor_cmds[ids_curl] = curl_cmds

        # thumb joints
        # thumb flex
        idx_thumb_flex = [i for i, name in enumerate(
            self.motor_names) if 'thumb_flex' in name]
        idx_thumb_flex_manus = [0]
        # map from manus to motor cmds for the thumb
        # thumb add
        idx_thumb_add = [i for i, name in enumerate(
            self.motor_names) if 'thumb_add' in name]
        idx_thumb_add_manus = [1]
        thumb_manus_activation_axis_1 = (x[idx_thumb_flex_manus]-self.min_angle[idx_thumb_flex_manus])/(
            self.max_angle[idx_thumb_flex_manus]-self.min_angle[idx_thumb_flex_manus])
        # import pdb
        # pdb.set_trace()
        thumb_manus_activation_axis_2 = (-x[idx_thumb_add_manus]-self.min_angle[idx_thumb_add_manus])/(
            self.max_angle[idx_thumb_add_manus]-self.min_angle[idx_thumb_add_manus])
        # rotate the thumb activations
        thumb_axis_1_rotated = np.cos(
            self.alpha)*thumb_manus_activation_axis_1 - np.sin(self.alpha)*thumb_manus_activation_axis_2
        thumb_axis_2_rotated = np.sin(
            self.alpha)*thumb_manus_activation_axis_1 + np.cos(self.alpha)*thumb_manus_activation_axis_2

        motor_cmds[idx_thumb_flex] = -1 + 2*thumb_axis_1_rotated
        motor_cmds[idx_thumb_add] = thumb_axis_2_rotated

        idx_thumb2 = [i for i, name in enumerate(
            self.motor_names) if 'thumb_2' in name]
        idx_thumb2_manus = [2, 3]
        thumb2_cmds = (x[idx_thumb2_manus]-self.min_angle[idx_thumb2_manus]) / \
            (self.max_angle[idx_thumb2_manus]-self.min_angle[idx_thumb2_manus])
        thumb2_cmds = (thumb2_cmds[0] + thumb2_cmds[1])/2.0
        motor_cmds[idx_thumb2] = thumb2_cmds

        # index add
        idx_index_add = [i for i, name in enumerate(
            self.motor_names) if 'index_add' in name]
        idx_index_add_manus = [4]

        motor_cmds[idx_index_add] = (x[idx_index_add_manus] -
                                     self.min_angle[idx_index_add_manus]) / (self.max_angle[idx_index_add_manus] - self.min_angle[idx_index_add_manus])
        motor_cmds[idx_index_add] = -1*(-1 + 2*motor_cmds[idx_index_add])

        return self.coeff*motor_cmds

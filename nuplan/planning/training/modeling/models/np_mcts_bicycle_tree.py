from __future__ import annotations

from time import perf_counter

import numpy as np

from nuplan.planning.training.modeling.models.np_mcts_bicycle_node import Node
from nuplan.planning.training.modeling.models.np_mcts_bicycle_utils import (
    check_drivable_area,
    check_ego_collisions_idx,
    get_directional_values,
    split_with_ratio,
    trajectory2action,
)


class Tree:
    """
    Class describing the full MCTS tree handling the planning
    """

    def __init__(
        self,
        init_state: dict,
        policy,
        graph_encoding=None,
        prediction=None,
        action=None,
        root=None,
        pred_idx=None,
        prior=(None, None),
        count=0,
    ):
        """
        Args:
            init_state: initial state of the tree
            policy: policy used to evaluate the nodes
            graph_encoding: graph encoding used to encode the map
            prediction: prediction of the agents
            action: initial action
            root: root node of the tree
            pred_idx: index of the prediction to use
            prior: prior of the root node
            count: number of simulations
        """

        self.count = count

        self.margin = 0.5
        self.pred_idx = pred_idx

        self.disp = False
        self.start_time = init_state["start_time"]
        self.prior = None
        if prior[0] is not None:
            self.prior = prior

        self.acc_target_range = [-3, 3]
        self.steering_target_range = [-np.pi / 4, np.pi / 4]

        self.acc_values = 13
        self.steering_values = 13

        self.acc_coef = (self.acc_target_range[1] - self.acc_target_range[0]) / (self.acc_values - 1)
        self.steer_coef = (self.steering_target_range[1] - self.steering_target_range[0]) / (self.steering_values - 1)

        self.default_possible_actions = np.arange(self.steering_values * self.acc_values)
        self.default_global2possible = {
            self.default_possible_actions[i]: i for i in range(len(self.default_possible_actions))
        }

        self.masks = {}
        self.masks_action = {}
        self.dt = 0.1

        self.rear2center = 1.461
        self.wheel_base = 3.1
        self.max_speed = init_state["max_speed"]
        offset_vector = np.array([self.rear2center, 0])[None, None]
        init_state["ego_pos"] = init_state["ego_pos"] + offset_vector
        init_state["map"] = init_state["map"] + offset_vector
        init_state["additional_map"] = init_state["additional_map"] + offset_vector

        self.n_nodes = 1
        self.Ts = np.zeros(81)
        self.nodes = {}
        self.mask_init = None
        self.penalty_init = None

        if action is not None:
            a, st = action
            a = np.fix(a / self.acc_coef) + 6
            st = np.fix(st / self.steer_coef) + 6
            action = (a, st)

            acc_vals = np.arange(self.acc_values)
            st_vals = np.arange(self.steering_values)
            acc_dif = np.abs(acc_vals - a)
            st_diff = np.abs(st_vals - st)
            speed = init_state["ego_speed"][0, -1][0]
            mask_a = (acc_dif < 5) | (speed < 0.1) * (a < self.acc_values // 2) * (
                acc_vals < (self.acc_values // 2 + 3)
            )
            mask_st = st_diff < 5
            self.mask_init = mask_a[:, None] * mask_st[None, :]

            self.penalty_init = np.zeros_like(self.mask_init)

            mask_normal, penalty, _ = self.get_action_masks(None, speed)

            self.mask_init = self.mask_init.flatten() * mask_normal
            self.penalty_init = self.penalty_init.flatten() + penalty

        predict_traj, predict_yaw = init_state["ego_pos"][:, -1], init_state["ego_yaw"][:, -1:]
        predicted_xy = predict_traj[:, :]
        predicted_yaw = predict_yaw[:, :]
        agents_xy = init_state["agents"][0, :, -1:, :2]
        agents_yaw = init_state["agents"][0, :, -1:, 2:3]
        mask_agents = init_state["agents_mask"][0, :, -1, 0]
        dim_agents = init_state["agents_dim"]
        is_collision = check_ego_collisions_idx(
            predicted_xy,
            predicted_yaw,
            agents_xy[None],
            agents_yaw[None],
            mask_agents[None],
            margin=[0.1, 0.1],
            other_dims=dim_agents,
        )
        if is_collision.sum():
            is_collision = is_collision > 0
            init_state["agents"] = init_state["agents"][0][~is_collision[0]][None]
            init_state["agents_mask"] = init_state["agents_mask"][0][~is_collision[0]][None]
            init_state["agents_dim"] = init_state["agents_dim"][0][~is_collision[0]][None]
            init_state["prediction"] = init_state["prediction"][0][~is_collision[0]][None]

        if root is not None:
            self.root = root
            root.T = 0
        else:
            self.root = Node(0, None, self, state=init_state, parent_state=init_state, action=action)

        self.policy = policy
        self.map_encoding = graph_encoding

        self.max_T = 80

        self.frames_history = 10
        self.n_actions = self.acc_values * self.steering_values
        self.prediction = prediction

        self.eval_frames = 10
        self.action_frames = 30

        self.eval_ratio = 5

        self.c_puct = 2
        self.tau = 1
        self.relu = lambda x: x * (x > 0)

        self.map_info = self.compute_map_infos(map=init_state["map"], map_mask=init_state["map_mask"])
        self.behind_mask = init_state["agents"][:, :, -1, 0] > 0

        agents_pos = init_state["prediction"][0, :, 0, 10, :2] - init_state["prediction"][0, :, 0, 0, :2]
        norm = np.sqrt((agents_pos**2).sum(-1))
        self.other_speeds = norm

        map = init_state["additional_map"]
        mask = init_state["additional_map_mask"]
        self.map_info_total = self.compute_map_infos(map=map, map_mask=mask)

        batch_size, n_agents, ntime, n_features = init_state["agents"].shape
        self.zeros = np.zeros((batch_size, 1 + n_agents, self.eval_frames, n_features))

        static_objects = init_state["static_objects"][:, :, None]
        static_mask = np.ones_like(static_objects)[..., 0]
        self.static_objects = static_objects, init_state["static_objects_dims"], static_mask

        pedestrians = init_state["pedestrians"][:, :, None]
        pedestrian_mask = np.ones_like(pedestrians)[..., 0]
        self.pedestrians = pedestrians, init_state["pedestrians_dims"], pedestrian_mask

        self.no_goal = False
        map, map_yaw, map_mask, map_avg_tan, map_avg_norm, max_lat, max_tan = self.map_info
        drivable, baseline_dist, time_drive, _, is_in_goal = check_drivable_area(
            np.array([[[self.rear2center, 0]]]),
            map,
            map_mask,
            np.array(
                [
                    [
                        [
                            0,
                        ],
                    ],
                ],
            ),
            map_yaw,
            map_avg_tan,
            map_avg_norm,
            max_lat,
            max_tan,
        )
        drivable = is_in_goal.min() == 0
        self.baseline_dist = 0

        if drivable:
            self.no_goal = True
        else:
            self.baseline_dist = baseline_dist

        self.no_drive = False
        map, map_yaw, map_mask, map_avg_tan, map_avg_norm, max_lat, max_tan = self.map_info_total
        drivable, baseline_dist, time_drive, _, is_in_drive = check_drivable_area(
            np.array([[[self.rear2center, 0]]]),
            map,
            map_mask,
            np.array(
                [
                    [
                        [
                            0,
                        ],
                    ],
                ],
            ),
            map_yaw,
            map_avg_tan,
            map_avg_norm,
            max_lat,
            max_tan,
        )
        is_in_drive = is_in_drive + is_in_goal
        drivable = is_in_drive.min() == 0
        if drivable:
            self.no_drive = True
        elif self.no_goal:
            self.baseline_dist = baseline_dist
        self.init_action = trajectory2action(trajectory=init_state["ego_pred"])
        self.acc_coef = (self.acc_target_range[1] - self.acc_target_range[0]) / (self.acc_values - 1)

    def simulate(self, k=128):
        start_time = perf_counter()
        self.k = k
        self.root.expand()
        self.first = True
        best_T = 0
        max_it_time = 0

        for _i in range(k):
            last_it = perf_counter()
            t, value, failure, success, s_node = self.search()
            if self.first:
                if failure:
                    self.first = False
                elif t >= self.max_T:
                    self.first = False
                    break
            if s_node and t > best_T:
                best_T = t
            now = perf_counter()
            max_it_time = max(max_it_time, now - last_it)
            time_elapsed = now - self.start_time

        return self.get_probas(), best_T

    def search(self):
        success = False
        leaf, value, path = self.root.select([])
        sucess_node = False

        if value is not None:
            value, success = value
            leaf.backup(value, T=leaf.T, success=success)
            failure = False
        else:
            if leaf.parent is not None:
                if leaf.state is None:
                    leaf.update_state()

                value, success, failure, fail_index, sucess_node = leaf.evaluate()

                if self.max_T - 1 > leaf.T and not failure:
                    leaf.expand()

                if leaf.predicted_score is not None:
                    value += -leaf.predicted_score[1]

                leaf.backup(value, T=leaf.T, success=success, fail_index=fail_index)

        return leaf.T, value, failure, success, sucess_node

    def get_probas(self):
        """
        Returns:
            acc_ns: acceleration value
            yr_ns: yaw rate value
        """

        n_frames = self.action_frames
        actions = np.zeros((80, self.acc_values * self.steering_values))
        actions_q = np.zeros((n_frames, self.acc_values * self.steering_values))
        self.root.update_probas_argmax(actions, actions_q)
        actions = actions.reshape((80, self.acc_values, self.steering_values))

        acc_ns = actions.max(-1)
        yr_ns = actions.max(-2)

        return acc_ns, yr_ns

    def ctridx2pos(self, acc, st, dt, initial_pos, initial_speed, initial_yaw):
        """
        Args:
            acc: acceleration value
            st: yaw rate value
            dt: time step
            initial_pos: initial position
            initial_speed: initial speed
            initial_yaw: initial yaw
        """

        acc_values = self.acc_values
        steering_values = self.steering_values

        discrete_acc = (self.acc_target_range[1] - self.acc_target_range[0]) * acc / (
            acc_values - 1
        ) + self.acc_target_range[0]
        discrete_steer = (self.steering_target_range[1] - self.steering_target_range[0]) * st / (
            steering_values - 1
        ) + self.steering_target_range[0]

        pred_speed = initial_speed + np.cumsum(discrete_acc, -1) * dt
        pred_speed = self.relu(pred_speed)

        discrete_yr = pred_speed * np.tan(discrete_steer) / self.wheel_base

        pred_yaw = initial_yaw + np.cumsum(discrete_yr, -1) * dt
        yaw_vec = np.stack([np.cos(pred_yaw), np.sin(pred_yaw)], -1)
        pred_velocity = yaw_vec * pred_speed[..., None]
        pred_pos = initial_pos[:, None] + np.cumsum(pred_velocity, 1) * dt

        return pred_pos, pred_speed[..., None], pred_yaw[..., None]

    def roll_sample(self, sample, pos, speed, yaw):
        """
        Args:
            sample: sample to roll
            pos: position to add
            speed: speed to add
            yaw: yaw to add
        """

        steps = pos.shape[1]
        steps = steps // self.eval_ratio
        new_sample = {}

        ego_features = sample["ego"]

        ego_pos = np.concatenate([sample["ego_pos"], pos], 1)
        ego_speed = np.concatenate([sample["ego_speed"], speed], 1)
        ego_yaw = np.concatenate([sample["ego_yaw"], yaw], 1)
        ego_features = np.concatenate([ego_features, self.zeros[:, 0]], 1)

        other_features = sample["prediction"][:, :, self.pred_idx, : self.eval_frames]

        agents_mask_shape_0, agents_mask_shape_1, _, agents_mask_shape_3 = sample["agents_mask"][
            :,
            :,
            -1:,
        ].shape
        other_mask = np.concatenate(
            [
                sample["agents_mask"],
                np.broadcast_to(
                    sample["agents_mask"][:, :, -1:],
                    (agents_mask_shape_0, agents_mask_shape_1, steps, agents_mask_shape_3),
                ),
            ],
            2,
        )

        new_sample["ego_pos"] = ego_pos
        new_sample["ego_speed"] = ego_speed
        new_sample["ego_yaw"] = ego_yaw

        new_sample["ego"] = ego_features[:, -self.frames_history :]

        new_sample["agents"] = other_features[:, :, -self.frames_history :]
        new_sample["agents_mask"] = other_mask[:, :, -self.frames_history :]
        new_sample["agents_dim"] = sample["agents_dim"]

        new_sample["prediction"] = sample["prediction"][:, :, :, self.eval_frames :]

        return new_sample

    def compute_map_infos(self, map, map_mask, th=0.1):
        """
        Args:
            map: map to use
            map_mask: mask of the map
            th: threshold to use
        """

        nodes = map[0]
        mask = map_mask[0]

        _, norm, yaw = get_directional_values(nodes=nodes, mask=mask)
        start = nodes[:, 0]
        rel_map = nodes - start[:, None]
        self_norm = (norm[:, None, None] @ rel_map[..., None])[..., 0]
        self_norm = self_norm * mask[..., None]
        max_lat = np.abs(self_norm[..., 0]).max(-1)

        lat_ratios = (max_lat // th).astype(np.int32)
        nodes, mask = split_with_ratio(ratios=lat_ratios, nodes=nodes, mask=mask)

        max_x = (nodes[..., 0] * mask).max(-1)
        mask_behind = max_x > -5
        nodes = nodes[mask_behind]
        mask = mask[mask_behind]

        map_avg_tan, map_avg_norm, yaw = get_directional_values(nodes=nodes, mask=mask)
        start = nodes[:, 0]
        rel_map = nodes - start[:, None]
        self_norm = (map_avg_norm[:, None, None] @ rel_map[..., None])[..., 0]
        self_norm = self_norm * mask[..., None]
        max_lat = np.abs(self_norm[..., 0]).max(-1)
        self_tan = (map_avg_tan[:, None, None] @ rel_map[..., None])[..., 0] * mask[..., None]
        max_tan = self_tan[..., 0].max(-1)

        map = nodes
        map_yaw = yaw
        map_mask = mask[..., None]

        map_avg_tan = map_avg_tan
        map_avg_norm = map_avg_norm

        return (
            map[None],
            map_yaw[None],
            map_mask[None],
            map_avg_tan[None],
            map_avg_norm[None],
            max_lat[None],
            max_tan[None],
        )

    def get_action_masks(self, action, value, dt=1):
        """
        Args:
            action: action to use
            value: value to use
            dt: time step
        """

        if action is not None:
            a, st = action
            key = (a, st, value // 0.1)
        else:
            key = value // 0.1

        if key not in self.masks:
            stopped = value <= 0

            mask_a, mask_st, acc_pen, st_pen, lon_jerk = self.get_action_value_mask(action, stopped, dt)
            mask_a, mask_st, acc_pen, st_pen = mask_a.copy(), mask_st.copy(), acc_pen, st_pen.copy()

            acc_values, steering_values = self.acc_values, self.steering_values
            mix_pen = np.zeros(acc_values * steering_values)

            if value <= 0:
                mask_a[: acc_values // 2] = 0
            if value <= 0.3:
                acc_step = (self.acc_target_range[1] - self.acc_target_range[0]) / (acc_values - 1)
                max_a = int(np.ceil(value / (self.dt * dt) / acc_step))
                mask_a[: acc_values // 2 - max_a] = 0

            st_vals = np.arange(steering_values)
            st_real_vals = (self.steering_target_range[1] - self.steering_target_range[0]) * st_vals / (
                steering_values - 1
            ) + self.steering_target_range[0]
            yr_real_vals = value * np.tan(st_real_vals) / self.wheel_base

            acc_lat = np.abs(yr_real_vals * value)
            mask_st *= acc_lat < 10

            if mask_a.sum() == 0:
                mask_a[acc_values // 2] = 1
            if mask_st.sum() == 0:
                mask_st[steering_values // 2] = 1
            mask_action = mask_a[:, None] * mask_st[None, :]

            mask_action = (mask_action).flatten()

            continuity_penalty = (acc_pen[:, None] + st_pen[None, :]).flatten() + mix_pen

            mask_action = mask_action == 1

            self.masks[key] = (mask_action, continuity_penalty, continuity_penalty[mask_action])

        return self.masks[key]

    def get_action_value_mask(self, action, stopped, dt):
        """
        Args:
            action: action to use
            stopped: whether the vehicle is stopped
            dt: time step
        """

        if action is not None:
            a, st = action
            key = (a, st, stopped)
        else:
            key = stopped
        if key not in self.masks_action:
            acc_values, steering_values = self.acc_values, self.steering_values

            mask_a = np.ones(acc_values)
            mask_st = np.ones(steering_values)

            acc_pen = np.zeros(acc_values)
            acc_pen[-2:] = 1
            st_pen = np.zeros(steering_values)
            acc_step = (self.acc_target_range[1] - self.acc_target_range[0]) / (acc_values - 1)

            lon_jerk = np.zeros(acc_values)

            if action is not None:
                acc_vals = np.arange(acc_values)
                acc_real_vals = (self.acc_target_range[1] - self.acc_target_range[0]) * acc_vals / (
                    acc_values - 1
                ) + self.acc_target_range[0]

                st_vals = np.arange(steering_values)
                acc_dif = np.abs(acc_vals - a)

                st_diff = np.abs(st_vals - st)

                lon_jerk = (not stopped) * acc_dif * acc_step / dt
                acc_pen += (lon_jerk > 4.13) * 1
                acc_pen += np.abs(acc_real_vals) / 10

                acc_pen += acc_dif / 5
                st_pen += st_diff / 10000

                mask_a *= acc_dif < 3 + (stopped) * (a < acc_values // 2) * (acc_vals < (acc_values // 2 + 3))
                mask_st *= st_diff < 3

            self.masks_action[key] = (mask_a, mask_st, acc_pen, st_pen, lon_jerk)

        return self.masks_action[key]

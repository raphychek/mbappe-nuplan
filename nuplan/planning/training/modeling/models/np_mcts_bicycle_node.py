from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nuplan.planning.training.modeling.models.np_mcts_bicycle_utils import check_drivable_area, check_ego_collisions


if TYPE_CHECKING:
    from nuplan.planning.training.modeling.models.np_mcts_bicycle_tree import Tree


class Node:
    """
    Class describing an MCTS node inside the Tree simulation. A node is made of a state, its values and its potential
    children
    """

    __slots__ = (
        "w",
        "p",
        "n",
        "q",
        "value",
        "children",
        "parent",
        "state",
        "parent_state",
        "next_actions",
        "tree",
        "action",
        "T",
        "children_p",
        "children_w",
        "children_n",
        "children_q",
        "n_perfect",
        "predicted_score",
        "speed",
        "hash_tuple",
        "last_parent",
        "parents",
        "hash_children",
        "failed",
        "drivable",
        "mask_action",
        "continuity_penalty",
        "probas",
        "total_mask",
        "possible_actions",
        "global2possible",
    )

    def __init__(
        self,
        p: float,
        parent: Node,
        tree: Tree,
        action: tuple | None = None,
        next_actions: tuple = ([[]], [[]]),
        parent_state=None,
        state: dict | None = None,
        T: int = 0,
    ):
        """
        Args:
            p: prior probability of the node
            parent: parent node
            tree: tree object
            action: action taken to reach this node
            next_actions: next actions to take
            parent_state: state of the parent node
            state: state of the node
            T: time step of the node
        """

        self.w = 0
        self.p = p
        self.n = 0
        self.q = 0
        self.n_perfect = 0
        self.children = {}
        self.parent = parent
        self.last_parent = parent
        self.parents = {(parent, action)}
        self.state = state
        self.hash_children = {}
        self.failed = False
        self.drivable = False
        self.value = None

        self.children_p = None
        if self.state is not None:
            self.speed = self.state["ego_speed"][0, -1][0]
        else:
            acc = action[0]
            discrete_acc = tree.acc_coef * acc + tree.acc_target_range[0]
            self.speed = max(self.parent.speed + discrete_acc * tree.dt, 0)

        self.parent_state = parent_state

        self.next_actions = next_actions
        self.tree = tree

        self.action = action

        self.mask_action, self.continuity_penalty, self.probas, self.total_mask = None, None, None, None

        self.possible_actions = self.tree.default_possible_actions
        self.global2possible = self.tree.default_global2possible

        self.tree.n_nodes += 1
        self.tree.Ts[T] += 1
        self.T = T
        self.predicted_score = None

        if self.parent is not None:
            self.predicted_score = self.parent.predicted_score

    def select(self, path: list | None = None):
        """
        Args:
            path: path to the node
        """

        path.append((self.T, self.action))

        if self.tree.max_T - 1 < self.T:
            return self, None, path

        if self.children_p is None:
            if len(self.next_actions[0][0]) == 0:
                return self, None, path
            else:
                self.expand()

        acc_values, steering_values = self.tree.acc_values, self.tree.steering_values

        if self.T > 0 and self.T % self.tree.eval_frames != 0:
            a, yr = self.action
            if self.speed <= 0:
                a = max(a, 6)

            selected = a * steering_values + yr

        else:
            if self.T == 0 and self.tree.penalty_init is not None:
                mask_action = self.tree.mask_init
                total_mask = self.tree.penalty_init[mask_action]
            else:
                mask_action, _, total_mask = self.tree.get_action_masks(self.action, self.speed)

            if self.probas is None:
                self.probas = self.tree.c_puct * self.children_p[mask_action]
                self.possible_actions = np.arange(steering_values * acc_values)[mask_action]
                self.global2possible = {self.possible_actions[i]: i for i in range(len(self.possible_actions))}
                self.children_q = self.children_q[mask_action]
                self.children_n = self.children_n[mask_action]
                self.children_w = self.children_w[mask_action]

            probas = self.probas
            children_counts = self.children_n
            summed = children_counts.sum()

            if self.tree.first or summed == 0:
                children_pucts = probas
            else:
                root_count_sum = np.sqrt(summed)
                children_pucts = self.children_q + probas * root_count_sum / (1 + children_counts)

            selected = np.argmax(children_pucts - total_mask)
            selected = self.possible_actions[selected]

        if selected in self.children:
            self.children[selected].last_parent = self
            return self.children[selected].select(path)
        else:
            a, yr = selected // steering_values, selected % steering_values

            next_acc, next_yr = self.next_actions
            next_action = (next_acc[:, 1:], next_yr[:, 1:])
            new_child = Node(
                self.children_p[selected],
                self,
                self.tree,
                action=(a, yr),
                next_actions=next_action,
                T=self.T + 1,
                parent_state=self.parent_state,
            )
            self.children[selected] = new_child

            return new_child.select(path)

    def evaluate(self):
        """Evaluate the node."""

        if self.n == 0:
            predict_traj, predict_yaw = (
                self.state["ego_pos"][:, -self.tree.eval_frames :],
                self.state["ego_yaw"][:, -self.tree.eval_frames :],
            )

            predicted_xy = predict_traj[:, :]
            predicted_yaw = predict_yaw[:, :]
            predicted_state_xy = self.state["agents"][:, :, -self.tree.eval_frames :, :2]
            predicted_state_yaw = self.state["agents"][:, :, -self.tree.eval_frames :, 2:3]
            mask_agents = self.state["agents_mask"][:, :, -self.tree.eval_frames :, 0]
            dim_agents = self.state["agents_dim"]

            margins = None

            real_collisions = check_ego_collisions(
                predicted_xy,
                predicted_yaw,
                predicted_state_xy[self.tree.behind_mask][None],
                predicted_state_yaw[self.tree.behind_mask][None],
                mask_agents[self.tree.behind_mask][None],
                margin=[0.7, 0.3],
                other_dims=dim_agents[self.tree.behind_mask][None],
                margins=margins,
                speed=self.speed,
                other_speeds=self.tree.other_speeds[self.tree.behind_mask[0]][None],
            )

            margin_collision = 0
            if real_collisions.sum():
                new_collisions = check_ego_collisions(
                    predicted_xy,
                    predicted_yaw,
                    predicted_state_xy[self.tree.behind_mask][None],
                    predicted_state_yaw[self.tree.behind_mask][None],
                    mask_agents[self.tree.behind_mask][None],
                    margin=[0.7, 0.3],
                    other_dims=dim_agents[self.tree.behind_mask][None],
                    margins=margins,
                    speed=0,
                )
                if not new_collisions.sum():
                    margin_collision = 1
                    real_collisions = np.array([0])

            behind_cols = 0
            static_cols = 0
            pedestrian_cols = 0

            if not real_collisions.sum():
                pedestrians_coords, pedestrians_dims, pedestrian_mask = self.tree.pedestrians

                pedestrian_cols = check_ego_collisions(
                    predicted_xy,
                    predicted_yaw,
                    pedestrians_coords[:, :, :, :2],
                    pedestrians_coords[:, :, :, 2:],
                    pedestrian_mask,
                    margin=[0.5, 0.5],
                    other_dims=pedestrians_dims,
                )
                pedestrian_cols = pedestrian_cols.sum()

                static_coords, static_dims, static_mask = self.tree.static_objects
                static_cols = check_ego_collisions(
                    predicted_xy,
                    predicted_yaw,
                    static_coords[:, :, :, :2],
                    static_coords[:, :, :, 2:],
                    static_mask,
                    margin=[0.2, 0.2],
                    other_dims=static_dims,
                )
                static_cols = static_cols.sum()

            progress = (self.state["ego_speed"][:, -1]).sum() / self.tree.max_speed
            progress = min(progress, 1)

            map, map_yaw, map_mask, map_avg_tan, map_avg_norm, max_lat, max_tan = self.tree.map_info
            drivable_goal, closest_dist, time_drive, closest_angle, is_in_goal = check_drivable_area(
                predict_traj,
                map,
                map_mask,
                predict_yaw,
                map_yaw,
                map_avg_tan,
                map_avg_norm,
                max_lat,
                max_tan,
            )
            drivable_goal = is_in_goal.min() == 0

            if drivable_goal:
                map, map_yaw, map_mask, map_avg_tan, map_avg_norm, max_lat, max_tan = self.tree.map_info_total
                (
                    drivable,
                    closest_dist_drive,
                    time_drive,
                    closest_angle_drive,
                    is_in_drivable,
                ) = check_drivable_area(
                    predict_traj,
                    map,
                    map_mask,
                    predict_yaw,
                    map_yaw,
                    map_avg_tan,
                    map_avg_norm,
                    max_lat,
                    max_tan,
                )
                is_in_drivable = is_in_drivable + is_in_goal
                closest_angle = min(closest_angle_drive, closest_angle)
                closest_dist = min(closest_dist_drive, closest_dist)
                drivable = is_in_drivable.min() == 0
            else:
                drivable = np.array([0])

            fail_index = None

            if drivable:
                fail_index = self.T - 10 + time_drive

            collision = real_collisions.sum() > 0
            drivable = drivable.sum()
            both = collision + drivable_goal
            goal_penality = -0.5 * drivable_goal * (drivable == 0)
            if self.tree.no_goal:
                goal_penality = 0.1 * (1 - drivable_goal)
                both = collision + drivable
            drivable_penalty = -drivable
            if self.tree.no_drive:
                drivable_penalty = 0.1 * (1 - drivable)
                both = collision
            closest_angle = np.abs(closest_angle)
            score = (
                -5 * collision
                - 2 * static_cols
                - behind_cols * 0.5
                - 0.5 * margin_collision
                - 3 * pedestrian_cols
                + drivable_penalty
                + goal_penality
                + progress * (both == 0) * (closest_angle < 0.20)
                + (both == 0) * 0.05 * (closest_angle < 0.35)
            )
            score = score.sum()

            if closest_dist < 100 and (both == 0):
                score -= closest_angle / 2 + closest_dist / 2

            if both > 0:
                self.failed = True
                if drivable:
                    self.drivable = True
            self.value = score
            return score, (self.tree.max_T == self.T) * 100 * (both == 0), both > 0, fail_index, (drivable) == 0
        else:
            return self.w / self.n, False, False, None, False

    def expand(self):
        """Expand the node."""

        if len(self.next_actions[0][0]) == 0:
            self.expand_nn()

        next_acc, next_yr = self.next_actions
        pred_acc = next_acc[0, 0]
        pred_yr = next_yr[0, 0]

        proba_action = pred_acc[:, None] * pred_yr[None, :]

        self.children_p = proba_action.flatten()
        self.children_w = 0 * self.children_p
        self.children_n = 0 * self.children_p
        self.children_q = 0 * self.children_p

    def expand_nn(self):
        """Expand the node."""

        baseline_acc, baseline_yr = 6, 6
        th = 0.0

        if self.speed < (self.tree.max_speed - th):
            baseline_acc = 8

        temp_acc, temp_yr = 100, 100
        power = 1

        init_acc = np.ones((1, self.tree.eval_frames, 13))
        acc_values = np.abs(np.arange(13) - baseline_acc)[None, None, :]
        acc_values = acc_values * init_acc
        next_acc = np.exp(-(acc_values**power) / temp_acc)
        next_acc[:, :, 11:] = 0

        init_yr = np.ones((1, self.tree.eval_frames, 13))
        yr_values = np.abs(np.arange(13) - baseline_yr)[None, None, :] * init_yr
        next_yr = np.exp(-(yr_values**power) / temp_yr)

        if self.T == 0:
            next_yr = np.zeros((1, self.tree.eval_frames, 13))
            nn_acc, nn_yr, nn_yrs = self.tree.init_action
            nn_acc = np.fix(nn_acc / self.tree.acc_coef) + 6
            nn_yr = np.fix(nn_yr / self.tree.steer_coef) + 6

            temp_acc = 100
            acc_values = np.abs(np.arange(13) - nn_acc)[None, None, :]
            acc_values = acc_values * init_acc

            next_acc += np.exp(-(acc_values**power) / temp_acc) * 0.9

            yr_values = np.abs(np.arange(13) - nn_yr)[None, None, :] * init_yr
            next_yr += np.exp(-(yr_values**power) / temp_yr)

        next_acc = next_acc / next_acc.sum(-1, keepdims=True)

        next_yr = next_yr / next_yr.sum(-1, keepdims=True)
        self.predicted_score = [0, 0]

        self.next_actions = (next_acc, next_yr)

    def backup(self, value, T, success=False, fail_index=None):
        """Back propagate the value of the node."""

        value = value + self.value if self.value is not None and self.n > 0 else value

        self.n += 1
        self.w += value
        self.q = self.w / self.n
        if success:
            self.n_perfect += 1

        if self.last_parent:
            steering_values = self.tree.steering_values

            idx = self.action[0] * steering_values + self.action[1]
            for parent, action in self.parents:
                idx = action[0] * steering_values + action[1]
                real_idx = self.parent.global2possible[idx]
                parent.children_n[real_idx] += 1
                parent.children_w[real_idx] += value
                parent.children_q[real_idx] = parent.children_w[real_idx] / parent.children_n[real_idx]
            self.last_parent.backup(value, T, success, fail_index)

    def get_past_action(self, list_acc, list_yr):
        """
        Get the past action of the node.

        Args:
            list_acc: list of accelerations
            list_yr: list of yaw rates

        Returns:
            list_acc: list of accelerations
            list_yr: list of yaw rates
        """

        list_acc.append(self.action[0])
        list_yr.append(self.action[1])

        if len(list_acc) < self.tree.eval_frames:
            return self.parent.get_past_action(list_acc, list_yr)
        else:
            list_acc.reverse()
            list_yr.reverse()
            return list_acc, list_yr

    def update_state(self):
        """Update the state of the node."""

        past_state = self.parent.parent_state
        acc, yr = self.get_past_action([], [])

        initial_pos = past_state["ego_pos"][:, -1]
        initial_speed = past_state["ego_speed"][:, -1]
        initial_yaw = past_state["ego_yaw"][:, -1]

        acc = np.array(acc)[None, :]
        yr = np.array(yr)[None, :]

        pred_pos, pred_speed, pred_yaw = self.tree.ctridx2pos(
            acc[:, -self.tree.eval_frames :],
            yr[:, -self.tree.eval_frames :],
            self.tree.dt,
            initial_pos,
            initial_speed,
            initial_yaw,
        )

        self.state = self.tree.roll_sample(sample=past_state, pos=pred_pos, speed=pred_speed, yaw=pred_yaw)
        self.parent_state = self.state

    def update_probas_argmax(self, actions, actions_q):
        """
        Update the probabilities of the node.

        Args:
            actions: actions
            actions_q: actions q values

        Returns:
            self
        """

        acc_values, steering_values = self.tree.acc_values, self.tree.steering_values

        if self.children:
            for a in range(acc_values):
                for yr in range(steering_values):
                    idx = a * steering_values + yr
                    if idx in self.children:
                        actions[self.T, idx] += (
                            self.children[a * self.tree.steering_values + yr].n
                            + 10 * self.children[a * self.tree.steering_values + yr].n_perfect
                        )

            if actions[self.T].sum() > 0 and self.tree.action_frames - 1 > self.T:
                child_max = np.argmax(actions[self.T])
                return self.children[child_max].update_probas_argmax(actions, actions_q)
            else:
                return self

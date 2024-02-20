from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def compute_rot_matrix(theta):
    """
    Computes the rotation matrix for a given angle theta

    Args:
        theta: angle in radians

    Returns:
        rotation matrix
    """

    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def compute_vertexes(x, y, heading, vertexes):
    """
    Computes the vertexes of the agent given its center position, heading and dimensions.

    Args:
        x: x coordinate of the center
        y: y coordinate of the center
        heading: heading of the agent
        vertexes: vertexes of the agent

    Returns:
        vertexes of the agent
    """

    agent_center = np.array([[x, y]])
    agent_center = np.transpose(agent_center)
    rot_matrix = compute_rot_matrix(heading)
    vertexes_rot = agent_center + np.dot(rot_matrix, vertexes)

    return np.transpose(vertexes_rot).tolist()


def compute_vertexes_batch(pos, heading, vertexes):
    """
    Computes the vertexes of the agent given its center position, heading and dimensions.

    Args:
        pos: position of the agent
        heading: heading of the agent
        vertexes: vertexes of the agent

    Returns:
        vertexes of the agent
    """

    agent_center = pos
    rot_matrix = compute_rot_matrix(heading)
    rot_matrix = np.transpose(rot_matrix, (2, 3, 0, 1))
    vertexes_rot = agent_center[..., None] + np.dot(rot_matrix, vertexes)
    vertexes_rot = np.transpose(vertexes_rot, (0, 1, 3, 2))
    return vertexes_rot


def compute_corners(length, width):
    """
    Computes the corners of the agent given its dimensions.

    Args:
        length: length of the agent
        width: width of the agent

    Returns:
        corners of the agent
    """

    vertexes = []
    vertexes.append((0 + length / 2, 0 + width / 2))
    vertexes.append((0 + length / 2, 0 - width / 2))
    vertexes.append((0 - length / 2, 0 - width / 2))
    vertexes.append((0 - length / 2, 0 + width / 2))
    vertexes = np.transpose(np.array(vertexes))

    return vertexes


def check_ego_collisions(
    ego_pred,
    ego_heading,
    other_gt,
    other_heading,
    other_mask,
    margin=(0, 0),
    other_dims=None,
    margins=None,
    speed=0,
    other_speeds=None,
):
    """
    Checks for collisions at every timestep between an ego prediction and the ground truth position of other vehicles.

    Args:
        ego_pred: tensor of size (batch, time, 2) indicating the sequence of coordinates of the ego center
        ego_heading: tensor of size (batch, time, 1) indicating the sequence of heading of the ego
        other_gt: tensor of size (batch, agent, time, 2) indicating the sequence of ccordinates for each agent
        other_heading: tensor of size (batch, agent, time, 1) indicating the sequence of heading for each agent
        other_mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent
        margin: margin to add to the dimensions of the agents
        other_dims: tensor of size (batch, agent, 2) indicating the dimensions of each agent
        margins: tensor of size (batch, agent, 2) indicating the margin to add to the dimensions of each agent
        speed: speed of the ego
        other_speeds: tensor of size (batch, agent) indicating the speed of each agent

    Returns:
        collision: boolean (batch)
    """

    diff = other_gt - ego_pred[:, None]
    dists = np.sqrt((diff**2).sum(-1))

    length, width = 5.176, 2.3

    length, width = length + margin[0] + speed / 2, width + margin[1]
    L1 = np.sqrt(length**2 + width**2)

    half_length, half_width = length / 2, width / 2
    if other_dims[0].shape[0] > 0:
        l_, w_ = other_dims[:, :, 0] / 2 + margin[0] / 2, other_dims[:, :, 1] / 2 + margin[1] / 2
        l_, w_ = l_[..., None], w_[..., None]
    else:
        l_, w_ = length / 2, width / 2
    if margins is not None:
        w_ = w_ + margins[..., None]
    if other_speeds is not None:
        l_ = l_ + other_speeds[..., None] / 4

    L2 = np.sqrt((2 * l_) ** 2 + (2 * w_) ** 2)
    L = (L1 + L2) / 2

    radius_small = (dists < width) * other_mask
    radius_check = (dists < L) * other_mask

    if radius_check.sum() == 0:
        return np.array([0])

    cos1, sin1 = np.cos(ego_heading[..., 0]), np.sin(ego_heading[..., 0])
    vx = np.stack([cos1, sin1], -1)
    vy = np.stack([-sin1, cos1], -1)
    ego_matrix = np.stack([vx, vy], -2)
    rel_diff = ego_matrix[:, None, :] @ diff[..., None]
    rel_diff = rel_diff[..., 0]

    delta_angle = np.abs(((other_heading - ego_heading[:, None]) + np.pi / 2) % np.pi - np.pi / 2)[..., 0]
    cos, sin = np.cos(delta_angle), np.sin(delta_angle)
    rel_diff = rel_diff[..., ::-1]

    enveloppe = (np.abs(rel_diff[..., 0]) <= (half_width + w_ * cos + l_ * sin)) & (
        np.abs(rel_diff[..., 1]) <= (half_length + l_ * cos + w_ * sin)
    )
    enveloppe_small = (np.abs(rel_diff[..., 0]) <= (half_width + half_width)) & (
        np.abs(rel_diff[..., 1]) <= (half_length + half_width)
    )

    total_check = radius_check * enveloppe
    radius_small = enveloppe_small * other_mask

    batch_size, n_other, n_steps = other_gt.shape[:3]
    real_cols = np.array(radius_small.sum((1, 2)) > 0)

    vertexes = compute_corners(length, width)
    worth_check = total_check.sum(1)

    for b in range(batch_size):
        if not real_cols[b]:
            batch_vertices_ = compute_vertexes_batch(ego_pred, ego_heading[..., 0], vertexes)
            for t in range(n_steps):
                if worth_check[b, t]:
                    ego_vertice = batch_vertices_[b, t]
                    ego_box = Polygon(ego_vertice)
                    for a in range(n_other):
                        if total_check[b, a, t]:
                            other_corners = compute_corners(2 * l_[b, a, 0], 2 * w_[b, a, 0])
                            other_vertice = compute_vertexes(
                                other_gt[b, a, t, 0],
                                other_gt[b, a, t, 1],
                                other_heading[b, a, t, 0],
                                other_corners,
                            )
                            other_box = Polygon(other_vertice)
                            is_col = ego_box.intersects(other_box)

                            if is_col:
                                real_cols[b] = 1
                                return real_cols

    return real_cols


def check_ego_collisions_idx(
    ego_pred,
    ego_heading,
    other_gt,
    other_heading,
    other_mask,
    margin=(0, 0),
    other_dims=None,
    margins=None,
    speed=0,
    other_speeds=None,
):
    """
    Checks for collisions at every timestep between an ego prediction and the ground truth position of other vehicles.

    Args:
        ego_pred: tensor of size (batch, time, 2) indicating the sequence of coordinates of the ego center
        ego_heading: tensor of size (batch, time, 1) indicating the sequence of heading of the ego
        other_gt: tensor of size (batch, agent, time, 2) indicating the sequence of ccordinates for each agent
        other_heading: tensor of size (batch, agent, time, 1) indicating the sequence of heading for each agent
        other_mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent
        margin: margin to add to the dimensions of the agents
        other_dims: tensor of size (batch, agent, 2) indicating the dimensions of each agent
        margins: tensor of size (batch, agent, 2) indicating the margin to add to the dimensions of each agent
        speed: speed of the ego
        other_speeds: tensor of size (batch, agent) indicating the speed of each agent

    Returns:
        collision: boolean (batch)
    """

    diff = other_gt - ego_pred[:, None]
    dists = np.sqrt((diff**2).sum(-1))

    length, width = 5.176, 2.3

    length, width = length + margin[0] + speed / 2, width + margin[1]
    L1 = np.sqrt(length**2 + width**2)

    half_length, half_width = length / 2, width / 2
    if other_dims[0].shape[0] > 0:
        l_, w_ = other_dims[:, :, 0] / 2 + margin[0] / 2, other_dims[:, :, 1] / 2 + margin[1] / 2
        l_, w_ = l_[..., None], w_[..., None]
    else:
        l_, w_ = length / 2, width / 2
    if margins is not None:
        w_ = w_ + margins[..., None]
    if other_speeds is not None:
        l_ = l_ + other_speeds[..., None] / 4

    L2 = np.sqrt((2 * l_) ** 2 + (2 * w_) ** 2)
    L = (L1 + L2) / 2

    radius_small = (dists < width) * other_mask
    radius_check = (dists < L) * other_mask

    batch_size, n_other, n_steps = other_gt.shape[:3]

    if radius_check.sum() == 0:
        return radius_check.sum(2)

    cos1, sin1 = np.cos(ego_heading[..., 0]), np.sin(ego_heading[..., 0])
    vx = np.stack([cos1, sin1], -1)
    vy = np.stack([-sin1, cos1], -1)
    ego_matrix = np.stack([vx, vy], -2)
    rel_diff = ego_matrix[:, None, :] @ diff[..., None]
    rel_diff = rel_diff[..., 0]

    delta_angle = np.abs(((other_heading - ego_heading[:, None]) + np.pi / 2) % np.pi - np.pi / 2)[..., 0]
    cos, sin = np.cos(delta_angle), np.sin(delta_angle)
    rel_diff = rel_diff[..., ::-1]

    enveloppe = (np.abs(rel_diff[..., 0]) <= (half_width + w_ * cos + l_ * sin)) & (
        np.abs(rel_diff[..., 1]) <= (half_length + l_ * cos + w_ * sin)
    )
    enveloppe_small = (np.abs(rel_diff[..., 0]) <= (half_width + half_width)) & (
        np.abs(rel_diff[..., 1]) <= (half_length + half_width)
    )

    total_check = radius_check * enveloppe
    radius_small = enveloppe_small * other_mask

    vertexes = compute_corners(length, width)
    worth_check = total_check.sum(1)

    is_collision = radius_small.sum(2)
    for b in range(batch_size):
        batch_vertices_ = compute_vertexes_batch(ego_pred, ego_heading[..., 0], vertexes)
        for t in range(n_steps):
            if worth_check[b, t]:
                ego_vertice = batch_vertices_[b, t]
                ego_box = Polygon(ego_vertice)
                for a in range(n_other):
                    if total_check[b, a, t]:
                        other_corners = compute_corners(2 * l_[b, a, 0], 2 * w_[b, a, 0])
                        other_vertice = compute_vertexes(
                            other_gt[b, a, t, 0],
                            other_gt[b, a, t, 1],
                            other_heading[b, a, t, 0],
                            other_corners,
                        )
                        other_box = Polygon(other_vertice)
                        is_col = ego_box.intersects(other_box)
                        if is_col:
                            is_collision[b, a] = 1

    return is_collision


def check_drivable_area(ego_pred, map, map_mask, ego_yaw, map_yaw, map_avg_tan, map_avg_norm, max_lat, max_tan):
    """
    Checks if the ego prediction is in the drivable area.

    Args:
        ego_pred: tensor of size (batch, time, 2) indicating the sequence of coordinates of the ego center
        map: tensor of size (batch, time, 2) indicating the sequence of coordinates of the map center
        map_mask: tensor of size (batch, time, 2) indicating the sequence of mask of the map
        ego_yaw: tensor of size (batch, time, 1) indicating the sequence of heading of the ego
        map_yaw: tensor of size (batch, time, 1) indicating the sequence of heading of the map
        map_avg_tan: tensor of size (batch, time, 2) indicating the sequence of average tangent of the map
        map_avg_norm: tensor of size (batch, time, 2) indicating the sequence of average normal of the map
        max_lat: tensor of size (batch, time, 1) indicating the sequence of maximum lateral distance of the map
        max_tan: tensor of size (batch, time, 1) indicating the sequence of maximum tangent of the map

    Returns:
        boolean (batch)
    """

    batch_size, n_lanes, n_points, d = map.shape

    length, width = 5.176, 2.3
    mask_diff = ((map_mask[:, :, 1:] & map_mask[:, :, :-1])[..., 0]).max(2)

    avg_tan = map_avg_tan
    norm = map_avg_norm

    start = map[:, :, 0]
    rel_ego = ego_pred[:, None] - start[:, :, None]

    lat_proj = (norm[:, :, None, None] @ rel_ego[..., None])[..., 0]

    within_norm_time = (np.abs(lat_proj[..., 0]) < (2 + max_lat[..., None] / 2 + length / 2)) * mask_diff[..., None]
    within_norm = within_norm_time.max(2)
    map_mask = map_mask[..., 0]
    batchsize = len(ego_pred)
    num_lanes = within_norm.sum(-1).astype(np.int32)
    max_lanes = int(num_lanes.max())

    if max_lanes == 0:
        return (
            np.ones(1),
            1001,
            0,
            0,
            np.zeros(1),
        )

    new_start = np.zeros((batchsize, max_lanes, 2))
    new_yaw = np.zeros((batchsize, max_lanes, n_points - 1))
    new_tan = np.zeros((batchsize, max_lanes, 2))
    new_mask = np.zeros((batchsize, max_lanes, n_points))
    new_abs_dists = np.zeros((batchsize, max_lanes))
    new_norm = np.zeros((batchsize, max_lanes, 2))
    new_max_lat = np.zeros((batchsize, max_lanes))
    new_max_tan = np.zeros((batchsize, max_lanes))

    for b in range(batchsize):
        new_start[b, : num_lanes[b]] = start[b][within_norm[b]]
        new_tan[b, : num_lanes[b]] = avg_tan[b][within_norm[b]]
        new_mask[b, : num_lanes[b]] = map_mask[b][within_norm[b]]
        new_yaw[b, : num_lanes[b]] = map_yaw[b][within_norm[b]]
        new_norm[b, : num_lanes[b]] = norm[b][within_norm[b]]
        new_max_lat[b, : num_lanes[b]] = max_lat[b][within_norm[b]]
        new_max_tan[b, : num_lanes[b]] = max_tan[b][within_norm[b]]

    vertexes = []
    vertexes.append((0 + length / 2, 0 + width / 2))
    vertexes.append((0 + length / 2, 0 - width / 2))
    vertexes.append((0 - length / 2, 0 - width / 2))
    vertexes.append((0 - length / 2, 0 + width / 2))
    vertexes = np.transpose(np.array(vertexes))
    batch_size, n_steps = ego_pred.shape[:2]

    batch_vertices_ = compute_vertexes_batch(ego_pred, ego_yaw[..., 0], vertexes)
    batch_vertices = batch_vertices_.reshape(batch_size, n_steps * 4, 2)
    batch_vertices = np.concatenate([batch_vertices, ego_pred[:, -1:]], 1)

    mask_diff = (new_mask[:, :, 1:] + new_mask[:, :, :-1]) > 1
    mask_tot = mask_diff.max(2)

    rel_ego = batch_vertices[:, None] - new_start[:, :, None]

    lat_proj = (new_norm[:, :, None, None] @ rel_ego[..., None])[..., 0]

    within_norm_time = (np.abs(lat_proj[..., 0]) < (1.8 + new_max_lat[..., None] / 2)) * mask_tot[..., None]
    new_abs_dists = np.abs(lat_proj[..., 0])[:, :, -1]

    tan_proj = (new_tan[:, :, None, None] @ rel_ego[..., None])[..., 0]
    within_tan = (
        within_norm_time
        * (tan_proj[..., 0] > -1)
        * (tan_proj[..., 0] < new_max_tan[..., None] + 1)
        * mask_tot[..., None]
    )

    selected = within_tan[:, :, -1]
    if selected.sum():
        new_tan = new_tan[selected]
        mask_diff = mask_diff[selected].max(1)

        new_abs_dists = new_abs_dists[selected]
        selected = selected[0]

        yaw_avg = np.arctan2(new_tan[..., 1], new_tan[..., 0])
        within_yaw = (ego_yaw[0, -1, 0] - yaw_avg) + (1 - mask_diff) * 10
        within_yaw = np.abs(within_yaw)

        closest_angle = np.sin(within_yaw) + (1 - mask_diff) * 1000
        angle_mask = (within_yaw > (np.ones_like(within_yaw) * np.pi / 4)).astype(np.float32) + (
            within_yaw > (np.ones_like(within_yaw) * np.pi / 2)
        ).astype(np.float32)
        within_both = (1 - mask_diff) * 1000 + new_abs_dists + angle_mask
        closest_angle = np.sin(np.abs(within_yaw)) + (1 - mask_diff) * 1000

    else:
        closest_angle = np.array([1000])
        within_both = np.array([1000])
    lat = within_tan.max(1)

    return (
        (1 - lat).max(1),
        within_both.min(),
        (1 - lat).argmax(),
        closest_angle.min(),
        lat[:, :-1],
    )


def split_with_ratio(ratios, nodes, mask):
    """
    Splits the nodes and mask with a given ratio.

    Args:
        ratios: tensor of size (batch, agent, time) indicating the ratio of each agent
        nodes: tensor of size (batch, agent, time, 2) indicating the sequence of coordinates for each agent
        mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent

    Returns:
        nodes: tensor of size (batch, agent, time, 2) indicating the sequence of coordinates for each agent
        mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent
    """

    ratio_mask = ratios >= 2
    ratios = ratios[ratio_mask]
    nodes_selected = nodes[ratio_mask]
    mask_selected = mask[ratio_mask]
    new_nodes = []
    new_masks = []
    n_long, n_points = nodes_selected.shape[:2]

    for i, lane in enumerate(nodes_selected):
        ratio = ratios[i]

        if ratio > 1:
            if ratio > 10:
                ratio = 10
            elif ratio > 5:
                ratio = 5
            elif ratio < 4 and ratio > 2:
                ratio = 2

            lanes = np.zeros((ratio, n_points, 2))
            masks = np.zeros((ratio, n_points))
            lanes[:, : n_points // ratio] = lane.reshape(ratio, n_points // ratio, 2)
            masks[:, : n_points // ratio] = mask_selected[i].reshape(ratio, n_points // ratio)

            if ratio < 10:
                next = lanes[1:, 0]
                lanes[:-1, n_points // ratio] = next
                next_mask = masks[1:, 0]
                masks[:-1, n_points // ratio] = next_mask
                new_nodes.append(lanes)
                new_masks.append(masks)
            else:
                additional_lanes = np.zeros((ratio - 1, n_points, 2))
                additional_masks = np.zeros((ratio - 1, n_points))
                additional_lanes[:, : n_points // ratio] = lane[1:-1].reshape(ratio - 1, n_points // ratio, 2)
                additional_masks[:, : n_points // ratio] = mask_selected[i][1:-1].reshape(ratio - 1, n_points // ratio)
                new_nodes.append(lanes)
                new_masks.append(masks)
                new_nodes.append(additional_lanes)
                new_masks.append(additional_masks)

    if new_nodes:
        new_nodes = np.concatenate(new_nodes, 0)
        new_masks = np.concatenate(new_masks, 0)
        nodes = np.concatenate([nodes[~ratio_mask], new_nodes], 0)
        mask = np.concatenate([mask[~ratio_mask], new_masks], 0)
        mask = mask > 0.1

    return nodes, mask


def get_directional_values(nodes, mask):
    """
    Computes the directional values of the nodes.

    Args:
        nodes: tensor of size (batch, agent, time, 2) indicating the sequence of coordinates for each agent
        mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent

    Returns:
        avg_tan: tensor of size (batch, agent, 2) indicating the average tangent for each agent
        norm: tensor of size (batch, agent, 2) indicating the average normal for each agent
        yaw: tensor of size (batch, agent, 1) indicating the average yaw for each agent
    """

    p1 = nodes[:, 1:]
    p2 = nodes[:, :-1]
    diff = p2 - p1

    mask_total = mask[:, 1:] & mask[:, :-1]
    mask_total = mask_total[..., None]
    norm = (diff**2).sum(-1, keepdims=True)
    tangent = diff / np.maximum(norm, 1e-6)

    avg_tan = (-tangent * mask_total).sum(-2) / np.maximum(mask_total[..., 0].sum(-1, keepdims=True), 1)
    tan_norm = np.sqrt((avg_tan**2).sum(-1))[..., None]
    avg_tan = avg_tan / np.maximum(tan_norm, 1e-6)

    norm = (np.array([[0, -1], [1, 0]]) @ avg_tan[..., None])[..., 0]
    yaw = np.arctan2(diff[..., 1], diff[..., 0])

    return avg_tan, norm, yaw


def trajectory2action(trajectory, dt=0.5):
    """
    Computes the action from a trajectory.

    Args:
        trajectory: tensor of size (time, 2) indicating the sequence of coordinates
        dt: time step

    Returns:
        acc: acceleration
        st2: steering
    """

    trajectory = trajectory[0]
    traj_diffs = trajectory[1:] - trajectory[:-1]
    speeds = np.sqrt((traj_diffs[..., :2] ** 2).sum(-1)) / dt
    accs = (speeds[1:] - speeds[:-1]) / dt

    yaws = np.arctan2(traj_diffs[:, 1], traj_diffs[:, 0])
    yrs = (yaws[1:] - yaws[:-1]) / dt

    yr = np.mean(yrs[:4])

    yrs_pred = traj_diffs[..., 2] / dt
    yr_pred = np.mean(yrs_pred[:2])
    yr = yr + yr_pred / 2
    yr = yr_pred

    speed_avg = (speeds[1:] + speeds[:-1]) / 2
    steerings2 = np.arctan(yrs * 3.1 / (speed_avg + 1e-6))
    st2 = np.mean(steerings2[:4])

    acc = np.mean(accs[:2])

    return acc, st2, None

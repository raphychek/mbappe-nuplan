from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class TrajectoryWeightDecayMultipleImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'trajectory_decay_multiple_imitation_objective'
        self._weight = weight
        self._fn_xy = torch.nn.modules.loss.L1Loss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting
        self._decay=1.0

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectories"]

    def compute(self, predictions: List[FeaturesType], targets: List[TargetsType], scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        batch_size = len(targets["agents_trajectories"].data)
        total = []
        traj_0 = targets["trajectory"].data[0]
        ego_mask = torch.ones((1,len(traj_0)), device=traj_0.device)
        #print(len(predictions["agents_trajectories"].data), len(targets["agents_trajectories"].data))
        planner_output_steps = traj_0.shape[0]
        decay_weight = torch.ones((1,len(traj_0), 1), device=traj_0.device)
        decay_value = torch.exp(-torch.Tensor(range(planner_output_steps)) / (planner_output_steps * self._decay))
        decay_weight[:, :] = decay_value[None, :, None]

        for sample_idx in range(batch_size):

            #predicted_trajectory = cast(Trajectory, predictions["agents_trajectories"].data[sample_idx])
            #targets_trajectory = Trajectory(targets["agents_trajectories"].data[sample_idx])
            #print(targets["agents_trajectories"].data[sample_idx].shape)
            #print(predicted_trajectory.xy.shape, targets_trajectory.xy.shape)

            #predicted_trajectory_xy = predictions["trajectory"].data[sample_idx][..., :2] 
            #predicted_trajectory_heading = predictions["trajectory"].data[sample_idx][..., 2]
            #print('  ', predictions["agents_trajectories"].data[sample_idx].shape, targets["agents_trajectories"].data[sample_idx].shape)
            predicted_trajectory_xy =  predictions["agents_trajectories"].data[sample_idx][..., :2]
            #print(predicted_trajectory_xy[1:, -1])
            predicted_trajectory_heading = predictions["agents_trajectories"].data[sample_idx][..., 2]

            #targets_trajectory_xy = targets["agents_trajectories"].data[sample_idx][..., :2]
            #targets_trajectory_heading = targets["agents_trajectories"].data[sample_idx][..., 2]

            targets_trajectory_xy = torch.cat((targets["trajectory"].data[sample_idx][None, ..., :2], targets["agents_trajectories"].data[sample_idx][..., :2]))
            targets_trajectory_heading = torch.cat((targets["trajectory"].data[sample_idx][None, ..., 2], targets["agents_trajectories"].data[sample_idx][..., 2]))
            target_mask = torch.cat((ego_mask, targets["agents_trajectories"].data[sample_idx][..., 3]))
            #print(targets_trajectory_xy.shape, target_mask.shape)

            if len(predicted_trajectory_xy) != len(targets_trajectory_xy):
                #print(predicted_trajectory_xy.shape, targets_trajectory_xy.shape)
                #print(targets_trajectory_xy)
                min_len = min(len(predicted_trajectory_xy), len(targets_trajectory_xy))
                predicted_trajectory_xy = predicted_trajectory_xy[:min_len]
                predicted_trajectory_heading = predicted_trajectory_heading[:min_len]
                targets_trajectory_xy = targets_trajectory_xy[:min_len]
                targets_trajectory_heading = targets_trajectory_heading[:min_len]
                target_mask = target_mask[:min_len]

            #loss_weights = extract_scenario_type_weight(
            #    scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory_xy.device
            #)
            #
            #broadcast_shape_xy = tuple([-1] + [1 for _ in range(predicted_trajectory_xy.dim() - 1)])
            #broadcasted_loss_weights_xy = loss_weights.view(broadcast_shape_xy)
            #broadcast_shape_heading = tuple([-1] + [1 for _ in range(predicted_trajectory_heading.dim() - 1)])
            #broadcasted_loss_weights_heading = loss_weights.view(broadcast_shape_heading)
            #
            #print("predicted_trajectory_xy, targets_trajectory_xy: ", predicted_trajectory_xy.shape, targets_trajectory_xy.shape)
            #print("broadcasted_loss_weights_xy: ", broadcasted_loss_weights_xy.shape)
            
            weighted_xy_loss = self._fn_xy(predicted_trajectory_xy, targets_trajectory_xy)*target_mask[..., None] #* broadcasted_loss_weights_xy
            #print(weighted_xy_loss.shape, target_mask.shape)
            weighted_heading_loss = (
                self._fn_heading(predicted_trajectory_heading, targets_trajectory_heading)*target_mask
                #* broadcasted_loss_weights_heading
            )
            # Assert that broadcasting was done correctly
            assert weighted_xy_loss.size() == predicted_trajectory_xy.size()
            assert weighted_heading_loss.size() == predicted_trajectory_heading.size()
            # print(torch.abs(targets_trajectory_xy).sum(-1))
            # print(target_mask.sum(-1))
            # print(weighted_xy_loss.mean(-1).mean(-1))
            total.append(torch.mean(weighted_xy_loss * decay_weight) + torch.mean(weighted_heading_loss * decay_weight[:, :, 0]))
        # print(torch.mean(torch.stack(total, 0)))
        return self._weight * torch.mean(torch.stack(total, 0))

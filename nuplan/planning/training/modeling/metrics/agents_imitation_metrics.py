from typing import List

import torch

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories
from nuplan.planning.training.preprocessing.features.multiple_agents_trajectories import MultipleAgentsTrajectories


class AgentsAverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of all agents' trajectory.
    """

    def __init__(self, name: str = 'agents_avg_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: AgentsTrajectories = predictions["agents_trajectory"]
        target_agents: AgentsTrajectories = targets["agents_trajectory"]
        batch_size = predicted_agents.batch_size

        error = torch.mean(
            torch.tensor(
                [
                    torch.norm(predicted_agents.xy[sample_idx] - target_agents.xy[sample_idx], dim=-1).mean()
                    for sample_idx in range(batch_size)
                ]
            )
        )

        return error


class AgentsFinalDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from the final pose of all agents trajectory.
    """

    def __init__(self, name: str = 'agents_final_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: AgentsTrajectories = predictions["agents_trajectory"]
        target_agents: AgentsTrajectories = targets["agents_trajectory"]
        batch_size = predicted_agents.batch_size

        error = torch.mean(
            torch.tensor(
                [
                    torch.norm(
                        predicted_agents.terminal_xy[sample_idx] - target_agents.terminal_xy[sample_idx], dim=-1
                    ).mean()
                    for sample_idx in range(batch_size)
                ]
            )
        )
        return error


class AgentsAverageHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error averaged from all poses of all agents trajectory.
    """

    def __init__(self, name: str = 'agents_avg_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: AgentsTrajectories = predictions["agents_trajectory"]
        target_agents: AgentsTrajectories = targets["agents_trajectory"]
        batch_size = predicted_agents.batch_size

        errors = []
        for sample_idx in range(batch_size):
            error = torch.abs(predicted_agents.heading[sample_idx] - target_agents.heading[sample_idx])
            error_wrapped = torch.atan2(torch.sin(error), torch.cos(error)).mean()
            errors.append(error_wrapped)
        return torch.mean(torch.tensor(errors))


class AgentsFinalHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error from the final pose of all agents agents.
    """

    def __init__(self, name: str = 'agents_final_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: AgentsTrajectories = predictions["agents_trajectory"]
        target_agents: AgentsTrajectories = targets["agents_trajectory"]
        batch_size = predicted_agents.batch_size

        errors = []
        for sample_idx in range(batch_size):
            error = torch.abs(
                predicted_agents.terminal_heading[sample_idx] - target_agents.terminal_heading[sample_idx]
            )
            error_wrapped = torch.atan2(torch.sin(error), torch.cos(error)).mean()
            errors.append(error_wrapped)

        return torch.mean(torch.tensor(errors))



class MultiAverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of all agents' trajectory.
    """

    def __init__(self, name: str = 'multi_avg_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectories"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: MultipleAgentsTrajectories = predictions["agents_trajectories"]
        target_agents: MultipleAgentsTrajectories = targets["agents_trajectories"]
        batch_size = predicted_agents.batch_size
        total = []
        traj_0 = targets["trajectory"].data[0]
        ego_mask = torch.ones((1,len(traj_0)), device=traj_0.device)
        for sample_idx in range(batch_size):
  
            predicted_trajectory_xy =  predictions["agents_trajectories"].data[sample_idx][..., :2]
            predicted_trajectory_heading = predictions["agents_trajectories"].data[sample_idx][..., 2]

            targets_trajectory_xy = torch.cat((targets["trajectory"].data[sample_idx][None, ..., :2], targets["agents_trajectories"].data[sample_idx][..., :2]))
            targets_trajectory_heading = torch.cat((targets["trajectory"].data[sample_idx][None, ..., 2], targets["agents_trajectories"].data[sample_idx][..., 2]))
            target_mask = torch.cat((ego_mask, targets["agents_trajectories"].data[sample_idx][..., 3]))

            if len(predicted_trajectory_xy) != len(targets_trajectory_xy):
                min_len = min(len(predicted_trajectory_xy), len(targets_trajectory_xy))
                predicted_trajectory_xy = predicted_trajectory_xy[:min_len]
                predicted_trajectory_heading = predicted_trajectory_heading[:min_len]
                targets_trajectory_xy = targets_trajectory_xy[:min_len]
                targets_trajectory_heading = targets_trajectory_heading[:min_len]
                target_mask = target_mask[:min_len]

            l2 = torch.norm(predicted_trajectory_xy - targets_trajectory_xy, dim=-1)*target_mask #* broadcasted_loss_weights_xy
            full_traj = target_mask.amin(-1)

            l2 = l2.mean(-1)
            # print(l2)
            # print(full_traj)
            total.append((l2*full_traj).sum(-1)/full_traj.sum(-1))

        # error = torch.mean(
        #     torch.tensor(
        #         [
        #             torch.norm(predicted_agents.data[sample_idx][..., :2] - target_agents.data[sample_idx][..., :2], dim=-1).mean()
        #             for sample_idx in range(batch_size)
        #         ]
        #     )
        # )

        return torch.mean(torch.stack(total, 0))
    
class NumberAgents(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of all agents' trajectory.
    """

    def __init__(self, name: str = 'number_agents') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectories"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_agents: MultipleAgentsTrajectories = predictions["agents_trajectories"]
        target_agents: MultipleAgentsTrajectories = targets["agents_trajectories"]
        batch_size = predicted_agents.batch_size
        traj_0 = targets["trajectory"].data[0]
        ego_mask = torch.ones((1,len(traj_0)), device=traj_0.device)
        total = []
        for sample_idx in range(batch_size):
  
            predicted_trajectory_xy =  predictions["agents_trajectories"].data[sample_idx][..., :2]
            predicted_trajectory_heading = predictions["agents_trajectories"].data[sample_idx][..., 2]

            targets_trajectory_xy = torch.cat((targets["trajectory"].data[sample_idx][None, ..., :2], targets["agents_trajectories"].data[sample_idx][..., :2]))
            targets_trajectory_heading = torch.cat((targets["trajectory"].data[sample_idx][None, ..., 2], targets["agents_trajectories"].data[sample_idx][..., 2]))
            target_mask = torch.cat((ego_mask, targets["agents_trajectories"].data[sample_idx][..., 3]))

            if len(predicted_trajectory_xy) != len(targets_trajectory_xy):
                min_len = min(len(predicted_trajectory_xy), len(targets_trajectory_xy))
                predicted_trajectory_xy = predicted_trajectory_xy[:min_len]
                predicted_trajectory_heading = predicted_trajectory_heading[:min_len]
                targets_trajectory_xy = targets_trajectory_xy[:min_len]
                targets_trajectory_heading = targets_trajectory_heading[:min_len]
                target_mask = target_mask[:min_len]

            
            #l2 = torch.norm(predicted_agents.data[sample_idx][..., :2] - target_agents.data[sample_idx][..., :2], dim=-1)*target_mask[..., None] #* broadcasted_loss_weights_xy
            full_traj = target_mask.amin(-1)


            total.append(full_traj.sum(-1))

        # error = torch.mean(
        #     torch.tensor(
        #         [
        #             torch.norm(predicted_agents.data[sample_idx][..., :2] - target_agents.data[sample_idx][..., :2], dim=-1).mean()
        #             for sample_idx in range(batch_size)
        #         ]
        #     )
        # )

        return torch.mean(torch.stack(total, 0))

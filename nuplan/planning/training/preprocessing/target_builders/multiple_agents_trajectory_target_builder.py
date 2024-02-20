from __future__ import annotations

from typing import Type, List
import numpy as np
import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.multiple_agents_trajectories import MultipleAgentsTrajectories
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex

from nuplan.planning.training.preprocessing.features.agents_multi import AgentsMulti

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    sampled_past_ego_states_to_tensor,
    sampled_tracked_objects_to_tensor_list_mask,
    sampled_tracked_objects_to_tensor_list,
    filter_agents_tensor_novalidate,
    convert_absolute_quantities_to_relative,
    sampled_past_timestamps_to_tensor,
    compute_yaw_rate_from_state_tensors,
    pack_agents_tensor,
    pad_agent_states,
    pad_agent_states_mask
)



class MultipleAgentsTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""

    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes the class.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        """
        self._num_future_poses = 16#future_trajectory_sampling.num_poses
        self._time_horizon = 8#future_trajectory_sampling.time_horizon
        #print("Init")
        self._agents_states_dim = 4

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "agents_trajectories"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return MultipleAgentsTrajectories  # type: ignore

    # def get_targets(self, scenario: AbstractScenario) -> Trajectory:
    #     """Inherited, see superclass."""
    #     current_absolute_state = scenario.initial_ego_state
    #     print("current_absolute_state: ", current_absolute_state)
    #     trajectories = scenario.get_agents_future_trajectory(
    #         iteration=0, future_time_horizon=self._time_horizon
    #     )

    #     print("trajectory_absolute_states: ", trajectories)

    #     ## Get all future poses relative to the ego coordinate system
    #     #trajectory_relative_poses = convert_absolute_to_relative_poses(
    #     #    current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
    #     #)

    #     #if len(trajectory_relative_poses) != self._num_future_poses:
    #     #    raise RuntimeError(f'Expected {self._num_future_poses} num poses but got {len(trajectory_absolute_states)}')

    #     return Trajectory(data=trajectories)

    def get_targets(
            self, scenario: AbstractScenario
        ) -> MultipleAgentsTrajectories:
            """Inherited, see superclass."""

            #print(targets)
            traj_sampling = TrajectorySampling(time_horizon=self._time_horizon, num_poses = 16)
            trajectories = []
            #others = [scenario.get_future_tracked_objects(iteration=0, time_horizon=self.future_time_horizon, future_trajectory_sampling = traj_sampling, num_samples=16)
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            others = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(iteration=0, time_horizon=self._time_horizon, future_trajectory_sampling = traj_sampling, num_samples=16)
            ]
            future_others = [present_tracked_objects]+ others 
            future_others = sampled_tracked_objects_to_tensor_list(future_others)

            time_stamps = [scenario.start_time]+list(scenario.get_future_timestamps(
                iteration=0, num_samples=16, time_horizon=self._time_horizon))
            time_stamps = sampled_past_timestamps_to_tensor(time_stamps)

            past_ego_states = [scenario.initial_ego_state]
            past_ego_states_tensor = sampled_past_ego_states_to_tensor(past_ego_states)
            anchor_ego_state = past_ego_states_tensor[-1]

            agent_history = filter_agents_tensor_novalidate(future_others, reverse=False)
            #print('targets', agent_history[1].shape, agent_history[1][ ..., 0])
            #print('targets2', agent_history[1][ ..., 1:3])
            #print('targets3', agent_history[0][ ..., -1])
            mask = torch.abs(agent_history[1][ ..., 1:3]).sum(-1)<0.001
            #print('targets3', mask)
            

            if agent_history[-1].shape[0] == 0:
                # Return zero array when there are no agents in the scene
                #agents_tensor: torch.Tensor = torch.zeros((0, len(agent_history)-1, self._agents_states_dim+1)).float()
                agents_tensor = np.zeros((0, len(agent_history)-1, self._agents_states_dim), dtype=np.float32)
                #agents_tensor = agents_tensor.detach().numpy()#[1:]
            else:

                padded_agent_states, mask = pad_agent_states_mask(agent_history, reverse=False)
                for frame in range(len(padded_agent_states)):
                     mask2 = torch.abs(padded_agent_states[frame][..., 1:3]).sum(-1)>0.001
                     mask[frame] = mask[frame]*mask2
                #print('targets2', padded_agent_states[0].shape, padded_agent_states[0][ ..., 0])
                local_coords_agent_states = convert_absolute_quantities_to_relative(padded_agent_states, anchor_ego_state)
                yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
                agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon) 
            #print('target', agents_tensor.shape)

                agents_tensor = agents_tensor.detach().numpy()[1:]
                # print('mask0', mask.shape, mask[0])
                mask = mask*mask[:1]
                mask = mask.detach().numpy()[1:]
                #print('mask', mask.shape, mask[-1])
                agents_tensor = np.stack([agents_tensor[..., AgentFeatureIndex.x()], agents_tensor[..., AgentFeatureIndex.y()], agents_tensor[..., AgentFeatureIndex.heading()], mask], -1)
                agents_tensor = agents_tensor.transpose(1,0,2)
                #print((agents_tensor[..., 1:3].max(-1)*agents_tensor[..., -1]).max())
            #print('target2', agents_tensor.shape)
            # agents_tensor = agents_tensor.transpose(1,0,2)
            # mask = agents_tensor[:, 0,-1]
            
            # print(mask, agents_tensor[mask<1][..., :2].shape)
            # print(agents_tensor[mask<1][..., :2])
            

            return MultipleAgentsTrajectories([agents_tensor])
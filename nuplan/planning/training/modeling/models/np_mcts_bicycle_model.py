"""
Copyright 2022 Motional
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.np_mcts_bicycle_tree import Tree
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model_utils import (
    LocalSubGraph,
    MultiheadAttentionGlobalHeadMulti,
    SinusoidalPositionalEmbedding,
    TypeEmbedding,
    pad_avails,
    pad_polylines,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import (
    GenericAgentsFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder_mcts import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.multiple_agents_trajectories import MultipleAgentsTrajectories
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.multiple_agents_trajectory_target_builder import (
    MultipleAgentsTrajectoryTargetBuilder,
)


warnings.filterwarnings("ignore")


@dataclass
class UrbanDriverOpenLoopModelParams:
    """
    Parameters for UrbanDriverOpenLoop model.
        local_embedding_size: embedding dimensionality of local subgraph layers.
        global_embedding_size: embedding dimensionality of global attention layers.
        num_subgraph_layers: number of stacked PointNet-like local subgraph layers.
        global_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it.
    """

    local_embedding_size: int
    global_embedding_size: int
    num_subgraph_layers: int
    global_head_dropout: float


@dataclass
class UrbanDriverOpenLoopModelFeatureParams:
    """
    Parameters for UrbanDriverOpenLoop features.
        feature_types: List of feature types (agent and map) supported by model. Used in type embedding layer.
        total_max_points: maximum number of points per element, to maintain fixed sized features.
        feature_dimension: feature size, to maintain fixed sized features.
        agent_features: Agent features to request from agent feature builder.
        ego_dimension: Feature dimensionality to keep from ego features.
        agent_dimension: Feature dimensionality to keep from agent features.
        max_agents: maximum number of agents, to maintain fixed sized features.
        past_trajectory_sampling: Sampling parameters for past trajectory.
        map_features: Map features to request from vector set map feature builder.
        max_elements: Maximum number of elements to extract per map feature layer.
        max_points: Maximum number of points per feature to extract per map feature layer.
        vector_set_map_feature_radius: The query radius scope relative to the current ego-pose.
        interpolation_method: Interpolation method to apply when interpolating to maintain fixed size map elements.
        disable_map: whether to ignore map.
        disable_agents: whether to ignore agents.
    """

    feature_types: dict[str, int]
    total_max_points: int
    feature_dimension: int
    agent_features: list[str]
    ego_dimension: int
    agent_dimension: int
    max_agents: int
    past_trajectory_sampling: TrajectorySampling
    map_features: list[str]
    max_elements: dict[str, int]
    max_points: dict[str, int]
    vector_set_map_feature_radius: int
    interpolation_method: str
    disable_map: bool
    disable_agents: bool

    def __post_init__(self) -> None:
        """
        Sanitize feature parameters.
        :raise AssertionError if parameters invalid.
        """
        if not self.total_max_points > 0:
            raise AssertionError(f"Total max points must be >0! Got: {self.total_max_points}")

        if not self.feature_dimension >= 2:
            raise AssertionError(f"Feature dimension must be >=2! Got: {self.feature_dimension}")

        # sanitize feature types
        for feature_name in ["NONE", "EGO"]:
            if feature_name not in self.feature_types:
                raise AssertionError(f"{feature_name} must be among feature types! Got: {self.feature_types}")

        self._sanitize_agent_features()
        self._sanitize_map_features()

    def _sanitize_agent_features(self) -> None:
        """
        Sanitize agent feature parameters.
        :raise AssertionError if parameters invalid.
        """
        if "EGO" in self.agent_features:
            raise AssertionError("EGO must not be among agent features!")
        for feature_name in self.agent_features:
            if feature_name not in self.feature_types:
                raise AssertionError(f"Agent feature {feature_name} not in feature_types: {self.feature_types}!")

    def _sanitize_map_features(self) -> None:
        """
        Sanitize map feature parameters.
        :raise AssertionError if parameters invalid.
        """
        for feature_name in self.map_features:
            if feature_name not in self.feature_types:
                raise AssertionError(f"Map feature {feature_name} not in feature_types: {self.feature_types}!")
            if feature_name not in self.max_elements:
                raise AssertionError(f"Map feature {feature_name} not in max_elements: {self.max_elements.keys()}!")
            if feature_name not in self.max_points:
                raise AssertionError(f"Map feature {feature_name} not in max_points types: {self.max_points.keys()}!")


@dataclass
class UrbanDriverOpenLoopModelTargetParams:
    """
    Parameters for UrbanDriverOpenLoop targets.
        num_output_features: number of target features.
        future_trajectory_sampling: Sampling parameters for future trajectory.
    """

    num_output_features: int
    future_trajectory_sampling: TrajectorySampling


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


class MCTSModel(TorchModuleWrapper):
    """
    Vector-based model that uses PointNet-based subgraph layers for collating loose collections of vectorized inputs
    into local feature descriptors to be used as input to a global Transformer.
    Adapted from L5Kit's implementation of "Urban Driver: Learning to Drive from Real-world Demonstrations
    Using Policy Gradients":
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py
    Only the open-loop  version of the model is here represented, with slight modifications to fit the nuPlan framework.
    Changes:
        1. Use nuPlan features from NuPlanScenario
        2. Format model for using pytorch_lightning
    """

    def __init__(
        self,
        model_params: UrbanDriverOpenLoopModelParams,
        feature_params: UrbanDriverOpenLoopModelFeatureParams,
        target_params: UrbanDriverOpenLoopModelTargetParams,
    ):
        """
        Initialize UrbanDriverOpenLoop model.
        :param model_params: internal model parameters.
        :param feature_params: agent and map feature parameters.
        :param target_params: target parameters.
        """
        agent_features = feature_params.agent_features
        agent_features2 = [*agent_features, "TRAFFIC_CONE", "GENERIC_OBJECT", "PEDESTRIAN"]
        super().__init__(
            feature_builders=[
                VectorSetMapFeatureBuilder(
                    map_features=feature_params.map_features,
                    max_elements=feature_params.max_elements,
                    max_points=feature_params.max_points,
                    radius=feature_params.vector_set_map_feature_radius,
                    interpolation_method=feature_params.interpolation_method,
                ),
                GenericAgentsFeatureBuilder(agent_features2, feature_params.past_trajectory_sampling),
            ],
            target_builders=[
                EgoTrajectoryTargetBuilder(target_params.future_trajectory_sampling),
                MultipleAgentsTrajectoryTargetBuilder(
                    future_trajectory_sampling=target_params.future_trajectory_sampling,
                ),
            ],
            future_trajectory_sampling=target_params.future_trajectory_sampling,
        )
        self._model_params = model_params
        self._feature_params = feature_params
        self._target_params = target_params
        self.dt = 0.5

        self.feature_embedding = nn.Linear(
            self._feature_params.feature_dimension,
            self._model_params.local_embedding_size,
        )
        self.positional_embedding = SinusoidalPositionalEmbedding(self._model_params.local_embedding_size)
        self.type_embedding = TypeEmbedding(
            self._model_params.global_embedding_size,
            self._feature_params.feature_types,
        )
        self.local_subgraph = LocalSubGraph(
            num_layers=self._model_params.num_subgraph_layers,
            dim_in=self._model_params.local_embedding_size,
        )
        if self._model_params.global_embedding_size != self._model_params.local_embedding_size:
            self.global_from_local = nn.Linear(
                self._model_params.local_embedding_size,
                self._model_params.global_embedding_size,
            )
        num_timesteps = self.future_trajectory_sampling.num_poses
        self.global_map = MultiheadAttentionGlobalHeadMulti(
            self._model_params.global_embedding_size,
            num_timesteps,
            self._target_params.num_output_features // num_timesteps,
            dropout=self._model_params.global_head_dropout,
        )
        self.global_head1 = MultiheadAttentionGlobalHeadMulti(
            self._model_params.global_embedding_size,
            num_timesteps,
            self._target_params.num_output_features // num_timesteps,
            dropout=self._model_params.global_head_dropout,
        )
        self.global_head2 = MultiheadAttentionGlobalHeadMulti(
            self._model_params.global_embedding_size,
            num_timesteps,
            self._target_params.num_output_features // num_timesteps,
            dropout=self._model_params.global_head_dropout,
        )
        self.final_output = nn.Linear(self._model_params.global_embedding_size, self._target_params.num_output_features)

        self.count = 0

    def extract_agent_features(
        self,
        ego_agent_features: GenericAgents,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ego and agent features into format expected by network and build accompanying availability matrix.
        :param ego_agent_features: agent features to be extracted (ego + other agents)
        :param batch_size: number of samples in batch to extract
        :return:
            agent_features: <torch.FloatTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element, feature_dimension>. Stacked ego, agent, and map features.
            agent_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        agent_features = []  # List[<torch.FloatTensor: max_agents+1, total_max_points, feature_dimension>: batch_size]
        agent_avails = []  # List[<torch.BoolTensor: max_agents+1, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            # Ego features
            # maintain fixed feature size through trimming/padding
            sample_ego_feature = ego_agent_features.ego[sample_idx][
                ...,
                : min(self._feature_params.ego_dimension, self._feature_params.feature_dimension),
            ].unsqueeze(0)
            if (
                min(self._feature_params.ego_dimension, GenericAgents.ego_state_dim())
                < self._feature_params.feature_dimension
            ):
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.feature_dimension, dim=2)

            sample_ego_avails = torch.ones(
                sample_ego_feature.shape[0],
                sample_ego_feature.shape[1],
                dtype=torch.bool,
                device=sample_ego_feature.device,
            )

            # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
            sample_ego_feature = torch.flip(sample_ego_feature, dims=[1])

            # maintain fixed number of points per polyline
            sample_ego_feature = sample_ego_feature[:, : self._feature_params.total_max_points, ...]
            sample_ego_avails = sample_ego_avails[:, : self._feature_params.total_max_points, ...]
            if sample_ego_feature.shape[1] < self._feature_params.total_max_points:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.total_max_points, dim=1)
                sample_ego_avails = pad_avails(sample_ego_avails, self._feature_params.total_max_points, dim=1)

            sample_features = [sample_ego_feature]
            sample_avails = [sample_ego_avails]

            # Agent features
            for feature_name in self._feature_params.agent_features:
                # if there exist at least one valid agent in the sample
                if ego_agent_features.has_agents(feature_name, sample_idx):
                    # num_frames x num_agents x num_features -> num_agents x num_frames x num_features
                    sample_agent_features = torch.permute(
                        ego_agent_features.agents[feature_name][sample_idx],
                        (1, 0, 2),
                    )
                    # maintain fixed feature size through trimming/padding
                    sample_agent_features = sample_agent_features[
                        ...,
                        : min(self._feature_params.agent_dimension, self._feature_params.feature_dimension),
                    ]
                    if (
                        min(self._feature_params.agent_dimension, GenericAgents.agents_states_dim())
                        < self._feature_params.feature_dimension
                    ):
                        sample_agent_features = pad_polylines(
                            sample_agent_features,
                            self._feature_params.feature_dimension,
                            dim=2,
                        )

                    sample_agent_avails = torch.ones(
                        sample_agent_features.shape[0],
                        sample_agent_features.shape[1],
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                    # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
                    sample_agent_features = torch.flip(sample_agent_features, dims=[1])

                    # maintain fixed number of points per polyline
                    sample_agent_features = sample_agent_features[:, : self._feature_params.total_max_points, ...]
                    sample_agent_avails = sample_agent_avails[:, : self._feature_params.total_max_points, ...]
                    if sample_agent_features.shape[1] < self._feature_params.total_max_points:
                        sample_agent_features = pad_polylines(
                            sample_agent_features,
                            self._feature_params.total_max_points,
                            dim=1,
                        )
                        sample_agent_avails = pad_avails(
                            sample_agent_avails,
                            self._feature_params.total_max_points,
                            dim=1,
                        )

                    # maintained fixed number of agent polylines of each type per sample
                    sample_agent_features = sample_agent_features[: self._feature_params.max_agents, ...]
                    sample_agent_avails = sample_agent_avails[: self._feature_params.max_agents, ...]
                    if sample_agent_features.shape[0] < (self._feature_params.max_agents):
                        sample_agent_features = pad_polylines(
                            sample_agent_features,
                            self._feature_params.max_agents,
                            dim=0,
                        )
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.max_agents, dim=0)

                else:
                    sample_agent_features = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        self._feature_params.feature_dimension,
                        dtype=torch.float32,
                        device=sample_ego_feature.device,
                    )
                    sample_agent_avails = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                # add features, avails to sample
                sample_features.append(sample_agent_features)
                sample_avails.append(sample_agent_avails)

            sample_features = torch.cat(sample_features, dim=0)
            sample_avails = torch.cat(sample_avails, dim=0)

            agent_features.append(sample_features)
            agent_avails.append(sample_avails)
        agent_features = torch.stack(agent_features)
        agent_avails = torch.stack(agent_avails)

        return agent_features, agent_avails

    def extract_map_features(
        self,
        vector_set_map_data: VectorSetMap,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract map features into format expected by network and build accompanying availability matrix.
        :param vector_set_map_data: VectorSetMap features to be extracted
        :param batch_size: number of samples in batch to extract
        :return:
            map_features: <torch.FloatTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element, feature_dimension>. Stacked map features.
            map_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        map_features = []  # List[<torch.FloatTensor: max_map_features, total_max_points, feature_dim>: batch_size]
        map_avails = []  # List[<torch.BoolTensor: max_map_features, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            sample_map_features = []
            sample_map_avails = []

            for feature_name in self._feature_params.map_features:
                coords = vector_set_map_data.coords[feature_name][sample_idx]
                tl_data = (
                    vector_set_map_data.traffic_light_data[feature_name][sample_idx]
                    if feature_name in vector_set_map_data.traffic_light_data
                    else None
                )
                avails = vector_set_map_data.availabilities[feature_name][sample_idx]

                # add traffic light data if exists for feature
                if tl_data is not None:
                    coords = torch.cat((coords, tl_data), dim=2)

                # maintain fixed number of points per map element (polyline)
                coords = coords[:, : self._feature_params.total_max_points, ...]
                avails = avails[:, : self._feature_params.total_max_points]

                if coords.shape[1] < self._feature_params.total_max_points:
                    coords = pad_polylines(coords, self._feature_params.total_max_points, dim=1)
                    avails = pad_avails(avails, self._feature_params.total_max_points, dim=1)

                # maintain fixed number of features per point
                coords = coords[..., : self._feature_params.feature_dimension]
                if coords.shape[2] < self._feature_params.feature_dimension:
                    coords = pad_polylines(coords, self._feature_params.feature_dimension, dim=2)

                sample_map_features.append(coords)
                sample_map_avails.append(avails)

            map_features.append(torch.cat(sample_map_features))
            map_avails.append(torch.cat(sample_map_avails))

        map_features = torch.stack(map_features)
        map_avails = torch.stack(map_avails)

        return map_features, map_avails

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        # Recover features
        features_init = features
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])

        batch_size = ego_agent_features.batch_size

        # Extract features across batch
        agent_features, agent_avails = self.extract_agent_features(ego_agent_features, batch_size)

        n_agents = agent_features.size()[1]
        xyh = agent_features[:, :, :1, :3]
        map_features, map_avails = self.extract_map_features(vector_set_map_data, batch_size)
        features = torch.cat([agent_features, map_features], dim=1)
        avails = torch.cat([agent_avails, map_avails], dim=1)

        # embed inputs
        feature_embedding = self.feature_embedding(features)

        # calculate positional embedding, then transform [num_points, 1, feature_dim] -> [1, 1, num_points, feature_dim]
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)

        # invalid mask
        invalid_mask = ~avails
        invalid_polys = invalid_mask.all(-1)

        # local subgraph
        embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, "global_from_local"):
            embeddings = self.global_from_local(embeddings)
        embeddings = F.normalize(embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5)
        embeddings = embeddings.transpose(0, 1)

        type_embedding = self.type_embedding(
            batch_size,
            self._feature_params.max_agents,
            self._feature_params.agent_features,
            self._feature_params.map_features,
            self._feature_params.max_elements,
            device=features.device,
        ).transpose(0, 1)

        # disable certain elements on demand
        if self._feature_params.disable_agents:
            invalid_polys[
                :,
                1 : (1 + self._feature_params.max_agents * len(self._feature_params.agent_features)),
            ] = 1  # agents won't create attention

        if self._feature_params.disable_map:
            invalid_polys[
                :,
                (1 + self._feature_params.max_agents * len(self._feature_params.agent_features)) :,
            ] = 1  # map features won't create attention

        invalid_polys[:, 0] = 0  # make ego always available in global graph

        agent_embeddings = embeddings[:n_agents]
        map_embeddings = embeddings[n_agents:]
        agent_types = type_embedding[:n_agents]
        map_types = type_embedding[n_agents:]
        agent_polys = invalid_polys[:, :n_agents]
        map_polys = invalid_polys[:, n_agents:]

        # global attention layers (transformer)
        map_embeddings, attns = self.global_map(map_embeddings, map_embeddings, map_types, map_polys, n_agents)
        map_embeddings = map_embeddings.transpose(0, 1)

        agent_embeddings, attns = self.global_head1(agent_embeddings, map_embeddings, map_types, map_polys, n_agents)
        agent_embeddings = agent_embeddings.transpose(0, 1)
        agent_embeddings, attns = self.global_head2(
            agent_embeddings,
            agent_embeddings,
            agent_types,
            agent_polys,
            n_agents,
        )

        outputs = self.final_output(agent_embeddings)
        outputs = outputs.view(
            batch_size,
            n_agents,
            self.future_trajectory_sampling.num_poses,
            self._target_params.num_output_features // self.future_trajectory_sampling.num_poses,
        )
        ego_pred = outputs[:, 0]
        outputs = torch.cat((xyh, outputs), dim=-2)

        outputs = (
            torch.nn.functional.interpolate(
                outputs[0].permute(0, 2, 1),
                size=16 * 5 + 1,
                mode="linear",
                align_corners=True,
            )
            .permute(0, 2, 1)
            .unsqueeze(0)
        )
        outputs = outputs[..., 1:, :]

        if 1:  # (time.perf_counter()-features_init['start_time']) < 0.8:
            batch_pos = []
            batch_acc = []
            batch_yr = []
            batch_speed = []
            batch_yaw = []
            agent_features = torch.flip(agent_features, dims=[2])
            agent_avails = torch.flip(agent_avails, dims=[2])
            ego_poses = agent_features[:, 0, :, :2]
            ego_speeds = torch.sqrt(((ego_poses[:, 1:] - ego_poses[:, :-1]) ** 2).sum(-1)) / 0.5

            if "current_speed" in features_init:
                ego_speeds[0, -1] = features_init["current_speed"]

            map = vector_set_map_data.coords["ROUTE_LANES"][0]  # map_features[:, :, :, :2]
            mask = vector_set_map_data.availabilities["ROUTE_LANES"][0]

            map_lane = vector_set_map_data.coords["LANE"][0]  # map_features[:, :, :, :2]
            mask_lane = vector_set_map_data.availabilities["LANE"][0]

            lights = vector_set_map_data.traffic_light_data["LANE"][0][:, -1, 2] > 0
            lights_route = vector_set_map_data.traffic_light_data["ROUTE_LANES"][0][:, -1, 2] > 0
            map = map[~lights_route][None]
            mask = mask[~lights_route][None]

            map_lane = map_lane[~lights][None]
            mask_lane = mask_lane[~lights][None]

            other_dim = ego_agent_features.agents["VEHICLE"][0][-1, :30, 6:8]
            other_dim = other_dim[None]

            traffic_cones = ego_agent_features.agents["TRAFFIC_CONE"][0][-1, :, :3]
            traffic_cones_dims = ego_agent_features.agents["TRAFFIC_CONE"][0][-1, :, 6:8]

            objects = ego_agent_features.agents["GENERIC_OBJECT"][0][-1, :, :3]
            objects_dims = ego_agent_features.agents["GENERIC_OBJECT"][0][-1, :, 6:8]

            pedestrians = ego_agent_features.agents["PEDESTRIAN"][0][-1, :, :3]
            pedestrians_dims = ego_agent_features.agents["PEDESTRIAN"][0][-1, :, 6:8]
            pedestrian_mask = pedestrians[..., 0] > 0
            pedestrians = pedestrians[pedestrian_mask]
            pedestrians_dims = pedestrians_dims[pedestrian_mask]

            static = torch.cat([traffic_cones, objects], 0)
            static_dims = torch.cat([traffic_cones_dims, objects_dims], 0)
            behind_mask_static = static[..., 0] > -2
            static = static[behind_mask_static]
            static_dims = static_dims[behind_mask_static]
            success = False
            for b in range(batch_size):
                other_agents = agent_features[b : b + 1, 1:]
                other_mask = agent_avails[b : b + 1, 1:]
                prediction = outputs[b : b + 1, 1:]
                all_time_mask = other_mask.amax(-1)

                other_agents_poses = other_agents[:, :, :, :2]  # batch, vehicle, time, positions
                speed_mask = other_mask[..., 1:] * other_mask[..., :-1]
                other_agents_speed = (
                    torch.sqrt(((other_agents_poses[:, :, 1:, :] - other_agents_poses[:, :, :-1, :]) ** 2).sum(-1))
                    / 0.5
                )
                other_agents_speed = other_agents_speed * speed_mask
                other_agents_max_speeds = other_agents_speed[:, 0, ...].amax(-1)
                other_agents_max_speeds = max(other_agents_max_speeds, 6)
                other_agents_max_speeds = min(other_agents_max_speeds, 10)

                other_agents = other_agents[all_time_mask][None]
                other_mask = other_mask[all_time_mask][None]
                prediction = prediction[all_time_mask][None]

                behind_mask = other_agents[:, :, -1, 0] > 0

                map = map[b : b + 1, :, :, :2]
                map_mask = mask[b : b + 1]
                lane_mask = map_mask.amax(-1)
                map = map[lane_mask][None]
                map_mask = map_mask[lane_mask][None]

                additional_map = map_lane[b : b + 1, :, :, :2]
                additional_map_mask = mask_lane[b : b + 1]
                lane_mask = additional_map_mask.amax(-1)
                additional_map = additional_map[lane_mask][None]
                additional_map_mask = additional_map_mask[lane_mask][None]

                batch_sample = {
                    "ego": agent_features[b : b + 1, 0].detach().cpu().numpy(),
                    "agents": other_agents.detach().cpu().numpy(),
                    "map": map.detach().cpu().numpy(),
                    "prediction": prediction[:, :, None].detach().cpu().numpy(),
                    "agents_mask": other_mask[..., None].detach().cpu().numpy(),
                    "map_mask": map_mask.detach().cpu().numpy(),
                    "additional_map": additional_map.detach().cpu().numpy(),
                    "additional_map_mask": additional_map_mask.detach().cpu().numpy(),
                    "agents_dim": other_dim.detach().cpu().numpy(),
                    "static_objects": static[None].detach().cpu().numpy(),
                    "static_objects_dims": static_dims[None].detach().cpu().numpy(),
                    "pedestrians": pedestrians[None].detach().cpu().numpy(),
                    "pedestrians_dims": pedestrians_dims[None].detach().cpu().numpy(),
                    "ego_pred": ego_pred.detach().cpu().numpy(),
                }
                batch_sample["ego_pos"] = ego_poses[b : b + 1].detach().cpu().numpy()
                batch_sample["ego_speed"] = ego_speeds[b : b + 1][..., None].detach().cpu().numpy()
                batch_sample["ego_yaw"] = agent_features[:, 0, :, 2][..., None].detach().cpu().numpy()
                batch_sample["max_speed"] = other_agents_max_speeds
                batch_sample["start_time"] = features_init["start_time"]

                if "speed_limit" in features_init and features_init["speed_limit"] is not None:
                    batch_sample["max_speed"] = min(features_init["speed_limit"], 13.4)
                    batch_sample["max_speed"] = min(batch_sample["max_speed"], other_agents_max_speeds + 1)

                tree = Tree(
                    batch_sample,
                    self,
                    None,
                    outputs,
                    action=features_init["action"],
                    pred_idx=0,
                    count=self.count,
                )
                (tree_acc, tree_yr), best_T = tree.simulate(k=256)
                tree_acc[:, 6] += (tree_acc.sum(-1) == 0) * 0.1
                tree_yr[:, 6] += (tree_yr.sum(-1) == 0) * 0.1
                batch_acc.append(tree_acc)
                batch_yr.append(tree_yr)

            self.count += 1
            tree_acc = np.stack(batch_acc, 0)
            tree_yr = np.stack(batch_yr, 0)
            tree_acc = torch.tensor(tree_acc[:, :], device=agent_features.device)
            tree_yr = torch.tensor(tree_yr[:, :], device=agent_features.device)

            initial_speed = ego_speeds[
                :,
                -1,
            ]
            initial_pos = torch.zeros((batch_size, 2), device=agent_features.device)
            initial_yaw = torch.zeros(batch_size, device=agent_features.device)
            trajectory = discrete_actions_to_trajectory(
                initial_speed,
                initial_pos,
                initial_yaw,
                tree_yr,
                tree_acc,
                0.1,
            )
        else:
            trajectory = outputs[:, 0]

        return {
            "trajectory": Trajectory(data=convert_predictions_to_trajectory(trajectory)),
            "agents_trajectories": MultipleAgentsTrajectories(outputs),
        }


def discrete_actions_to_trajectory(
    initial_speed,
    initial_pos,
    initial_yaw,
    discrete_steering,
    discrete_acc,
    dt,
):
    def to_acc(acc):
        return 3 * (acc - 13 // 2) / 6

    def to_steer(steering):
        return np.pi / 4 * (steering - 13 // 2) / 6

    new_discrete_acc = to_acc(torch.argmax(discrete_acc, -1))

    pred_speed = initial_speed + torch.cumsum(new_discrete_acc, -1) * dt
    pred_speed = torch.relu(pred_speed)

    new_discrete_steering = to_steer(torch.argmax(discrete_steering, -1))
    new_discrete_yaw_rate = pred_speed * torch.tan(new_discrete_steering) / 3.1
    pred_yaw = initial_yaw[..., None] + torch.cumsum(new_discrete_yaw_rate, -1) * dt
    yaw_vec = torch.stack([torch.cos(pred_yaw), torch.sin(pred_yaw)], -1)

    pred_velocity = yaw_vec * pred_speed[..., None]
    pred_pos = initial_pos.unsqueeze(-2) + torch.cumsum(pred_velocity, -2) * dt
    outputs = torch.cat((pred_pos, pred_yaw[..., None]), -1)

    return outputs

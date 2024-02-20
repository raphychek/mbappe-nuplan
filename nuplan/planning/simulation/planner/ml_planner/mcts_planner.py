import time
from typing import List, Optional, Type, cast

import numpy as np
import numpy.typing as npt

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

from nuplan.planning.metrics.utils.route_extractor import get_current_route_objects


class MCTSPlanner(AbstractPlanner):
    """
    Implements abstract planner interface.
    Used for simulating any ML planner trained through the nuPlan training framework.
    """

    def __init__(self, model: TorchModuleWrapper) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time
        self._num_output_dim = model.future_trajectory_sampling.num_poses

        self._model_loader = ModelLoader(model)

        self._initialization: Optional[PlannerInitialization] = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        # Propagate model
        predictions = self._model_loader.infer(features)

        # Extract trajectory prediction
        trajectory_predicted = cast(Trajectory, predictions['trajectory'])
        trajectory_tensor = trajectory_predicted.data
        trajectory = trajectory_tensor.cpu().detach().numpy()[0]  # retrive first (and only) batch as a numpy array

        return cast(npt.NDArray[np.float32], trajectory)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._model_loader.initialize()
        self._initialization = initialization

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        # Extract history
        initial_time = time.perf_counter()
        history = current_input.history
        current_speed =  current_input.history.current_state[0].dynamic_car_state.rear_axle_velocity_2d.x
        action = (current_input.history.current_state[0].dynamic_car_state.rear_axle_acceleration_2d.x, current_input.history.current_state[0].tire_steering_angle)
        #print(current_input.history.current_state[0].dynamic_car_state.rear_axle_acceleration_2d.x, current_input.history.current_state[0].tire_steering_angle)

        curr_route_obj = get_current_route_objects(self._initialization.map_api, current_input.history.current_state[0].center.point)
       # print()
        #print("Planner before features", time.perf_counter()-initial_time)
        # Construct input features
        start_time = time.perf_counter()
        features = self._model_loader.build_features(current_input, self._initialization)
        features['current_speed'] = current_speed
        features['action'] = action
        if curr_route_obj: features['speed_limit'] = curr_route_obj[0].speed_limit_mps
        self._feature_building_runtimes.append(time.perf_counter() - start_time)

        # Infer model
        start_time = time.perf_counter()
        features['start_time'] = initial_time
        t1 = time.time()
        predictions = self._infer_model(features)
        t2 = time.time()
        #print('PLanning time', t2-t1)
        self._inference_runtimes.append(time.perf_counter() - start_time)
        #print('Features + MCTS time', time.perf_counter() - initial_time, time.perf_counter() - start_time)

        # Convert relative poses to absolute states and wrap in a trajectory object.
        # states = transform_predictions_to_states(
        #     predictions, history.ego_states, self._future_horizon, self._step_interval
        # )
        states = transform_predictions_to_states(
            predictions, history.ego_states, 8, 0.1
        )
        trajectory = InterpolatedTrajectory(states)
        #if (time.perf_counter() - initial_time)>0.8:print('Total planning time', time.perf_counter() - initial_time)
        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []
        return report

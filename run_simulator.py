import os
import warnings
from pathlib import Path

import hydra
import numpy as np

from nuplan.planning.script.run_simulation import main as main_simulation


os.environ["USE_PYGEOS"] = "0"

# Location of path with all simulation configs
CONFIG_PATH = "nuplan/planning/script/config/simulation"
CONFIG_NAME = "default_simulation"

MODEL_PATH = "urban_multi_model.ckpt"

EXPERIMENT = "simulation"
PLANNER = "mcts_planner"
MODEL = "np_mcts_bicycle_model"
CHALLENGE = (
    "closed_loop_nonreactive_agents"  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
)
DATASET_PARAMS = [
    "scenario_builder=nuplan",  # use nuplan mini database
    "scenario_filter=nuplan_challenge_scenarios",  # initially select all scenarios in the database
    "scenario_filter.scenario_types=[starting_right_turn]",  # select scenario types
    "scenario_filter.num_scenarios_per_type=20",  # use 10 scenarios per scenario type
]
SAVE_DIR = Path("nuboard")  # optionally replace with persistent dir

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    # Run the simulation loop
    main_simulation(
        hydra.compose(
            config_name=CONFIG_NAME,
            overrides=[
                f"experiment_name={EXPERIMENT}",
                f"group={SAVE_DIR}",
                f"planner={PLANNER}",
                f"model={MODEL}",
                "planner.mcts_planner.model_config=${model}",  # hydra notation to select model config
                f"planner.mcts_planner.checkpoint_path={MODEL_PATH}",  # this path can be replaced by the checkpoint of the model trained in the previous section
                f"+simulation={CHALLENGE}",
                # "worker=sequential",
                *DATASET_PARAMS,
            ],
        ),
    )

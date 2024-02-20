import hydra

from nuplan.planning.script.run_nuboard import main as main_nuboard

experiment_path = ""

# Location of path with all nuBoard configs
CONFIG_PATH = "nuplan/planning/script/config/nuboard"
CONFIG_NAME = "default_nuboard"

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

main_nuboard(
    hydra.compose(
        config_name=CONFIG_NAME,
        overrides=[
            "port_number=5010",
            f"simulation_path={[experiment_path]}",  # nuboard file path(s), if left empty the user can open the file inside nuBoard
        ],
    )
)

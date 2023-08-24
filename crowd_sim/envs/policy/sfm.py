import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
import pysocialforce as psf

class SFM(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'SFM'
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.sim = None


    def configure(self, config):
        self.config_file = config
        return
    

    def predict(self, state):
        """
        Create a psf simulation at each time step and run one step
        PySocialForce: https://github.com/yuxiang-gao/PySocialForce/

        :param state:
        :return:
        """
        self_state = state.self_state

        # if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
        #     del self.sim
        #     self.sim = None
        if self.sim is None: # simulator has not been initialized
            initial_state = np.array(
                [
                    [self_state.px, self_state.py, self_state.vx, self_state.vy, self_state.gx, self_state.gy],
                ]
            )
            for human_state in state.human_states:
                initial_state = np.vstack([initial_state, np.array([human_state.px, human_state.py, human_state.vx, human_state.vy, human_state.px, human_state.py])])

            # initiate the simulator, hardcode the config file for now
            self.sim = psf.Simulator(
                initial_state,
                config_file="/home/vale/github/mpc-nav/PySocialForce/examples/example.toml",
            )
        else:
            initial_state = np.array(
                [
                    [self_state.px, self_state.py, self_state.vx, self_state.vy, self_state.gx, self_state.gy],
                ]
            )
            for human_state in state.human_states:
                initial_state = np.vstack([initial_state, np.array([human_state.px, human_state.py, human_state.vx, human_state.vy, human_state.px, human_state.py])])

            self.sim.peds.update(initial_state, None)

        self.sim.step()
        action = ActionXY(*self.sim.peds.vel()[0,:])

        return action

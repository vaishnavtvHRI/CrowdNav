from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.sfm import SFM


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['sfm'] = SFM
policy_factory['none'] = none_policy

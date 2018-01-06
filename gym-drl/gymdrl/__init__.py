from gym.envs.registration import register

register(
    id='LunarLanderTest-v0',
    entry_point='gymdrl.envs:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='Membrane-v1',
    entry_point='gymdrl.envs:Membrane',
    max_episode_steps=1000,
    reward_threshold=300,
)

register(
    id='MembraneWithoutLinkages-v1',
    entry_point='gymdrl.envs:MembraneWithoutLinkages',
    max_episode_steps=1000,
    reward_threshold=300,
)

register(
    id='MembraneJump-v0',
    entry_point='gymdrl.envs:MembraneJump',
    max_episode_steps=500,
    reward_threshold=300,
)

register(
    id='DoubleJoint-v0',
    entry_point='gymdrl.envs:DoubleJoint',
    max_episode_steps=1000,
    reward_threshold=500,
)

register(
    id='MembraneHardware-v0',
    entry_point='gymdrl.envs:MembraneHardware',
    max_episode_steps=300,
    reward_threshold=300,
)

register(
    id='MembraneCalibration-v0',
    entry_point='gymdrl.envs:MembraneCalibration',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='HardwareSimulation-v0',
    entry_point='gymdrl.envs:HardwareSimulation',
    max_episode_steps=300,
    reward_threshold=300,
)
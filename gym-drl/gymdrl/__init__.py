from gym.envs.registration import register

register(
    id='LunarLanderTest-v0',
    entry_point='gymdrl.envs:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)


# HARDWARE MEMBRANE ENVIRONMENT
register(
    id='MembraneHardware-v0',
    entry_point='gymdrl.envs:MembraneHardware',
    max_episode_steps=300,
    reward_threshold=300,
)

# SIMULATED ENVIRONMENTS
register(
    id='MembraneBasket-v0',
    entry_point='gymdrl.envs:MembraneBasket',
    max_episode_steps=500,
    reward_threshold=300,
)
register(
    id='MembraneCalibration-v0',
    entry_point='gymdrl.envs:MembraneCalibration',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='MembraneJump-v0',
    entry_point='gymdrl.envs:MembraneJump',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='MembraneOrder-v0',
    entry_point='gymdrl.envs:MembraneOrder',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='MembraneRotate-v0',
    entry_point='gymdrl.envs:MembraneRotate',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='MembraneStack-v0',
    entry_point='gymdrl.envs:MembraneStack',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='MembraneTarget-v0',
    entry_point='gymdrl.envs:MembraneTarget',
    max_episode_steps=300,
    reward_threshold=300,
)

register(
    id='MembraneMoveArb-v0',
    entry_point='gymdrl.envs:MembraneMoveArb',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='HardwareRand-v0',
    entry_point='gymdrl.envs:MembraneHardwareRand',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='HardwareClick-v0',
    entry_point='gymdrl.envs:MembraneHardwareClick',
    max_episode_steps=300,
    reward_threshold=300,
)
register(
    id='HardwareRandPoscontrol-v0',
    entry_point='gymdrl.envs:MembraneHardwareRandPoscontrol',
    max_episode_steps=300,
    reward_threshold=300,
)

from gym.envs.registration import register

register(
    id='random1-v0',
    entry_point='cheeta_gym.envs:Random1Env',
)

register(
    id='random2-v0',
    entry_point='cheeta_gym.envs:Random2Env',
)

register(
    id='racetrack-v0',
    entry_point='cheeta_gym.envs:RaceTrackEnv',
)

register(
    id='stairs-v0',
    entry_point='cheeta_gym.envs:StairsEnv',
)

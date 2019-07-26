from gym.envs.registration import register

register(
    id='toio-v0',
    entry_point='gym_toio.envs:ToioEnv',
)
register(
    id='toiofewactions-v0',
    entry_point='gym_toio.envs:ToiofewactionsEnv', 
)

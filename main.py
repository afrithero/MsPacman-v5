from dqn_agent_atari import AtariDQNAgent

if __name__ == "__main__":
    config = {
        "env_id": 'ALE/MsPacman-v5',
        "gpu": True,
        "algo": "dqn", # "dqn" | "ddqn"
        "model_type": "dueling", # "dqn" | "dueling"
        "training_steps": 1e8,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_min": 0.1,
        "warmup_steps": 20000,
        "eps_decay": 1000000,
        "eval_epsilon": 0.01,
        "replay_buffer_capacity": 100000,
        "update_freq": 4,
        "update_target_freq": 10000,
        "learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 1,
        "logdir": 'log/Dueling/',

    }

    agent = AtariDQNAgent(config)
    # agent.load_and_evaluate("log/DDQN/model_6695502_4800.pth")
    agent.train()
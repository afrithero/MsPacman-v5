import torch

# 1. get Q(s,a) from behavior net
# 2. get max_a Q(s',a) from target net
# 3. calculate Q_target = r + gamma * max_a Q(s',a)
# 4. calculate loss between Q(s,a) and Q_target
# 5. update behavior net
@torch.no_grad()
def td_target_dqn(batch_size, behavior_net, target_net, next_state, reward, done, gamma):
    q_next = target_net(next_state)
    q_next = q_next.max(dim=1).values.view(batch_size, 1)
    q_target = torch.where(done == 0, reward + gamma * q_next, reward)

    return q_target

@torch.no_grad()
def td_target_ddqn(batch_size, behavior_net, target_net, next_state, reward, done, gamma):
    q_next_esti = behavior_net(next_state) 
    q_next = target_net(next_state)
    q_next_action = torch.argmax(q_next_esti, dim=1).unsqueeze(1) # use q_next_esti to decide next action
    q_next = q_next.gather(1, q_next_action)  
    q_target = torch.where(done == 0, reward + gamma * q_next, reward)

    return q_target

TD_TARGETS = {
    "dqn": td_target_dqn,
    "ddqn": td_target_ddqn,
}
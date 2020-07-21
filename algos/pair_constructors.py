from abc import ABC, abstractmethod

"""
Pair Constructors turn a basic trajectory dataset into a dataset of `context`, `target`, and `extra_context` data 
elements, along with a metadata tensor containing the trajectory ID and timestep ID for each element in the dataset. 
The `context` element is conceptually thought of as the element you're using to do prediction, whereas the `target` 
is the ground truth or "positive" we're trying to predict from the context, though this prediction framework is 
admittedly a somewhat fuzzy match to the actual variety of techniques.

- In temporal contrastive loss settings, context is generally the element at position (t), and target the element at
position (t+k) 
- In pure-augmentation contrastive loss settings, context and target are the same element (which will be augmented 
in different ways) 
- In a VAE, context and target are also the same element. Context will be mapped into a representation and then decoded
back out, whereas the target will "tag along" as ground truth pixels needed to calculate the loss.  
- In Dynamics modeling, context is the current state at time (t), target is the state at time (t+1) and extra context
is the action taken at time (t) 
"""



class TargetPairConstructor(ABC):

    @abstractmethod
    def __call__(self, data_dict):
        pass


class IdentityPairConstructor(TargetPairConstructor):
    def __call__(self, data_dict):
        obs, actions, dones = data_dict['states'], data_dict['actions'], data_dict['dones']
        dataset = []
        trajectory_ind = timestep = 0
        for i in range(len(dones)):
            dataset.append({'context': obs[i], 'target': obs[i], 'extra_context': [], 'traj_ts_ids': [trajectory_ind, timestep]})
            timestep += 1
            if dones[i]:
                trajectory_ind += 1
                timestep = 0
        return dataset


class TemporalOffsetPairConstructor(TargetPairConstructor):
    #TODO generalize this to offsets of arbitrary k rather than 1
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, data_dict):
        obs, actions, dones = data_dict['states'], data_dict['actions'], data_dict['dones']
        dataset = []
        trajectory_ind = timestep = 0
        for i in range(len(dones)):
            if dones[i] or len(dones) == i + 1:
                # If dones[i] is true, next obs is from new trajectory, skip; a bit hacky, but fine for prototype
                trajectory_ind += 1
                timestep = 0
                continue

            if self.mode is None:
                dataset.append({'context': obs[i], 'target': obs[i + 1],
                                'extra_context': [], 'traj_ts_ids': [trajectory_ind, timestep]})
            elif self.mode == 'dynamics':
                dataset.append({'context': obs[i], 'target': obs[i + 1],
                                'extra_context': actions[i], 'traj_ts_ids': [trajectory_ind, timestep]})
            elif self.mode == 'inverse_dynamics':
                dataset.append({'context': obs[i], 'target': actions[i],
                                'extra_context': obs[i+1], 'traj_ts_ids': [trajectory_ind, timestep]})
            timestep += 1
        return dataset



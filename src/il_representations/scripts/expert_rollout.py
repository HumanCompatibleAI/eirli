import pickle
import pdb

if __name__ == "__main__":
    with open("demos/cartpole_rollout.pkl", 'rb') as fp:
        cartpole_rollout = pickle.load(fp)
    pdb.set_trace()
from algos import *

ALGOS_LIST = [el for el in globals() if issubclass(el, RepresentationLearner)]


if __name__ == "__main__":
    print(ALGOS_LIST)
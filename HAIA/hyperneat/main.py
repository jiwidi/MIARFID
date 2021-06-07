import neat
import os
import logging
try:
   import cPickle as pickle
except:
   import pickle
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_hyper
from pureples.hyperneat.hyperneat import create_phenotype_network

# Network input, hidden and output coordinates.
input_coordinates = []
for i in range(210):
    for j in range(160):
        input_coordinates.append((i, j))
hidden_coordinates = [[(-0.5, 0.5), (0.5, 0.5)], [(-0.5, -0.5), (0.5, -0.5)]]
output_coordinates = [(-1., 1.), (1., 1.)]
activations = len(hidden_coordinates) + 2

sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)

# Config for CPPN.
local_dir = os.getcwd()
config_file = os.path.join(local_dir, 'config.ini')
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            config_file)


# Use the gym_runner to run this experiment using HyperNEAT.
def run(gens, env):
    winner, stats = run_hyper(gens, env, 10000, config, sub, activations)
    print("hyperneat_breakout done") 
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make("Breakout-v0")

    # Run!
    winner = run(10, env)[0]

    # Save CPPN if wished reused and draw it + winner to file.
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    winner_net = create_phenotype_network(cppn, sub)
    draw_net(cppn, filename="hyperneat_breakout_cppn")
    with open('hyperneat_breakout_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)
    draw_net(winner_net, filename="hyperneat_breakout_winner")
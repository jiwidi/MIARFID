import os
import argparse
import numpy as np
import gym
import neat
import visualize
import pickle
import cv2

# ***** GLOBAL PARAMETERS *****
args = None

# ********* GYM *********
env_name = 'CartPole-v1'
env = gym.make(env_name)


# ******** NEAT *********
local_dir = os.getcwd()
out_dir = os.path.join(local_dir, 'out')
config_file = os.path.join(local_dir, 'config.ini')

# *****************************

def flatten_input(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.ndarray.flatten(img)
    return img


def run_net(net, episodes=1, steps=5000, render=False):
    fitness_list = list()
    for runs in range(episodes):
        inputs = env.reset()
        
        # This is for breakout with RGB image
        #inputs = flatten_input(inputs)

        total_reward = 0.0
        for j in range(steps):
            outputs = net.activate(inputs)
            action = np.argmax(outputs)
            inputs, reward, done, _ = env.step(action)

            # This is for breakout with RGB image
            #inputs = flatten_input(inputs)

            if render:
                env.render()
            if done:
                break
            total_reward += reward

        fitness_list.append(total_reward)

    fitness = np.array(fitness_list).mean()
    #print("Species fitness: %s" % str(fitness))
    return fitness

def worker_eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    #net = neat.nn.create_feed_forward_phenotype(genome)
    return run_net(net, args.episodes, args.max_steps, render=args.render)


def eval_fitness(net):
    """
    Evaluates fitness of the genome that was used to generate
    provided net
    Arguments:
        net: The feed-forward neural network generated from genome
    Returns:
        The fitness score - the higher score the means
        the better fit organism.
     """
    return run_net(net, args.episodes, args.max_steps, render=args.render)

def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitness = eval_fitness(net)
            genome.fitness = fitness

def start_simulation():
    # Print input and output sizes to use them in the config file
    print("Input Size: %s" % str(len(env.observation_space.high)))
    print("Output Size: %s" % str(env.action_space.n))

    config = neat.Config(neat.DefaultGenome,
        neat.DefaultReproduction, neat.DefaultSpeciesSet,
        neat.DefaultStagnation, config_file)
    population = neat.population.Population(config)

    # Add some reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5,
        filename_prefix='out/neat-checkpoint-'))

    # If checkpoint load checkpoint file
    if args.checkpoint:
        population = neat.Checkpointer().restore_checkpoint(args.checkpoint)

    # Run the simulation of all individuals in the population
    if args.render:
        best_genome = population.run(eval_genomes, args.generations)
    else:
        distributed_training = neat.parallel.ParallelEvaluator(args.workers, worker_eval_genome)
        best_genome = population.run(distributed_training.evaluate, args.generations)

    print('\nBest genome:\n{!s}'.format(best_genome))


    # Plot simulation stats
    node_names = {-1:'Cart Position', -2: 'Cart Velocity',
                  -3: 'Pole Angle', -4: 'Pole Angular Velocity', 0:'Push Left', 1: 'Push Right'}
    
    visualize.draw_net(config, best_genome, True,
    node_names=node_names, directory=out_dir)
    #
    visualize.plot_stats(stats, ylog=False, view=True,
    filename=os.path.join(out_dir, 'avg_fitness.svg'))
    
    visualize.plot_species(stats, view=True,
    filename=os.path.join(out_dir, 'speciation.svg'))

    # Run best genome
    best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    iterations = 100  # Number of iterations needed to check if problem is solved
    rewards = np.zeros(iterations)
    for i in range(iterations):
        rewards[i] = run_net(best_net, 1, args.max_steps, render=False)
    print("Average reward over %i iterations: %.1d" % (iterations, rewards.mean()))

    # Save best genome
    with open('best_genome.pkl', 'wb') as output:
        pickle.dump(best_genome, output, 1)



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='NEAT & OPENAI GYM Implementation')
    parser.add_argument('--max-steps', dest='max_steps', type=int, default=1000,
                        help='Max number of steps per genome')
    parser.add_argument('--episodes', type=int, default=1,
                        help="The number of times to run a single genome. This takes the fitness score from the worst run")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--generations', type=int, default=50,
                        help="The number of generations to evolve each network")
    parser.add_argument('--checkpoint', type=str,
                        help="Use a checkpoint to run the simulation")
    parser.add_argument('--workers', dest="workers", type=int, default=4,
                        help="Number of threads to use on a parallel execution")
    args = parser.parse_args()

    start_simulation()

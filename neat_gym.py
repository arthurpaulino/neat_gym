# -*- coding: utf-8 -*-
import numpy as np
from util import *
import neat
import gym
import sys
import os

################################################################################################

if len(sys.argv) != 2:
	print 'usage:\n\t$ python neat_gym.py experiment_file_name'
	exit()
try:
	exp = Exp().parse(sys.argv[1])
except ValueError as err:
    print(err.args[0])
    exit()

################################################################################################

def worker_evaluate_genome(g, config):
	fitnesses = []
	knowledge = []
	for run in range(exp.episodes):
		#attain knowledge from the first run, only
		fitnesses.append(evaluate_net(exp.task, g.net, env, exp.timeout, knowledge, run == 0, exp.syllabus_source, exp.syllabus_size))
	fitness = np.array(fitnesses).mean()
	return fitness, knowledge

def train_network(env, pe):

	pop = CustomPopulation(config)

	#start evolution
	best_fitnesses = []
	pop.run(pe.evaluate, best_fitnesses, exp)

	#commit statistics
	DataManager(exp).commit(best_fitnesses)

################################################################################################

env = gym.make(exp.task)
config = neat.Config(CustomGenome, CustomReproduction,	neat.DefaultSpeciesSet, neat.DefaultStagnation, exp.task)
config.compatibility_threshold = 3.0
if exp.learning_function != 'N/A':
	#preventing speciation boom
	config.genome_config.compatibility_weight_coefficient = 0.0
pe = CustomParallelEvaluator(exp.cores, worker_evaluate_genome)

for i in range(exp.repetitions):
	print 'Experiment: {}. Progress: {}/{}'.format(sys.argv[1], i+1, exp.repetitions)
	train_network(env, pe)

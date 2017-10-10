# -*- coding: utf-8 -*-
#import sys; sys.dont_write_bytecode = True
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
exp = Exp()
exp.parse(sys.argv[1])

################################################################################################

def worker_evaluate_genome(g, config):
	fitnesses = []
	knowledge = []
	for run in range(exp.epis):
		#attain knowledge on the first run, only
		fitnesses.append(evaluate_net(exp.task, g.net, env, exp.timeout, knowledge, run == 0))
	fitness = np.array(fitnesses).mean()
	return fitness, knowledge

def train_network(env, pe):

	pop = CustomPopulation(config)
		
	#start evolution
	best_fitnesses = []
	pop.run(pe.evaluate, exp.gens, best_fitnesses, exp.lf, exp.la, exp.lr)
#	TODO	
#	pop.run(pe.evaluate, exp.gens, best_fitnesses, exp.lf, exp.la, exp.lr, exp.lt, exp.li)
	
	#commit statistics
	DataManager(exp.task).commit(exp.lf, exp.la, exp.lr, exp.lt, exp.li, best_fitnesses)
		
################################################################################################

env = gym.make(exp.task)
config = neat.Config(CustomGenome, CustomReproduction,	neat.DefaultSpeciesSet, neat.DefaultStagnation, exp.task)
pe = CustomParallelEvaluator(exp.cores, worker_evaluate_genome)

for i in range(exp.reps):
	print '----------==========##########Experiment {}/{}:'.format(i+1, exp.reps)
	train_network(env, pe)

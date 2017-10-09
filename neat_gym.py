# -*- coding: utf-8 -*-
#import sys; sys.dont_write_bytecode = True
import numpy as np
from util import *
import argparse
import neat
import gym
import os

################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--task',    type=str,   default='CartPole-v0', help='The gym task to be performed')
parser.add_argument('--cores',   type=int,   default=4,    help="The number cores on your computer for parallel execution")
parser.add_argument('--timeout', type=int,   default=200,  help='The max number of steps to take per genome')
parser.add_argument('--epis',    type=int,   default=3,    help='The number or evaluations to compute mean fitness')
parser.add_argument('--gens',    type=int,   default=25,   help="The number of generations to reach")
parser.add_argument('--reps',    type=int,   default=1,    help="The number or experiment repetitions")
parser.add_argument('--lf',      type=str,   default='na', help="The learning function. It can be either na (not applicable), bp (backpropagation) or bt (batch)")
parser.add_argument('--lp',      type=float, default=0.0,  help="The proportion of the master's experience to be taught")
parser.add_argument('--lr',      type=float, default=0.0,  help="The learning rate parameter")
args = parser.parse_args()

################################################################################################

def worker_evaluate_genome(g, config):
	fitnesses = []
	knowledge = []
	for run in range(args.epis):
		if run == 0:
			#knowledge obtained from the first run, only
			fitnesses.append(evaluate_net(args.task, g.net, env, args.timeout, knowledge))
		else:
			fitnesses.append(evaluate_net(args.task, g.net, env, args.timeout, []))
	fitness = np.array(fitnesses).mean()
	return fitness, knowledge

def train_network(env, pe, learning_function):

	pop = CustomPopulation(config)
		
	#start evolution
	best_fitnesses = []
	pop.run(pe.evaluate, args.gens, best_fitnesses, learning_function, args.lp, args.lr)
	
	#commit statistics
	if learning_function is None or args.lp == 0.0 or args.lr == 0.0:
		#regular NEAT evolution, record data on both databases
		DataManager(args.task+'_bp').commit(0.0, 0.0, best_fitnesses)
		DataManager(args.task+'_bt').commit(0.0, 0.0, best_fitnesses)
	else:
		#has learning involved
		DataManager(args.task+'_'+args.lf).commit(args.lp, args.lr, best_fitnesses)
	
################################################################################################

if args.lf == 'bp':
	learning_function = backpropagation
elif args.lf == 'bt':
	learning_function = batch
elif args.lf == 'na':
	learning_function = None
else:
	print 'Unknown learning function'
	exit()

env = gym.make(args.task)
config = neat.Config(CustomGenome, CustomReproduction,	neat.DefaultSpeciesSet, neat.DefaultStagnation, args.task)
pe = CustomParallelEvaluator(args.cores, worker_evaluate_genome)

for i in range(args.reps):
	print '----------==========##########Experiment {}/{}:'.format(i+1, args.reps)
	train_network(env, pe, learning_function)

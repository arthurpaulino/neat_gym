# -*- coding: utf-8 -*-
from multiprocessing import Pool
from neat import Population, ParallelEvaluator, DefaultGenome, DefaultReproduction
from neat.six_util import iteritems, itervalues, iterkeys
from neat.nn import FeedForwardNetwork
from neat.reporting import ReporterSet
from neat.math_util import mean
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import ConfigParser
import numpy as np
import random
import pickle
import math
import gzip
import glob
import os

############################### EXPERIMENT ####################################

class Exp():
	def __init__(self):
		self.cores = 4
		self.task = 'CartPole-v1'
		self.timeout = 500
		self.episodes = 3
		self.generations = 25
		self.repetitions = 2
		self.learning_function = 'N/A'
		self.learning_rate = 0.0
		self.learning_target = 'N/A'
		self.children_inclusion = 'N/A'
		self.syllabus_source = 'EXP'
		self.syllabus_size = 30
	
	def parse(self, filename):
		config = ConfigParser.ConfigParser()
		config.read(filename)
		if config.has_option('exp', 'cores'):
			self.cores = int(config.get('exp', 'cores'))
		if config.has_option('exp', 'task'):
			self.task = config.get('exp', 'task')
		if config.has_option('exp', 'timeout'):
			self.timeout = int(config.get('exp', 'timeout'))
		if config.has_option('exp', 'episodes'):
			self.episodes = int(config.get('exp', 'episodes'))
		if config.has_option('exp', 'generations'):
			self.generations = int(config.get('exp', 'generations'))
		if config.has_option('exp', 'repetitions'):
			self.repetitions = int(config.get('exp', 'repetitions'))
		if config.has_option('exp', 'learning_function'):
			self.learning_function = config.get('exp', 'learning_function')
		if config.has_option('exp', 'learning_rate'):
			self.learning_rate = float(config.get('exp', 'learning_rate'))
		if config.has_option('exp', 'learning_target'):
			self.learning_target = config.get('exp', 'learning_target')
		if config.has_option('exp', 'children_inclusion'):
			self.children_inclusion = config.get('exp', 'children_inclusion')
		if config.has_option('exp', 'syllabus_source'):
			self.syllabus_source = config.get('exp', 'syllabus_source')
		if config.has_option('exp', 'syllabus_size'):
			self.syllabus_size = int(config.get('exp', 'syllabus_size'))
		return self

############################# DATA MANAGEMENT #################################

class DataManager():
	def __init__(self, inpt):
		if isinstance(inpt, Exp):
			self.basename = inpt.task
			self.filename = self.basename+'.db'
			self.exp = inpt
		elif isinstance(inpt, basestring):
			self.basename = inpt
			self.filename = self.basename+'.db'
	
	def commit(self, values_list):
		key = (self.exp.learning_function,self.exp.learning_rate,self.exp.learning_target,self.exp.children_inclusion,self.exp.syllabus_source,self.exp.syllabus_size)
		raw = self.querry_raw()
		if not raw.has_key(key):
			raw[key] = []
		counter = 0
		for value in values_list:
			if len(raw[key]) == counter:
				raw[key].append([value])
			else:
				raw[key][counter].append(value)
			counter+=1
		with gzip.open(self.filename, 'w') as f:
			pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
	
	def querry_raw(self):
		if os.path.isfile(self.filename):
			with gzip.open(self.filename) as f:
				return pickle.load(f)
		return {}
	
	def querry(self):
		d = {}
		raw = self.querry_raw()
		for key in raw:
			d[key] = ([],[])
			for values in raw[key]:
				array = np.array(values)
				d[key][0].append(array.mean())
				d[key][1].append(array.std())
		return d
	
	colors = ['blue', 'red', 'lime', 'orange', 'purple']
	def plot(self, keys_to_plot):
		if len(keys_to_plot) > len(self.colors):
			return
		ordered_keys_to_plot = list(keys_to_plot)
		ordered_keys_to_plot.sort(key=lambda tup: (tup[1], tup[2], tup[3], tup[4], tup[5]))
		ordered_keys_to_plot.sort(key=lambda tup: tup[0], reverse=True)
		data = self.querry()
		patches = []
		for key, color in zip(ordered_keys_to_plot, self.colors):
			(means, stds) = data[key]
			plt.plot(range(len(means)), means, color = color, alpha=0.8)
			plt.plot(range(len(stds)), stds, linestyle = '--', color = color, alpha=0.8)
			if key[0] == 'N/A':
				patches.append(mpatches.Patch(color=color, label='NEAT Regular'))
			else:
				label = u"Método: " + key[0] + '; Taxa: ' + str(key[1]) + '; Alvo: ' + key[2] + u"; Inclusão: " + key[3] + u"; Lições: " + str(key[4]) + '; Fonte: ' + str(key[5])
				patches.append(mpatches.Patch(color=color, label=label))

		
		continuous_line = mlines.Line2D([], [], label=u"Média dos melhores desempenhos", color = 'black')
		dashed_line = mlines.Line2D([], [], label=u"Desvio padrão", linestyle = '--', color = 'black')
		lines_legend = plt.legend(handles=[continuous_line, dashed_line], loc='upper left', fontsize = 'x-small')
		
		plt.legend(handles=patches, bbox_to_anchor=(0, 1, 1, 0), loc='lower center', title = 'Experimentos', fontsize = 'x-small')
		plt.gca().add_artist(lines_legend)

		plt.xlabel(u"Geração")
		plt.ylabel(u"Aptidão")
		plt.savefig(self.basename+'.png', bbox_inches='tight')
		
	def remove(self, keys):
		raw = self.querry_raw()
		for key in keys:
			if raw.has_key(key):
				del raw[key]
		with gzip.open(self.filename, 'w') as f:
			pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	def db_selection_menu():
		os.system('clear')
		db_names = []
		for raw_db_name in glob.glob('*.db'):
			db_names.append(raw_db_name.split('.db')[0])
		for i, db_name in zip(range(len(db_names)), db_names):
			print '{}. {}'.format(i+1, db_name)
		print '{}. Exit'.format(len(db_names)+1)
		try:
			i = input(": ")
			if i == len(db_names)+1:
				return
			db = DataManager(db_names[i-1])
			all_keys = []
			for key in db.querry_raw():
				all_keys.append(key)
			all_keys.sort(key=lambda tup: (tup[1], tup[2], tup[3], tup[4], tup[5]))
			all_keys.sort(key=lambda tup: tup[0], reverse=True)
			action_selection_menu(db, all_keys)
		except:
			db_selection_menu()
	
	def action_selection_menu(db, selected_keys):
		os.system('clear')
		print db.basename
		keys = []
		raw = db.querry_raw()
		for key in raw:
			keys.append(key)
		for i, key in zip(range(len(keys)), keys):
			if key in selected_keys:
				print '{}. +{}: {} experiment(s)'.format(i+1, key, len(raw[key][0]))
			else:
				print '{}. -{}: {} experiment(s)'.format(i+1, key, len(raw[key][0]))
		print '{}. Plot ({} colors available)'.format(len(keys)+1, len(db.colors))
		print '{}. Remove'.format(len(keys)+2)
		print '{}. Return'.format(len(keys)+3)
		print '{}. Exit'.format(len(keys)+4)
		try:
			i = input(": ")
			if i>0 and i<=len(keys):
				if keys[i-1] in selected_keys:
					selected_keys.remove(keys[i-1])
				else:
					selected_keys.append(keys[i-1])
				action_selection_menu(db, selected_keys)
			elif i == len(keys)+1:
				if len(selected_keys) > 0:
					db.plot(selected_keys)
				action_selection_menu(db, selected_keys)
			elif i == len(keys)+2:
				s = ''
				for key in selected_keys:
					s+=str(key)+' '
				print 'Removing keys {}'.format(s)
				answer = raw_input('Confirm (yes)? ')
				if answer == 'yes':
					db.remove(selected_keys)
					action_selection_menu(db, set())
				else:
					action_selection_menu(db, selected_keys)
			elif i == len(keys)+3:
				db_selection_menu()
			elif i == len(keys)+4:
				return
			else:
				action_selection_menu(db, selected_keys)
		except:
			action_selection_menu(db)
	
	db_selection_menu()

############################# LEARNING METHODS ################################

#-------------------------------Backpropagation-------------------------------#

def update(node, back_links, node_eval_pos, learning_rate, g, error):
	for i in range(len(back_links[node])):
		#updating network weights
		back_links[node][i] = (back_links[node][i][0], back_links[node][i][1] + learning_rate*error[node]*g.net.values[node])
		#updating genome weights
		g.connections[(back_links[node][i][0], node)].weight = back_links[node][i][1]
	#updating network bias
	lst = list(g.net.node_evals[node_eval_pos[node]])
	lst[3] += learning_rate*error[node]
	g.net.node_evals[node_eval_pos[node]] = tuple(lst)
	#updating genome bias
	g.nodes[node].bias = lst[3]

def recursion(node, error, front_neighbours, g, back_links, node_eval_pos, learning_rate):
	if error[node] is None:
		error[node] = 0.0
		if front_neighbours.has_key(node):
			for front_neighbour in front_neighbours[node]:
				error[node] += 5.0*g.net.values[node]*(1.0-
					g.net.values[node])*recursion(front_neighbour, error, front_neighbours,
						g, back_links, node_eval_pos, learning_rate)*g.connections[(node, front_neighbour)].weight
			update(node, back_links, node_eval_pos, learning_rate, g, error)
	return error[node]

def backpropagation(g, lessons, learning_rate):
	#setting up for backpropagation
	hidden_nodes = []
	output_nodes = []
	back_links = {}
	front_neighbours = {}
	node_eval_pos = {}
	error = {}
	i = 0
	for node, _, _, _, _, links in g.net.node_evals:
		node_eval_pos[node] = i
		back_links[node] = links
		for link in links:
			if not front_neighbours.has_key(link[0]):
				front_neighbours[link[0]] = [node]
			else:
				front_neighbours[link[0]].append(node)
		if node < len(g.net.output_nodes):
			output_nodes.append(node)
		else:
			hidden_nodes.append(node)
		i += 1
	
	for lesson in lessons:
		#computing the output layer
		(lesson_input, lesson_output) = lesson
		outputs = g.net.activate(lesson_input)
		for node in output_nodes:
			#y = 1/(1+exp(-5x)) => y' = 5y(1-y)
			error[node] = 5.0*outputs[node]*(1.0-outputs[node])*(lesson_output[node]-outputs[node])
			update(node, back_links, node_eval_pos, learning_rate, g, error)
			
		#computing hidden nodes
		for node in hidden_nodes:
			error[node] = None
		for node in hidden_nodes:
			recursion(node, error, front_neighbours, g, back_links, node_eval_pos, learning_rate)
	
#------------------------------------Batch------------------------------------#

def np_solve(X,Y):
	X = np.matrix(X)
	Xt = X.transpose()
	Y = np.matrix(Y).transpose()
	try:
		return np.linalg.inv(Xt*X)*Xt*Y
	except:
		return None

def batch(g, lessons, learning_rate):
	#w = {[(X^T)(X)]^(-1)}(X^T)(Y)
	#build X and Y for each output node
	X = {}
	Y = {}
	for node in g.net.output_nodes:
		X[node] = []
		Y[node] = []
	for lesson in lessons:
		(lesson_input, lesson_output) = lesson
		g.net.activate(lesson_input)
		for node, _, _, bias, response, links in g.net.node_evals:
			if X.has_key(node):
				X[node].append([g.net.values[i] for i,_ in links])
				#y = 1/(1+exp(-5x)) => x = ln(y/(1-y))/5
				#x = bias + response*aggreg => aggreg = (ln(y/(1-y))/5 - bias)/response
				if lesson_output[node] != 0.0 and lesson_output[node] != 1.0 and response != 0:
					Y[node].append((math.log(lesson_output[node]/(1.0-lesson_output[node]))/5.0 - bias)/response)
				else:
					Y[node].append(lesson_output[node])
	
	#compute optimal w for each output node
	w = {}
	for node in g.net.output_nodes:
		if len(X[node]) == 0: #if the output node is disconnected from the rest of the net, no fix can be done
			continue
		w[node] = np_solve(X[node], Y[node])
	
	#update the weights of the links with w
	for node, _, _, _, _, links in g.net.node_evals:
		if X.has_key(node) and not w[node] is None:
			for i in range(len(links)):
				links[i] = (links[i][0], (1.0-learning_rate)*links[i][1] + learning_rate*float(w[node][i]))
				g.connections[(links[i][0], node)].weight = links[i][1]

############################### EVALUATIONS ###################################
		
def evaluate_net(task, net, env, timeout, knowledge, attain_knowledge):
	if task == 'CartPole-v0' or task == 'CartPole-v1':
		return go_CartPole(net, env, timeout, knowledge, attain_knowledge)
	if task == 'MountainCar-v0':
		return go_MountainCar(net, env, timeout, knowledge, attain_knowledge)

#-----------------------------------------------------------------------------#

def go_CartPole(net, env, timeout, knowledge, attain_knowledge):
	inputs = env.reset()
	total_reward = 0.0
	for t in range(timeout):
		outputs = net.activate(inputs)
		if attain_knowledge:
			knowledge.append( (inputs,outputs) )
		action = np.argmax(outputs)
		inputs, reward, done, _ = env.step(action)
		if done:
			break
		#rewards are higher if the car is near the center
		total_reward += reward/(1.0 + abs(inputs[0]))
	return total_reward

def go_MountainCar(net, env, timeout, knowledge, attain_knowledge):
	inputs = env.reset()
	max_x = None
	for t in range(timeout):
		outputs = net.activate(inputs)
		if attain_knowledge:
			knowledge.append( (inputs,outputs) )
		action = np.argmax(outputs)
		inputs, reward, done, _ = env.step(action)
		if max_x is None or max_x < inputs[0]:
			max_x = inputs[0]
		if done:
			break
	#rewards are higher if the car finishes sooner
	return 10000*(max_x+0.5)/float(t)

################################ INHERITANCES #################################

class CustomReproduction(DefaultReproduction):
	#marks the genomes with the generation they're born at
	def __init__(self, config, reporters, stagnation):
		super(CustomReproduction, self).__init__(config.reproduction_config, reporters, stagnation)
		self.config = config
	
	def create_new(self, genome_type, genome_config, num_genomes):
		new_genomes = {}
		for i in range(num_genomes):
			key = self.genome_indexer.get_next()
			g = genome_type(key)
			g.configure_new(genome_config)
			g.net = FeedForwardNetwork.create(g, self.config)
			new_genomes[key] = g
			self.ancestors[key] = tuple()
		return new_genomes

	def reproduce(self, config, species, pop_size, generation, exp, syllabus):
		all_fitnesses = []
		for sid, s in iteritems(species.species):
			all_fitnesses.extend(m.fitness for m in itervalues(s.members))
		min_fitness = min(all_fitnesses)
		max_fitness = max(all_fitnesses)
		fitness_range = max(1.0, max_fitness - min_fitness)
		num_remaining = 0
		species_fitness = []
		avg_adjusted_fitness = 0.0
		for sid, s, stagnant in self.stagnation.update(species, generation):
			if stagnant:
				self.reporters.species_stagnant(sid, s)
			else:
				num_remaining += 1
				msf = mean([m.fitness for m in itervalues(s.members)])
				s.adjusted_fitness = (msf - min_fitness) / fitness_range
				species_fitness.append((sid, s, s.fitness))
				avg_adjusted_fitness += s.adjusted_fitness
		if 0 == num_remaining:
			species.species = {}
			return []
		avg_adjusted_fitness /= len(species_fitness)
		self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))
		spawn_amounts = []
		for sid, s, sfitness in species_fitness:
			spawn = len(s.members)
			if sfitness > avg_adjusted_fitness:
				spawn = max(spawn + 2, spawn * 1.1)
			else:
				spawn = max(spawn * 0.9, 2)
			spawn_amounts.append(spawn)
		total_spawn = sum(spawn_amounts)
		norm = pop_size / total_spawn
		spawn_amounts = [int(round(n * norm)) for n in spawn_amounts]
		new_population = {}
		species.species = {}
		
		learning_function = None
		if exp.learning_function == 'BP':
			learning_function = backpropagation
		elif exp.learning_function == 'BT':
			learning_function = batch
		
		has_learning = not learning_function is None
		if has_learning:
			(writer, lessons) = syllabus
			lessons_to_learn = random.sample(lessons, min(exp.syllabus_size, len(lessons)))
		
		for spawn, (sid, s, sfitness) in zip(spawn_amounts, species_fitness):
			spawn = max(spawn, self.elitism)
			if spawn <= 0:
				continue
			old_members = list(iteritems(s.members))
			s.members = {}
			species.species[sid] = s
			old_members.sort(reverse=True, key=lambda x: x[1].fitness)
			if self.elitism > 0:
				for i, m in old_members[:self.elitism]:
					new_population[i] = m
					spawn -= 1
			if spawn <= 0:
				continue
			repro_cutoff = int(math.ceil(self.survival_threshold * len(old_members)))
			repro_cutoff = max(repro_cutoff, 2)
			old_members = old_members[:repro_cutoff]
			
			if has_learning and (exp.learning_target == 'Ambos' or exp.learning_target == 'Pais'):
				for (_, g) in old_members:
					if g != writer:
						learning_function(g, lessons_to_learn, exp.learning_rate)
						
			while spawn > 0:
				spawn -= 1
				parent1_id, parent1 = random.choice(old_members)
				parent2_id, parent2 = random.choice(old_members)
				gid = self.genome_indexer.get_next()
				child = config.genome_type(gid)
				child.configure_crossover(parent1, parent2, config.genome_config)
				child.mutate(config.genome_config)
				child.net = FeedForwardNetwork.create(child, self.config)
				
				if has_learning and (exp.learning_target == 'Ambos' or exp.learning_target == 'Filhos'):
					if exp.children_inclusion == 'Inicial' or (exp.children_inclusion == 'Tardia' and generation >= exp.generations/2):
						learning_function(child, lessons_to_learn, exp.learning_rate)
				
				new_population[gid] = child
				self.ancestors[gid] = (parent1_id, parent2_id)
		return new_population

class CustomGenome(DefaultGenome):
	#has extra attributes
	def __init__(self, key, generation=0):
		super(CustomGenome, self).__init__(key)
		self.knowledge = None
		self.net = None

class CustomParallelEvaluator(ParallelEvaluator):
	#marks genomes with their knowledge
	def __init__(self, num_workers, eval_function, timeout=None):
		super (CustomParallelEvaluator, self).__init__(num_workers, eval_function, timeout)
	 
	def evaluate(self, genomes, config):
		jobs = []
		for genome_id, genome in genomes:
			jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))
		for job, (genome_id, genome) in zip(jobs, genomes):
			fitness, knowledge = job.get(timeout=self.timeout)
			genome.fitness = fitness
			genome.knowledge = knowledge

class CustomPopulation(Population):
	#has the syllabus mechanics
	
	def __init__(self, config, initial_state=None):
		self.reporters = ReporterSet()
		self.config = config
		stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
		self.reproduction = config.reproduction_type(config, self.reporters, stagnation)
		if config.fitness_criterion == 'max':
			self.fitness_criterion = max
		elif config.fitness_criterion == 'min':
			self.fitness_criterion = min
		elif config.fitness_criterion == 'mean':
			self.fitness_criterion = mean
		else:
			raise Exception("Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

		if initial_state is None:
			# Create a population from scratch, then partition into species.
			self.population = self.reproduction.create_new(config.genome_type, config.genome_config, config.pop_size)
			self.species = config.species_set_type(config, self.reporters)
			self.generation = 0
			self.species.speciate(config, self.population, self.generation)
		else:
			self.population, self.species, self.generation = initial_state

		self.best_genome = None
	
	def run(self, fitness_function, best_fitnesses, exp):
		syllabus = (None,[])
		for gen in range(exp.generations):
			self.reporters.start_generation(self.generation)
			fitness_function(list(iteritems(self.population)), self.config)
			best = None
			for g in itervalues(self.population):
				if best is None or g.fitness > best.fitness:
					best = g
			syllabus = (best, best.knowledge)
			best_fitnesses.append(best.fitness)
			self.reporters.post_evaluate(self.config, self.population, self.species, best)
			if self.best_genome is None or best.fitness > self.best_genome.fitness:
				self.best_genome = best
			fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
			if fv >= self.config.fitness_threshold:
				self.reporters.found_solution(self.config, self.generation, best)
				break
			self.population = self.reproduction.reproduce(self.config, self.species, self.config.pop_size, self.generation, exp, syllabus)
			if not self.species.species:
				self.reporters.complete_extinction()
				if self.config.reset_on_extinction:
					self.population = self.reproduction.create_new(self.config.genome_type, self.config.genome_config, self.config.pop_size)
				else:
					raise CompleteExtinctionException()
			self.species.speciate(self.config, self.population, self.generation)
			self.reporters.end_generation(self.config, self.population, self.species)
			self.generation += 1
		return self.best_genome

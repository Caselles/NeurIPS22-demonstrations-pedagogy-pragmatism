import sys
sys.path.append('/home/gohu/workspace/postdoc/decstr/gangstr_teacher/gangstr_predicates_instructions/')

import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from demo_encoder import DemoEncoder
from demonstrations_buffer import MultiDemoBuffer
from teacher.teacher import Teacher
import random
import torch
from utils import generate_goals_demonstrator, convert_goals
from utils_demo import select_obs_act_in_episode
import time
from pathlib import Path
from mpi_utils import logger
import pickle

VAL_FREQ = 1
ADD_GOAL_FREQ = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_env_params(env):
	obs = env.reset()

	# close the environment
	params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
			  'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
			  'max_timesteps': env._max_episode_steps, 'nb_possible_goals': 35, 'max_demo_size': 400}
	return params

def launch(args):

	save_path = '/home/gohu/workspace/postdoc/decstr/gangstr_teacher/gangstr_predicates_instructions/demo_encoder_baseline/results/'

	test_path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/' + args.teacher_mode + '_teacher_10k/'

	Path(save_path).mkdir(parents=True, exist_ok=True)

	file = open(save_path + "scores.txt","w")
	file.write("Epoch - Score \n")
	file.close()


	# Make the environment
	args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)
	env = gym.make(args.env_name)

	args.env_params = get_env_params(env)

	add_goal = False
	ind_goal_discovered = 1

	all_achievable_goals = generate_goals_demonstrator()

	#discovered_goals = [all_achievable_goals[0]]
	discovered_goals = all_achievable_goals

	# Initialize demo encoder goal predictor function module
	demo_encoder_training = DemoEncoder(args, gpu=True)

	# Initialize teacher
	teacher = Teacher(args)

	# Create training buffer
	sampled_goals = np.array([g for _ in range(args.nb_available_demos) for g in all_achievable_goals])
	demos = teacher.get_demo_for_goals_demo_encoder_training(sampled_goals, saved=True)
	target_goals = convert_goals(goals=sampled_goals, softmax_to_predicates=False)
	demo_encoder_batch = [[demos[i], target_goals[i]] for i in range(len(demos))]
	demo_encoder_training.store(demo_encoder_batch)

	# Create testing buffer
	testing_buffer = MultiDemoBuffer(env_params=args.env_params,
                                  buffer_size=args.buffer_size)
	sampled_goals = np.array([g for _ in range(10) for g in all_achievable_goals])
	demos = teacher.get_demo_for_goals_demo_encoder_testing(sampled_goals, saved=True)
	target_goals = convert_goals(goals=sampled_goals, softmax_to_predicates=False)
	demo_encoder_batch = [[demos[i], target_goals[i]] for i in range(len(demos))]
	testing_buffer.store_batch(demo_encoder_batch)

	# Main interaction loop
	for epoch in range(args.n_epochs + 200):

		# Demo encoder - Goal predictor function updates
		for _ in range(args.n_batches_demo_encoder):
			demo_encoder_training.train()

		if add_goal:
			if epoch % ADD_GOAL_FREQ == 0:
				if ind_goal_discovered <= 34:
					discovered_goals.append(all_achievable_goals[ind_goal_discovered])
					ind_goal_discovered += 1


		if epoch % VAL_FREQ == 0: 
			demo_encoder_training.save(save_path, epoch, training=True)
			print('\tRunning eval for epoch ', str(epoch), '..')

			scores_test = []

			data = testing_buffer.sample(350)

			# retrieve demo in episode
			demos_test = select_obs_act_in_episode(data[:,0])

			target_goals = np.where(np.array(list(data[:,1])) == 1)[1] #... the np.array(list()) is ridiculous, has to be changed
			target_goals = torch.tensor(target_goals, dtype=torch.long)

			predicted_goals = demo_encoder_training.encode(demos_test, no_grad=True, gpu=True)

			scores_test = [1 if x == g else 0 for x,g in zip(np.argmax(predicted_goals, axis=1),target_goals)]

			print(np.mean(scores_test), ' - TEST SCORE.')

			'''with open(save_path + "scores.txt", "a") as file:
				file.write(str(epoch) + ' - ' + str(np.mean(scores_test)) +"\n")
				file.close()'''






if __name__ == '__main__':

	# Get parameters
	args = get_args()

	launch(args)
from teacher.teacher import Teacher

import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
from language.build_dataset import sentence_from_configuration
from utils import get_instruction, generate_goals_demonstrator
from arguments import get_args
import pickle as pkl
from pathlib import Path
import os

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.


def get_env_params(env):
	obs = env.reset()
	# close the environment
	params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
			  'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
			  'max_timesteps': env._max_episode_steps}
	return params

if __name__ == '__main__':
	num_eval = 1

	path_save = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/'

	nb_demos = 100

	demonstrator = 'learner_naive_pragmatic'

	
	if demonstrator == 'naive_teacher':
		path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-07 12:36:50_FetchManipulate3Objects-v0_gnn_per_object_NAIVE_TEACHER_FOR_FINETUNING_70%_PEDAGOGICAL_SR/models/'
		model_path = path + 'model_150.pt'
		no_noise = False

	if demonstrator == 'pedagogical_teacher':
		path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-11 17:01:34_FetchManipulate3Objects-v0_gnn_per_object_PEDAGOGICAL_TEACHER_DONE_WORKS/models/'
		model_path = path + 'model_170.pt'
		no_noise = False

	if demonstrator == 'naive_teacher_no_noise':
		path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-07 12:36:50_FetchManipulate3Objects-v0_gnn_per_object_NAIVE_TEACHER_FOR_FINETUNING_70%_PEDAGOGICAL_SR/models/'
		model_path = path + 'model_150.pt'
		no_noise = True

	if demonstrator == 'pedagogical_teacher_no_noise':
		path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-11 17:01:34_FetchManipulate3Objects-v0_gnn_per_object_PEDAGOGICAL_TEACHER_DONE_WORKS/models/'
		model_path = path + 'model_170.pt'
		no_noise = True

	if demonstrator == 'naive_teacher_10k':
		path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-07 12:36:50_FetchManipulate3Objects-v0_gnn_per_object_NAIVE_TEACHER_FOR_FINETUNING_70%_PEDAGOGICAL_SR/models/'
		model_path = path + 'model_150.pt'
		no_noise = False

	if demonstrator == 'pedagogical_teacher_10k':
		path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-11 17:01:34_FetchManipulate3Objects-v0_gnn_per_object_PEDAGOGICAL_TEACHER_DONE_WORKS/models/'
		model_path = path + 'model_170.pt'
		no_noise = False

	# Only for testing the learner for OGIA

	if demonstrator == 'learner_naive_literal':
		path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/job_SIGNIFICANT_goal_pred_sqil_demos7_exp3_1000_demos/naive_literal/2022-03-09 17:23:02_FetchManipulate3Objects-v0_gnn_per_object/models/'
		model_path = path + 'model_100.pt'
		no_noise = False

	if demonstrator == 'learner_naive_pragmatic':
		path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/job_SIGNIFICANT_goal_pred_sqil_demos7_exp3_1000_demos/naive_pragmatic/2022-03-09 17:23:52_FetchManipulate3Objects-v0_gnn_per_object/models/'
		model_path = path + 'model_100.pt'
		no_noise = False


	rank = MPI.COMM_WORLD.Get_rank()

	# with open(path + 'config.json', 'r') as f:
	#     params = json.load(f)
	# args = SimpleNamespace(**params)
	args = get_args()

	if args.algo == 'continuous':
		args.env_name = 'FetchManipulate3ObjectsContinuous-v0'
		args.multi_criteria_her = True
	else:
		args.env_name = 'FetchManipulate3Objects-v0'

	# Make the environment
	env = gym.make(args.env_name)

	# set random seeds for reproduce
	args.seed = np.random.randint(1e6)
	env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
	if args.cuda:
		torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

	args.env_params = get_env_params(env)

	goal_sampler = GoalSampler(args)

	# create the sac agent to interact with the environment
	if args.agent == "SAC":
		policy = RLAgent(args, env.compute_reward, goal_sampler)
		policy.load(model_path, args)
	else:
		raise NotImplementedError

	# def rollout worker
	rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

	# eval_goals = goal_sampler.valid_goals
	#eval_goals, eval_masks = goal_sampler.generate_eval_goals()
	eval_goals = np.array(generate_goals_demonstrator())
	proba_goals = eval_goals.copy()
	eval_masks = np.zeros((len(eval_goals), 9))
	if args.algo == 'language':
		language_goal = get_instruction()
		eval_goals = np.array([goal_sampler.valid_goals[0] for _ in range(len(language_goal))])
	else:
		language_goal = None
	inits = [None] * len(eval_goals)
	all_results = []

	illustrative_example = False

	if illustrative_example:
		# one is 0 over 1, the other is 0 above 1 and 0 close to 1
		eval_goals = np.array([[1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.], [-1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]])
		proba_goals = eval_goals.copy()
		# the trick is that the initialization will gave 0 close to 1 already in one case, and not the other

		no_noise = False # we need to have a stochastic policy

	goal = eval_goals[rank]
	goal_ind = rank
	print('GOAL', goal, rank)
	Path(path_save + demonstrator + '/goal_' + str(goal_ind)).mkdir(parents=True, exist_ok=True)
	success = 0
	while success < nb_demos:
		episodes = rollout_worker.generate_rollout(np.array([goal]), eval_masks, self_eval=no_noise, true_eval=no_noise, biased_init=True, animated=False, 
			language_goal=language_goal, verbose=False, return_proba=False, illustrative_example=illustrative_example)

		print('Episode success: ', episodes[0]['success'][-1])

		if episodes[0]['success'][-1]:
			with open(path_save + demonstrator + '/goal_' + str(goal_ind) + '/demo_' + str(success) + '.pkl', 'wb') as f:
				pkl.dump(episodes[0], f)
			success += 1

	assert len(os.listdir(path_save + demonstrator + '/goal_' + str(goal_ind))) == nb_demos
		



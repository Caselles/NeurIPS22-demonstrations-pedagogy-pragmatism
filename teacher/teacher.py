import sys 

sys.path.append('../')

import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from utils import generate_goals_demonstrator
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import random
from mpi4py import MPI
import pickle
from copy import deepcopy
from arguments import get_args
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Teacher:
    def __init__(self, args):

        self.policy = None
        self.demo_dataset = None
        self.all_goals = generate_goals_demonstrator()
        if args.teacher_mode == 'naive':
            self.path_demos = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/naive_teacher_10k/'
        if args.teacher_mode == 'pedagogical':
            self.path_demos = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/pedagogical_teacher_10k/'

        if args.teacher_mode == 'naive_no_noise':
            self.path_demos = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/naive_teacher_no_noise/'
        if args.teacher_mode == 'pedagogical_no_noise':
            self.path_demos = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/pedagogical_teacher_no_noise/'

        if args.teacher_mode == 'learner_naive_literal':
            self.path_demos = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/learner_naive_literal/'

        if args.teacher_mode == 'learner_naive_pragmatic':
            self.path_demos = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/demos_datasets/learner_naive_pragmatic/'



        self.nb_available_demos = len(os.listdir(self.path_demos+'goal_0/'))


    def get_demo_for_goals(self, goals, saved=False):

        demos = []

        for goal in goals:

            if saved:

                goal_ind = self.all_goals.index(goal.tolist())

                ind = np.random.randint(self.nb_available_demos)

                with open(self.path_demos + 'goal_' + str(goal_ind) + '/demo_' + str(ind) + '.pkl', 'rb') as f:
                    demo = pickle.load(f)

                # check if we get demo for the right goal
                assert (goal == demo['g'][-1]).all()

                demos.append(demo)

            else:
                raise NotImplementedError

        return demos


    def get_demo_for_goals_demo_encoder_training(self, goals, saved=False):

        demos = []

        for goal in goals:

            if saved:

                goal_ind = self.all_goals.index(goal.tolist())

                ind = np.random.randint(9000)

                with open(self.path_demos + 'goal_' + str(goal_ind) + '/demo_' + str(ind) + '.pkl', 'rb') as f:
                    demo = pickle.load(f)

                # check if we get demo for the right goal
                assert (goal == demo['g'][-1]).all()

                demos.append(demo)

            else:
                raise NotImplementedError

        return demos

    def get_demo_for_goals_demo_encoder_testing(self, goals, saved=False):

        demos = []

        for goal in goals:

            if saved:

                goal_ind = self.all_goals.index(goal.tolist())

                ind = np.random.randint(9000, 10000)

                with open(self.path_demos + 'goal_' + str(goal_ind) + '/demo_' + str(ind) + '.pkl', 'rb') as f:
                    demo = pickle.load(f)

                # check if we get demo for the right goal
                assert (goal == demo['g'][-1]).all()

                demos.append(demo)

            else:
                raise NotImplementedError

        return demos


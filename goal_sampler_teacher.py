import numpy as np
from mpi4py import MPI
from utils import generate_goals_demonstrator



class GoalSamplerTeacher:
    def __init__(self, args):
        self.goal_dim = args.env_params['goal']
        #self.discovered_goals = [np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1.])]
        #self.discovered_goals_str = [str(self.discovered_goals)[0]]
        self.discovered_goals = generate_goals_demonstrator()
        self.discovered_goals_str = [str(g) for g in self.discovered_goals]
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.all_achievable_goals = generate_goals_demonstrator()

    def sample_goals(self, all_achievable_goals, nb_goals):
        """Samples n goals uniformly"""

        #print(len(self.discovered_goals), 'DISCOVERED GOALS')

        goal_indexes = np.random.choice(len(self.discovered_goals), nb_goals)
        goals = np.array(self.discovered_goals)[goal_indexes]

        return goals

    def add_discovered_goals(self, episodes):
        """
        Update discovered goals list from episodes
        """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                # Add last achieved goal to memory if first time encountered and valid goal!!
                if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                    #check for goal validity
                    if list(e['ag'][-1]) in self.all_achievable_goals:
                        print('valid goal, we add')
                        self.discovered_goals.append(e['ag_binary'][-1].copy())
                        self.discovered_goals_str.append(str(e['ag_binary'][-1]))
                    else:
                        print('not valid goal!!!')

        self.sync()

        return episodes

    def sync(self):
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)

    def resample_incorrect_goal_predictions(self, sampled_goals, prediction_success):

        resampled_predicted_goals = []

        for i, goal in enumerate(sampled_goals):

            if prediction_success[i]:
                resampled_predicted_goals.append(goal)
            else:
                resampled_goal = np.array(self.discovered_goals)[np.random.choice(len(self.discovered_goals))]
                resampled_predicted_goals.append(resampled_goal)

        return np.array(resampled_predicted_goals)

    
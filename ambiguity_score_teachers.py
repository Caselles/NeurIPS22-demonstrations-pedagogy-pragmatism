import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
from teacher.teacher import Teacher
import random
from mpi4py import MPI
from language.build_dataset import sentence_from_configuration
from utils import get_instruction, generate_goals_demonstrator
from arguments import get_args
import pickle as pkl

#from mujoco_py import GlfwContext
#GlfwContext(offscreen=True)  # Create a window to init GLFW.


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def run_test(args, model_path=None):

    ################ SETTING UP TESTING ENVIRONMENT ################

    stable_prediction_accuracy = []
    stable_success_rate = []

    ambiguity_score = []

    if args.compute_statistically_significant_results:
        nb_runs_predictability = 30
        nb_runs_reachability = 30
    else:
        nb_runs_predictability = 1
        nb_runs_reachability = 1

    ### CREATE ENV
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

    ### INITIALIZE TEACHER
    teacher = Teacher(args)

    ### LIST OF POSSIBLE AGENTS
    if model_path==None:

        if args.learner_to_test == 'naive_teacher':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-07 12:36:50_FetchManipulate3Objects-v0_gnn_per_object_NAIVE_TEACHER_FOR_FINETUNING_70%_PEDAGOGICAL_SR/models/'
            model_path = path + 'model_150.pt'

        if args.learner_to_test == 'pedagogical_teacher':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-11 17:01:34_FetchManipulate3Objects-v0_gnn_per_object_PEDAGOGICAL_TEACHER_DONE_WORKS/models/'
            model_path = path + 'model_150.pt'

        if args.learner_to_test == 'literal_learner_naive_teacher':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-01-18 13:42:27_FetchManipulate3Objects-v0_gnn_per_object_LEARNER_LITERAL_NAIVE_TEACHER/models/'
            model_path = path + 'model_90.pt'

        if args.learner_to_test == 'literal_learner_pedagogical_teacher':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-02 11:43:33_FetchManipulate3Objects-v0_gnn_per_object_EXP20PERCENT_PEDAGOGICAL_TEACHER_BC_REG_ALL_DEMOS/models/'
            model_path = path + 'model_180.pt'

        if args.learner_to_test == 'literal_learner_pedagogical_teacher_only_demos':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-03 23:24:46_FetchManipulate3Objects-v0_gnn_per_object/models/'
            model_path = path + 'model_80.pt'

        if args.learner_to_test == 'literal_learner_pedagogical_teacher_BC_80_percent':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-01-21 11:12:21_FetchManipulate3Objects-v0_gnn_per_object_PEDAGOGICAL_TEACHER_80percent/models/'
            model_path = path + 'model_100.pt'

        if args.learner_to_test == 'literal_learner_pedagogical_teacher_sqil':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-06 21:15:19_FetchManipulate3Objects-v0_gnn_per_object_SQUIL_PEDAGOGICAL_TEACHER_10k_DEMOS_BIASED_INIT/models/'
            model_path = path + 'model_110.pt'

        if args.learner_to_test == 'literal_learner_naive_teacher_sqil':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-06 20:34:35_FetchManipulate3Objects-v0_gnn_per_object_SQUIL_NAIVE_TEACHER_1000_DEMOS_NO_BIASED_INIT_FORCES_complete/models/'
            model_path = path + 'model_110.pt'

        if args.learner_to_test == 'literal_learner_naive_teacher_sqil_demos_r_1plusnormal_exp_r_0':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/plots/SQIL_BOTH_TEACHER_1k_DEMOS_REWARDSDEMOSNORMALPLUS1_REWARDSEXP0/SQIL_NAIVE_TEACHER_1k_DEMOS_REWARDSDEMOSNORMALPLUS1_REWARDSEXP0/run1/models/'
            model_path = path + 'model_90.pt'

        if args.learner_to_test == 'literal_learner_pedagogical_teacher_sqil_demos_r_1plusnormal_exp_r_0':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/plots/SQIL_BOTH_TEACHER_1k_DEMOS_REWARDSDEMOSNORMALPLUS1_REWARDSEXP0/SQIL_PEDAGOGICAL_TEACHER_1k_DEMOS_REWARDSDEMOSNORMALPLUS1_REWARDSEXP0/run1/models/'
            model_path = path + 'model_90.pt'

        if args.learner_to_test == 'literal_learner_naive_teacher_sqil_demos_r_10plusnormal_exp_r_normal':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-12 14:40:31_FetchManipulate3Objects-v0_gnn_per_object_SQIL_NAIVE_TEACHER_1k_DEMOS_REWARDSDEMOSNORMALPLUS10_REWARDSEXPNORMAL/models/'
            model_path = path + 'model_100.pt'

        if args.learner_to_test == 'literal_learner_pedagogical_teacher_sqil_demos_r_10plusnormal_exp_r_normal':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-12 15:36:27_FetchManipulate3Objects-v0_gnn_per_object_SQIL_PEDAGOGICAL_TEACHER_1k_DEMOS_REWARDSDEMOSNORMALPLUS10_REWARDSEXPNORMAL/models/'
            model_path = path + 'model_100.pt'

        if args.learner_to_test == 'literal_learner_pedagogical_teacher_sqil_demos_r_7_exp_r_normalplusrewardpedagogical':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-16 11:44:48_FetchManipulate3Objects-v0_gnn_per_object_SQIL_PEDAGOGICAL_TEACHER_1k_DEMOS_REWARDSDEMOS7_REWARDSEXPNORMAL_PRAGMATISMPEDAGOGICALREWARD/models/'
            model_path = path + 'model_100.pt'

        if args.learner_to_test == 'goal_pred_literal_learner_pedagogical_teacher_sqil_demos_r_7_exp_r_normal':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-19 16:40:59_FetchManipulate3Objects-v0_gnn_per_object_GOALPRED_SQIL_PEDAGOGICAL_TEACHER_1k_DEMOS_REWARDSDEMOS7_REWARDSEXPNORMAL_/models/'
            model_path = path + 'model_100.pt'

        if args.learner_to_test == 'goal_pred_literal_learner_naive_teacher_sqil_demos_r_7_exp_r_normal':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-19 15:47:51_FetchManipulate3Objects-v0_gnn_per_object_GOALPRED_SQIL_NAIVE_TEACHER_1k_DEMOS_REWARDSDEMOS7_REWARDSEXPNORMAL/models/'
            model_path = path + 'model_100.pt'

        if args.learner_to_test == 'naive_literal_100_demos':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/job_SIGNIFICANT_goal_pred_sqil_demos7_exp3_100_demos/naive_literal/2022-03-10 11:10:32_FetchManipulate3Objects-v0_gnn_per_object/models/'
            model_path = path + 'model_100.pt'

        if args.learner_to_test == 'pedagogic_pragmatic_100_demos':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/job_SIGNIFICANT_goal_pred_sqil_demos7_exp3_100_demos/pedagogical_pragmatic/2022-03-10 11:07:23_FetchManipulate3Objects-v0_gnn_per_object/models/'
            model_path = path + 'model_100.pt'

        if args.learner_to_test == 'custom':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/2022-02-21 17:08:06_FetchManipulate3Objects-v0_gnn_per_object/models/'
            model_path = path + 'model_100.pt'

    else:

        path = model_path
        model_path = path + '/models/model_100.pt'

    ### CREATE AND LOAD THE AGENT
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
    else:
        raise NotImplementedError

    ### ROLLOUT WORKER
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    ### INITIALIZE EVALUATION GOALS
    eval_goals = np.array(generate_goals_demonstrator())
    proba_goals = eval_goals.copy()
    no_noise = False
    eval_masks = np.zeros((len(eval_goals), 9))
    language_goal = None
    ambiguity_analysis = True

    ### ENABLE ILLUSTRATIVE EXAMPLE SETUP
    if args.illustrative_example:

        example = 1

        if example == 1:
            # 1 close at the beginning, goals are 3rd close to 1 close / all close
            eval_goals = [np.array([[1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.], [-1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]]) for _ in range(100)]
            proba_goals = eval_goals.copy()
            
            stack_pos_init = [np.array([[np.random.uniform(1.1, 1.5), np.random.uniform(0.55, 0.95)], [np.random.uniform(1.1, 1.5), np.random.uniform(0.55, 0.95)]]) for _ in range(100)]

        if example == 2:
            # 1 above at the beginning, goals are above and 3rd close / bring 3rd close to base of above 
            eval_goals = [np.array([[1.,1.,-1.,-1.,-1.,1.,-1.,-1.,-1.], [1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]]) for _ in range(100)]
            proba_goals = eval_goals.copy()
            
            stack_pos_init = [np.array([[np.random.uniform(1.1, 1.5), np.random.uniform(0.55, 0.95)], [np.random.uniform(1.1, 1.5), np.random.uniform(0.55, 0.95)]]) for _ in range(100)]

        if example == 3:
            # 1 above at the beginning, goals are above and 3rd close / bring 3rd close to base of above 
            eval_goals = [np.array([[1.,1.,-1.,-1.,-1.,-1.,-1.,1.,-1.], [-1.,1.,-1.,-1.,-1.,-1.,-1.,1.,-1.]]) for _ in range(100)]
            proba_goals = eval_goals.copy()
            
            stack_pos_init = [np.array([[np.random.uniform(1.1, 1.5), np.random.uniform(0.55, 0.95)], [np.random.uniform(1.1, 1.5), np.random.uniform(0.55, 0.95)]]) for _ in range(100)]

    else:
        stack_pos_init = [None for i in range(len(eval_goals))]
        example = 1 


    ################ PREDICTABILITY RESULTS ################
    print(args.predictability)
    if args.predictability:
        for i in range(nb_runs_predictability):
            demos = teacher.get_demo_for_goals(eval_goals, saved=True)
            predicted_goals = rollout_worker.predict_goal_from_demos(demos, list(eval_goals))
            self_eval = False
            prediction_success = [1 if (eval_goals[pg_ind] == predicted_goal).all() else 0 for pg_ind, predicted_goal in enumerate(predicted_goals)]
            if ambiguity_analysis:
                for pg_ind, pred in enumerate(prediction_success):
                    if not pred:
                        print('pursued goal in the demo =', eval_goals[pg_ind])
                        print('achieved goal in the demo =', demos[pg_ind]['ag'][-1])
                        print('is pursued goal is the same as achieved goal?', (eval_goals[pg_ind]==demos[pg_ind]['ag'][-1]).all())
                        print('predicted goal =', predicted_goals[pg_ind])
            print(prediction_success, 'Prediction success on this run.')
            print(np.mean(prediction_success), 'Mean predictability on this run.')
            stable_prediction_accuracy.append(prediction_success)

        if args.compute_statistically_significant_results:
            print(stable_prediction_accuracy, 'All prediction accuracy results.')
            print(np.mean(stable_prediction_accuracy), 'Average prediction accuracy.')

    #eval_goals = np.array([np.array([ 1.,  1.,  1., -1., -1., -1., -1., -1., -1.])])

    ################ REACHABILITY RESULTS ################
    if args.reachability:
        for i in range(len(eval_goals)):
            episodes = rollout_worker.generate_rollout(eval_goals[i], eval_masks, self_eval=no_noise, true_eval=no_noise, biased_init=True, animated=True, 
                language_goal=language_goal, verbose=False, return_proba=None, illustrative_example=args.illustrative_example, stack_pos_init=stack_pos_init[i], example=example,
                save_ambiguous_episodes=False)


            goals_achieved = np.sum(np.array([int(e['success'][-1]) for e in episodes])) == 2
            ambiguity = (episodes[0]['ag'][-1] == episodes[1]['ag'][-1]).all()

            if goals_achieved and not ambiguity:
                ambiguity_score.append(0)
                print('No ambiguity!')
            else:
                ambiguity_score.append(1)
                print('Ambiguity!')


        print(np.mean(ambiguity_score), 'Average ambiguity rate.')

    if 'models' in path:
        save_path = path.split('models')[0]
    else:
        save_path = path + '/'

    '''with open(save_path + 'results_predictability_reachability.txt', 'w') as f:
        f.write('Predictability:' + str(np.mean(stable_prediction_accuracy)))
        f.write('\n')
        f.write('Reachability:' + str(np.mean(stable_success_rate)))'''

    return np.mean(stable_prediction_accuracy), np.mean(stable_success_rate)

if __name__ == '__main__':

    args = get_args()
    args.env_name = 'FetchManipulate3Objects-v0'

    run_test(args)


    

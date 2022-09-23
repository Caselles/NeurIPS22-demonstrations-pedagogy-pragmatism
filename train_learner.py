import torch
import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
from rollout import RolloutWorker
from temporary_lg_goal_sampler import LanguageGoalSampler
from goal_sampler import GoalSampler
from goal_sampler_teacher import GoalSamplerTeacher
from teacher.teacher import Teacher
from utils import init_storage, get_instruction, get_eval_goals, generate_goals_demonstrator
import time
from mpi_utils import logger
from language.build_dataset import sentence_from_configuration
from test_learner_mpi import run_test_mpi

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def launch(args):

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

    # Make the environment
    args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)
    env = gym.make(args.env_name)

    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get saving paths
    if rank == 0:
        logdir, model_path = init_storage(args)
        global MODEL_PATH
        MODEL_PATH = logdir
        logger.configure(dir=logdir)
        logger.info(vars(args))
    else:
        logdir = ''
        
    logdir = MPI.COMM_WORLD.bcast(logdir, root=0)

    args.env_params = get_env_params(env)

    language_goal = None
    goal_sampler = GoalSampler(args)
    goal_sampler_teacher = GoalSamplerTeacher(args)
    all_achievable_goals = generate_goals_demonstrator()

    # Initialize RL Agent
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
    else:
        raise NotImplementedError

    # Fine-tuning for pedagogical teacher
    #if args.pedagogical_teacher:
        #ckpt_path = 
        #'/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output/2022-01-07 12:36:50_FetchManipulate3Objects-v0_gnn_per_object_NAIVE_TEACHER_FOR_FINETUNING/models/'
        #saved_model_path = ckpt_path + 'model_150.pt'
        #ckpt_path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output/2022-01-09 18:18:11_FetchManipulate3Objects-v0_gnn_per_object/models/'
        #saved_model_path = ckpt_path + 'model_60.pt'''
        #ckpt_path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output/2022-01-10 16:52:14_FetchManipulate3Objects-v0_gnn_per_object/models/'
        #saved_model_path = ckpt_path + 'model_60.pt'''
        #policy.load(saved_model_path, args)
        #goal_sampler.set_goal_sampler_for_fine_tuning()
        #policy.buffer.load_buffer(ckpt_path + 'buffer_60.pkl')

    # Initialize Rollout Worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # Initialize Teacher
    teacher = Teacher(args)

    # Main interaction loop
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()

        # setup time_tracking
        time_dict = dict(goal_sampler=0,
                         rollout=0,
                         gs_update=0,
                         store=0,
                         norm_update=0,
                         policy_train=0,
                         policy_train_bc=0,
                         policy_train_sil=0,
                         eval=0,
                         epoch=0)

        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Annealing SQIL ratio of experience vs demos in replay buffer
        if epoch == -1 and args.sqil:
            policy.her_module.demos_vs_exp = 0.9

            print(policy.her_module.demos_vs_exp, ' SQIL ratio changed!')

        # Cycles loop
        for _ in range(args.n_cycles):

            # Sample goals
            t_i = time.time()
            '''sampled_goals = goal_sampler_teacher.sample_goals(all_achievable_goals, nb_goals=args.num_rollouts_per_mpi)
            sampled_masks = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(len(sampled_goals))])
            self_eval = False'''

            '''if args.cuda:
                policy.model.actor.cpu()
                policy.model.critic.cpu()
                policy.model.critic_target.cpu()'''

            sampled_goals = goal_sampler_teacher.sample_goals(all_achievable_goals, nb_goals=args.num_rollouts_per_mpi)

            # fetch demo for the sampled goals
            demos = teacher.get_demo_for_goals(sampled_goals, saved=True)
            predicted_goals = rollout_worker.predict_goal_from_demos(demos, goal_sampler_teacher.discovered_goals)

            prediction_success = [1 if (sampled_goals[pg_ind] == predicted_goal).all() else 0 for pg_ind, predicted_goal in enumerate(predicted_goals)]
            resampled_predicted_goals = goal_sampler_teacher.resample_incorrect_goal_predictions(sampled_goals, prediction_success)

            all_prediction_success = MPI.COMM_WORLD.gather(prediction_success, root=0)
            if rank == 0: print(np.mean(np.array(all_prediction_success)), 'PREDICTION SUCCESS RATE') 
            
            demos_to_be_added_buffer = [demos[demo_ind] for demo_ind, success in enumerate(prediction_success) if success]
            resampled_predicted_masks = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(len(resampled_predicted_goals))])

            language_goal_ep = None
            time_dict['goal_sampler'] += time.time() - t_i

            # Control biased initializations
            if epoch < args.start_biased_init:
                biased_init = False
            else:
                biased_init = args.biased_init

            # Environment interactions
            t_i = time.time()

            self_eval = False
            
            '''if args.cuda:
                policy.model.actor.cuda()
                policy.model.critic.cuda()
                policy.model.critic_target.cuda()'''

            episodes = rollout_worker.generate_rollout(goals=resampled_predicted_goals,  # list of goal configurations
                                                       masks=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(len(resampled_predicted_goals))]),
                                                       self_eval=self_eval,  # whether the agent performs self-evaluations
                                                       true_eval=False,  # these are not offline evaluation episodes
                                                       biased_init=biased_init,  # whether initializations should be biased.
                                                       language_goal=language_goal_ep)   # ignore if no language used
            time_dict['rollout'] += time.time() - t_i

            # Goal Sampler updates
            t_i = time.time()
            #episodes = goal_sampler.update(episodes, episode_count)
            goal_sampler_teacher.add_discovered_goals(episodes)
            time_dict['gs_update'] += time.time() - t_i

            # Storing episodes
            t_i = time.time()
            policy.store(episodes)

            # option: store additional demos for each successfully predicted goal
            if args.share_demos:
                demos_to_be_added_buffer = MPI.COMM_WORLD.allgather(demos_to_be_added_buffer)
                demos_to_be_added_buffer_shared = []
                for d in demos_to_be_added_buffer:
                    if d != []:
                        for dd in d:
                            demos_to_be_added_buffer_shared.append(dd)
                demos_to_be_added_buffer = demos_to_be_added_buffer_shared

            # store demos for which learner successfully predicted the desired goal
            if len(demos_to_be_added_buffer) > 0:

                for demo_ind, _ in enumerate(demos_to_be_added_buffer):
                    demos_to_be_added_buffer[demo_ind]['is_demonstration'] = 1.0
                    demos_to_be_added_buffer[demo_ind]['reward_pedagogical'] = 0.0

                # store demos
                policy.store(demos_to_be_added_buffer)


            '''########## WITHOUT GOAL PREDICTION
            for demo_ind, _ in enumerate(demos):
                demos[demo_ind]['is_demonstration'] = 1.0
                demos[demo_ind]['reward_pedagogical'] = 0.0


            policy.store(demos)
            for e in episodes:
                policy._update_normalizer(e)
            for demo in demos:
                policy._update_normalizer(demo)
            ##########'''

            time_dict['store'] += time.time() - t_i

            # Updating observation normalization // with goal prediction we need to adjust the number of episodes with the max nb of demos to normalize
            t_i = time.time()

            # Add demos to experience episodes before normalization
            if len(demos_to_be_added_buffer) > 0:
                for demo in demos_to_be_added_buffer:
                    episodes.append(demo)
            # max nb of demos per worker added in the buffer 
            nb_max_demos = np.max(MPI.COMM_WORLD.allgather(len(demos_to_be_added_buffer)))
            # adjust so that each worker has the same number of episodes to normalize
            if len(episodes) < args.num_rollouts_per_mpi + nb_max_demos:
                for i in range(args.num_rollouts_per_mpi + nb_max_demos - len(episodes)):
                    episodes.append(episodes[i])
            for e in episodes:
                policy._update_normalizer(e)

            time_dict['norm_update'] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i

            # Behaviour cloning updates
            if args.bc:
                t_i = time.time()
                for _ in range(args.n_batches):
                    policy.train_bc()
                time_dict['policy_train_bc'] += time.time() - t_i

            # Self Imitation Learning updates
            if args.sil:
                t_i = time.time()
                for _ in range(args.n_batches):
                    policy.train_sil()
                time_dict['policy_train_sil'] += time.time() - t_i

            episode_count += args.num_rollouts_per_mpi * args.num_workers

        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()
            eval_goals = []
            if args.n_blocks == 3:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'pyramid_3', 'stack_3']
            elif args.n_blocks == 5:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3',
                                'mixed_2_3', 'trapeze_2_3', 'stack_4', 'stack_5']
            else:
                raise NotImplementedError
            for instruction in instructions:
                eval_goal = get_eval_goals(instruction, n=args.n_blocks)
                eval_goals.append(eval_goal.squeeze(0))

            eval_goals = np.array(eval_goals)
            eval_masks = np.array(np.zeros((eval_goals.shape[0], args.n_blocks * (args.n_blocks - 1) * 3 // 2)))
            episodes = rollout_worker.generate_rollout(goals=eval_goals,
                                                       masks=eval_masks,
                                                       self_eval=False,  # this parameter is overridden by true_eval
                                                       true_eval=False,  # this is offline evaluations
                                                       biased_init=False,
                                                       language_goal=language_goal)

            results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
            rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            all_rewards = MPI.COMM_WORLD.gather(rewards, root=0)
            if args.pedagogical_teacher:
                pedagogical_rewards = np.array([e['reward_pedagogical'] for e in episodes])/2
                all_reward_pedagogical = MPI.COMM_WORLD.gather(pedagogical_rewards, root=0)
            time_dict['eval'] += time.time() - t_i

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                av_rewards = np.array(all_rewards).mean(axis=0)
                global_sr = np.mean(av_res)
                if args.pedagogical_teacher:
                    av_ped_rew = np.mean(all_reward_pedagogical)
                else:
                    av_ped_rew = None
                log_and_save(goal_sampler, epoch, episode_count, av_res, av_rewards, global_sr, time_dict, av_ped_rew)

                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                    #policy.buffer.save_buffer(model_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


    return logdir




def log_and_save(goal_sampler, epoch, episode_count, av_res, av_rew, global_sr, time_dict, av_ped_rew):
    goal_sampler.save(epoch, episode_count, av_res, av_rew, global_sr, time_dict, av_ped_rew)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()


if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    ### RUN TRAINING
    logdir = launch(args)

    ### RUN TESTING
    run_test_mpi(args, logdir)
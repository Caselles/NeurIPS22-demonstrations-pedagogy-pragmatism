import argparse
import numpy as np
from mpi4py import MPI


"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # the general arguments
    parser.add_argument('--seed', type=int, default=np.random.randint(1e6), help='random seed')
    parser.add_argument('--num-workers', type=int, default=MPI.COMM_WORLD.Get_size(), help='the number of cpus to collect samples')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    # the environment arguments
    parser.add_argument('--algo', type=str, default='semantic', help="'semantic', 'continuous', 'language'")
    parser.add_argument('--agent', type=str, default='SAC', help='the RL algorithm name')
    parser.add_argument('--n-blocks', type=int, default=3, help='The number of blocks to be considered in the FetchManipulate env')
    parser.add_argument('--masks', type=bool, default=False, help='Whether or not to use masked semantic goals')
    parser.add_argument('--mask-application', type=str, default='hindsight', help='hindsight, initial or opaque')
    parser.add_argument('--biased-init', type=bool, default=True, help='use biased environment initializations')
    parser.add_argument('--start-biased-init', type=int, default=10, help='Number of epoch before biased initializations start')
    # the training arguments
    parser.add_argument('--n-epochs', type=int, default=102, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=30, help='the times to update the network')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    # the replay arguments
    parser.add_argument('--multi-criteria-her', type=bool, default=True, help='test')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--reward-type', type=str, default='per_object', help='per_object, per_relation, per_predicate or sparse')
    # The RL arguments
    parser.add_argument('--self-eval-prob', type=float, default=0.1, help='Probability to perform self-evaluation')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--alpha', type=float, default=0.2, help='entropy coefficient')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, help='Tune entropy')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--lr-entropy', type=float, default=0.001, help='the learning rate of the entropy')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--freq-target_update', type=int, default=2, help='the frequency of updating the target networks')
    # the output arguments
    parser.add_argument('--evaluations', type=bool, default=True, help='do evaluation at the end of the epoch w/ frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='the interval that save the trajectory')
    parser.add_argument('--save-dir', type=str, default='/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_learner/', help='the path to save the models')
    # the memory arguments
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--multihead-buffer', type=bool, default=False, help='use a multihead replay buffer')
    # the preprocessing arguments
    parser.add_argument('--clip-obs', type=float, default=5, help='the clip ratio')
    parser.add_argument('--normalize_goal', type=bool, default=False, help='do evaluation at the end of the epoch w/ frequency')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    # the gnns arguments
    parser.add_argument('--architecture', type=str, default='gnn', help='The architecture of the networks')
    parser.add_argument('--variant', type=int, default=1, help='1: no interaction graph, 2: with explicit interaction graph')
    parser.add_argument('--aggregation-fct', type=str, default='max', help='node-wise aggregation function')
    parser.add_argument('--readout-fct', type=str, default='sum', help='readout aggregation function')
    # the testing arguments
    parser.add_argument('--n-test-rollouts', type=int, default=1, help='the number of tests')

    # pedagogical training
    parser.add_argument('--pedagogical-teacher', type=bool, default=False, help='Train a pedagogical teacher')

    # teacher/learner training
    parser.add_argument('--learner-from-demos', type=bool, default=False, help='Train a learner from demonstrations')
    parser.add_argument('--teacher-mode', type=str, default='', help='Teacher can be naive or pedagogical')
    parser.add_argument('--demos-vs-exp', type=float, default=0.5, help='Percentage of demonstrations versus experience in HER transitions sampling')
    parser.add_argument('--bc', type=bool, default=False, help='Activate Behaviour Cloning regularization.')
    parser.add_argument('--sqil', type=bool, default=False, help='Activate Self Q-Imitation Learning.')
    parser.add_argument('--sil', type=bool, default=False, help='Activate Self-Imitation Learning on demonstrations.')
    parser.add_argument('--pragmatic-learner', type=bool, default=False, help='Activate pragmatism mechanism in the learner, will reward itself for goal prediction of its exp.')
    parser.add_argument('--reset-from-demos', type=bool, default=False, help='Reset from demonstration states.')


    # additional demos arguments
    parser.add_argument('--give-additional-demos', type=bool, default=False, help='Provide more demos at each cycle.')
    parser.add_argument('--nb-additional-demos', type=int, default=5, help='Number of additional demos to provide.')
    parser.add_argument('--share-demos', type=bool, default=False, help='All workers share demos.')



    # testing arguments
    parser.add_argument('--illustrative-example', type=bool, default=False, help='Restrict to results to illustrative example.')
    parser.add_argument('--compute-statistically-significant-results', type=bool, default=False, help='Compute statistically significant results.')
    parser.add_argument('--predictability', type=bool, default=False, help='Compute the predictability results.')
    parser.add_argument('--reachability', type=bool, default=False, help='Compute the reachability results.')
    parser.add_argument('--learner-to-test', type=str, default='', help='Specify learner to test.')

    # demo encoder baseline arguments
    parser.add_argument('--architecture-demo-encoder', type=str, default='lstm', help='Architecture of the demo encoder module')
    parser.add_argument('--lr-demo-encoder', type=float, default=0.001, help='the learning rate of the demo encoder module')
    parser.add_argument('--n-batches_demo_encoder', type=int, default=50, help='the times to update the demo encoder network')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden state dimension for lstm')
    parser.add_argument('--goal-distrib-dim', type=int, default=35, help='Number of reachable goals')
    parser.add_argument('--nb_timesteps_demos', type=int, default=400, help='Maximum number of timesteps for demonstration')
    parser.add_argument('--nb_available_demos', type=int, default=100, help='Number of saved demos per goal index')


    args = parser.parse_args()

    return args

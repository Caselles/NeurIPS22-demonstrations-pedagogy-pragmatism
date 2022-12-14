import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.replay_buffer import MultiBuffer
from rl_modules.networks import QNetworkFlat, GaussianPolicyFlat
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from updates import update_flat, update_deepsets, behavioral_cloning_update_deepsets, sil_update_deepsets
from utils import id_to_language


"""
SAC with HER (MPI-version)
"""

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class RLAgent:
    def __init__(self, args, compute_rew, goal_sampler):

        self.args = args
        self.alpha = args.alpha
        self.env_params = args.env_params

        self.goal_sampler = goal_sampler

        self.total_iter = 0
        self.total_iter_bc = 0
        self.total_iter_sil = 0

        self.freq_target_update = args.freq_target_update

        # create the network
        self.architecture = self.args.architecture

        if self.architecture == 'flat':
            self.actor_network = GaussianPolicyFlat(self.env_params)
            self.critic_network = QNetworkFlat(self.env_params)
            # sync the networks across the CPUs
            sync_networks(self.actor_network)
            sync_networks(self.critic_network)

            # build up the target network
            self.critic_target_network = QNetworkFlat(self.env_params)
            hard_update(self.critic_target_network, self.critic_network)
            sync_networks(self.critic_target_network)

            # create the optimizer
            self.policy_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        elif self.architecture == 'deepsets':
            if args.algo == 'language':
                from rl_modules.language_models import DeepSetLanguage
                self.model = DeepSetLanguage(self.env_params, args)
            elif args.algo == 'continuous':
                from rl_modules.continuous_models import DeepSetContinuous
                self.model = DeepSetContinuous(self.env_params, args)
            else:
                from rl_modules.semantic_models import DeepSetSemantic
                self.model = DeepSetSemantic(self.env_params, args)
            # sync the networks across the CPUs
            sync_networks(self.model.critic)
            sync_networks(self.model.actor)
            hard_update(self.model.critic_target, self.model.critic)
            sync_networks(self.model.critic_target)

            # create the optimizer
            self.policy_optim = torch.optim.Adam(list(self.model.actor.parameters()),
                                                 lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(list(self.model.critic.parameters()),
                                                 lr=self.args.lr_critic)
        elif self.architecture == 'gnn':
            if self.args.variant == 2:
                from rl_modules.gnn_models_v2 import GnnSemantic
            else:
                from rl_modules.gnn_models import GnnSemantic
            self.model = GnnSemantic(self.env_params, args)
            # sync the networks across the CPUs
            sync_networks(self.model.critic)
            sync_networks(self.model.actor)
            hard_update(self.model.critic_target, self.model.critic)
            sync_networks(self.model.critic_target)

            # create the optimizer
            self.policy_optim = torch.optim.Adam(list(self.model.actor.parameters()),
                                                 lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(list(self.model.critic.parameters()),
                                                 lr=self.args.lr_critic)

        else:
            raise NotImplementedError

        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)

        # if use GPU
        if self.args.cuda:
            self.model.actor.cuda()
            self.model.critic.cuda()
            self.model.critic_target.cuda()

        # Target Entropy
        if self.args.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.env_params['action'])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr_entropy)

        # her sampler
        if args.algo == 'continuous':
            self.continuous_goals = True
        else:
            self.continuous_goals = False
        if args.algo == 'language':
            self.language = True
        else:
            self.language = False
        self.her_module = her_sampler(self.args, compute_rew)

        # create the replay buffer
        self.buffer = MultiBuffer(env_params=self.env_params,
                                  buffer_size=self.args.buffer_size,
                                  sample_func=self.her_module.sample_her_transitions,
                                  multi_head=self.args.multihead_buffer if not self.language else False,
                                  goal_sampler=self.goal_sampler,
                                  pedagogical=self.args.pedagogical_teacher,
                                  pragmatic=self.args.pragmatic_learner,
                                  learner_from_demos=self.args.learner_from_demos
                                  )

        self.bc_loss = []
        self.sqil = args.sqil
        self.sil_loss = []

    def act(self, obs, ag, g, mask, no_noise, language_goal=None):
        # apply mask
        if mask is not None:
            if self.args.mask_application == 'hindsight':
                g = g * (1 - mask) + ag * mask
            elif self.args.mask_application == 'initial':
                g = g * (1 - mask) + ag * mask
            elif self.args.mask_application == 'opaque':
                g = g * (1 - mask) - 10 * mask
            else:
                raise NotImplementedError
        with torch.no_grad():
            # normalize policy inputs
            obs_norm = self.o_norm.normalize(obs)
            ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)

            if self.language:
                g_norm = g
            else:
                g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
            if self.architecture == 'gnn':
                obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
                if self.args.algo == 'language':
                    self.model.policy_forward_pass(obs_tensor, no_noise=no_noise, language_goal=language_goal)
                elif self.args.algo == 'continuous':
                    self.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, no_noise=no_noise)
                else:
                    if self.args.cuda:
                        self.model.policy_forward_pass(obs_tensor.cuda(), ag_norm.cuda(), g_norm.cuda(), no_noise=no_noise)
                    else:
                        self.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, no_noise=no_noise)
                if self.args.cuda:
                    action = self.model.pi_tensor.cpu().numpy()[0]
                else:
                    action = self.model.pi_tensor[0].numpy()

            else:
                input_tensor = self._preproc_inputs(obs, ag, g)
                action = self._select_actions(input_tensor, no_noise=no_noise)

        return action.copy()

    def get_action_proba(self, action, obs, ag, g, mask, language_goal=None):
        # apply mask
        if mask is not None:
            if self.args.mask_application == 'hindsight':
                g = g * (1 - mask) + ag * mask
            elif self.args.mask_application == 'initial':
                g = g * (1 - mask) + ag * mask
            elif self.args.mask_application == 'opaque':
                g = g * (1 - mask) - 10 * mask
            else:
                raise NotImplementedError
        with torch.no_grad():
            # normalize policy inputs
            obs_norm = self.o_norm.normalize(obs)
            ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32)

            if self.language:
                g_norm = g
            else:
                g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32)
            if self.architecture == 'gnn':
                obs_tensor = torch.tensor(obs_norm, dtype=torch.float32)
                if self.args.cuda:
                    action_proba = self.model.get_action_proba(action.cuda(), obs_tensor.cuda(), ag_norm.cuda(), g_norm.cuda(), cuda=self.args.cuda)
                else:
                    action_proba = self.model.get_action_proba(action, obs_tensor, ag_norm, g_norm)

        return action_proba
    
    def store(self, episodes):
        self.buffer.store_episode(episode_batch=episodes)

    # pre_process the inputs
    def _preproc_inputs(self, obs, ag, g):
        obs_norm = self.o_norm.normalize(obs)
        ag_norm = self.g_norm.normalize(ag)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, ag_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def train(self):
        # train the network with RL
        self.total_iter += 1
        self._update_network()

        # soft update
        if self.total_iter % self.freq_target_update == 0:
            if self.architecture == 'gnn':
                self._soft_update_target_network(self.model.critic_target, self.model.critic)
            else:
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

    def train_bc(self):
        # train the network with Behavioural Cloning 
        self.total_iter_bc += 1
        self._update_network(bc=True)

    def train_sil(self):
        # train the network with Behavioural Cloning 
        self.total_iter_sil += 1
        self._update_network(sil=True)

    def _select_actions(self, state, no_noise=False):
        if not no_noise:
            action, _, _ = self.actor_network.sample(state)
        else:
            _, _, action = self.actor_network.sample(state)
        return action.detach().cpu().numpy()[0]

    # update the normalizer
    def _update_normalizer(self, episode):

        mb_obs = episode['obs']
        mb_ag = episode['ag']
        mb_g = episode['g']
        mb_masks = episode['masks']
        mb_actions = episode['act']
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0]
        # create the new buffer to store them
        buffer_temp = {'obs': np.expand_dims(mb_obs, 0),
                       'ag': np.expand_dims(mb_ag, 0),
                       'g': np.expand_dims(mb_g, 0),
                       'masks': np.expand_dims(mb_masks, 0),
                       'actions': np.expand_dims(mb_actions, 0),
                       'obs_next': np.expand_dims(mb_obs_next, 0),
                       'ag_next': np.expand_dims(mb_ag_next, 0),
                       }
        # if 'language_goal' in episode.keys():
        #     buffer_temp['language_goal'] = np.array([episode['language_goal'] for _ in range(mb_g.shape[0])], dtype='object').reshape(1, -1)
        if 'lg_ids' in episode.keys():
            buffer_temp['lg_ids'] = np.expand_dims(episode['lg_ids'], 0)

        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions, normalization=True)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        # recompute the stats
        self.o_norm.recompute_stats()

        if self.args.normalize_goal:
            self.g_norm.update(transitions['g'])
            self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, bc=False, sil=False):

        # sample from buffer, this is done with LP is multi-head is true
        transitions = self.buffer.sample(self.args.batch_size, bc=bc, sqil=self.sqil, sil=sil)

        # pre-process the observation and goal
        o, o_next, g, ag, ag_next, actions, rewards = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag'], \
                                                      transitions['ag_next'], transitions['actions'], transitions['r']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['ag'] = self._preproc_og(o, ag)
        _, transitions['ag_next'] = self._preproc_og(o, ag_next)

        # apply normalization
        obs_norm = self.o_norm.normalize(transitions['obs'])
        if self.language:
            g_norm = transitions['g']
            lg_ids = transitions['lg_ids']
            language_goals = np.array([id_to_language[lg_id] for lg_id in lg_ids])
            # language_goals = transitions['language_goal']
        else:
            g_norm = self.g_norm.normalize(transitions['g'])
            language_goals = None
        ag_norm = self.g_norm.normalize(transitions['ag'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])

        anchor_g = transitions['g']

        if self.architecture == 'flat':
            critic_1_loss, critic_2_loss, actor_loss, alpha_loss, alpha_tlogs = update_flat(self.actor_network, self.critic_network,
                                                                           self.critic_target_network, self.policy_optim, self.critic_optim,
                                                                           self.alpha, self.log_alpha, self.target_entropy, self.alpha_optim,
                                                                           obs_norm, ag_norm, g_norm, obs_next_norm, actions, rewards, self.args)
        elif self.architecture == 'gnn' and not (bc or sil):
            critic_1_loss, critic_2_loss, actor_loss, alpha_loss, alpha_tlogs = update_deepsets(self.model, self.language,
                                                                               self.policy_optim, self.critic_optim, self.alpha, self.log_alpha,
                                                                               self.target_entropy, self.alpha_optim, obs_norm, ag_norm, g_norm,
                                                                               obs_next_norm, ag_next_norm, anchor_g, actions, rewards, language_goals, self.args)
        elif bc:
            loss = behavioral_cloning_update_deepsets(self.model, self.language, self.policy_optim, obs_norm, ag_norm, g_norm, 
                obs_next_norm, ag_next_norm, actions, rewards, self.args)
            
            self.bc_loss.append(loss)
            if self.total_iter_bc % 10 == 0:
                print(np.mean(self.bc_loss))

        elif sil:
            policy_loss, value_loss, total_loss, v_error = sil_update_deepsets(self.model, self.language, self.policy_optim, self.critic_optim, self.alpha, obs_norm, ag_norm, g_norm, 
                obs_next_norm, ag_next_norm, actions, rewards, transitions['returns'], self.args)
            
            self.sil_loss.append([policy_loss, value_loss, total_loss])
            if self.total_iter_sil % 10 == 0:
                print(np.mean(self.sil_loss, axis=0), 'mean policy loss, mean value loss, mean total loss')
                print([policy_loss, value_loss, total_loss], 'policy loss, value loss, total loss')
                print(torch.mean(v_error), 'clipped value error')

        else:
            raise NotImplementedError


    def save(self, model_path, epoch):
        # Store model
        if self.args.architecture == 'flat':
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                        self.actor_network.state_dict(), self.critic_network.state_dict()],
                       model_path + '/model_{}.pt'.format(epoch))
        elif self.args.architecture == 'gnn':
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                        self.model.actor.state_dict(), self.model.critic.state_dict()],
                       model_path + '/model_{}.pt'.format(epoch))
        else:
            raise NotImplementedError

    def load(self, model_path, args):

        if args.architecture == 'deepsets':
            o_mean, o_std, g_mean, g_std, phi_a, phi_c, rho_a, rho_c, enc = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.single_phi_actor.load_state_dict(phi_a)
            self.model.single_phi_critic.load_state_dict(phi_c)
            self.model.rho_actor.load_state_dict(rho_a)
            self.model.rho_critic.load_state_dict(rho_c)
            self.model.critic_sentence_encoder.load_state_dict(enc)
            self.o_norm.mean = o_mean
            self.o_norm.std = o_std
            self.g_norm.mean = g_mean
            self.g_norm.std = g_std
        else:
            o_mean, o_std, g_mean, g_std, actor, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.actor.load_state_dict(actor)
            self.model.critic.load_state_dict(critic)
            self.model.critic_target.load_state_dict(critic)
            # self.actor_network.eval()
            self.o_norm.mean = o_mean
            self.o_norm.std = o_std
            self.g_norm.mean = g_mean
            self.g_norm.std = g_std
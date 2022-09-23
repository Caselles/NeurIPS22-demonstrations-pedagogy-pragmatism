import torch
from torch import nn
import numpy as np
from mpi_utils.mpi_utils import sync_networks, sync_grads
from demonstrations_buffer import MultiDemoBuffer
from networks import DemoEncoderNetwork
from utils_demo import prepad_zeros, select_obs_act_in_episode
from mpi_utils.normalizer import normalizer
from mpi4py import MPI



"""
Reward function module (MPI-version)
"""

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DemoEncoder:
    def __init__(self, args, gpu=False):

        self.args = args
        self.gpu = gpu
        self.env_params = args.env_params

        self.total_iter = 0

        self.demo_encoder_loss_mean = 0


        # create the network
        self.architecture = self.args.architecture_demo_encoder

        if self.architecture == 'lstm':
            self.model = DemoEncoderNetwork(args.env_params, args.hidden_dim, args.goal_distrib_dim)
            if self.gpu:
                pass
            else:
                sync_networks(self.model)
        

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.demo_encoder_optim = torch.optim.Adam(list(self.model.parameters()),
                                             lr=self.args.lr_demo_encoder)



        # if use GPU
        if self.gpu:
            self.model.cuda()
            import os
            os.system('nvidia-smi')


        # create the buffer
        self.buffer = MultiDemoBuffer(env_params=self.env_params,
                                  buffer_size=self.args.buffer_size)

    
    def store(self, batch):
        self.buffer.store_batch(batch=batch)

    def sample(self, batch_size):
        batch = self.buffer.sample(batch_size=batch_size)

        return batch

    def encode(self, demos, no_grad=True, gpu=False):

        self.model.eval()

        if no_grad:
            with torch.no_grad():
                if gpu:
                    predicted_goals = self.model.forward(torch.tensor(demos, dtype=torch.float32).cuda())
                else:
                    predicted_goals = self.model.forward(torch.tensor(demos, dtype=torch.float32))
                #print(np.argmax(predicted_goals.detach().cpu(), axis=1), 'PRED GOALS ENCODE', MPI.COMM_WORLD.Get_rank())
                #for demo in demos:
                #    predicted_goal = self.model(torch.tensor([demo], dtype=torch.float32))
                #    predicted_goals.append(predicted_goal.numpy())

        return np.array(predicted_goals.detach().cpu())

    def train(self):
        # train the network
        self.total_iter += 1
        self._update_network()
        if self.total_iter % 100 == 0:
            print('demo encoder')
            print(self.demo_encoder_loss_mean/self.total_iter)

    # update the network
    def _update_network(self):

        # sample from buffer and reformat
        data = self.buffer.sample(self.args.batch_size)

        # retrieve demo in episode
        demos = select_obs_act_in_episode(data[:,0])

        inputs_demos = torch.tensor(demos, dtype=torch.float32) 

        #target goal is the index of the chosen goals among the 35 valid goals (np.where)
        target_goals = np.where(np.array(list(data[:,1])) == 1)[1] #... the np.array(list()) is ridiculous, has to be changed
        target_goals = torch.tensor(target_goals, dtype=torch.long) 

        if self.gpu:
            inputs_demos = inputs_demos.cuda()
            target_goals = target_goals.cuda()

        # forward pass
        self.model.train() # train mode
        pred_goals = self.model(inputs_demos)
        

        # loss computing
        demo_encoder_loss = self.cross_entropy_loss(pred_goals, target_goals)
        self.demo_encoder_loss_mean += demo_encoder_loss.item()

        # optimization
        self.demo_encoder_optim.zero_grad()
        demo_encoder_loss.backward()
        #sync_grads(self.model)
        self.demo_encoder_optim.step()

        if self.total_iter % 1 == 0:
            pred = np.argmax(pred_goals.detach().cpu(), axis=1)
            score = sum(pred == target_goals.detach().cpu())
            print(int(score)/len(pred), 'score demo encoder')
            """for i, _ in enumerate(pred):
                if pred[i] != target_goals[i].detach().cpu():
                    print(pred[i],target_goals[i].detach().cpu())
                    print("FAILED")
                else:
                    #print(pred[i],target_goals[i].detach().cpu())
                    #print("SUCCESS")
                    pass"""

        return demo_encoder_loss.item()


    def save(self, model_path, epoch, training=False):
        # Store model
        if self.args.architecture_demo_encoder == 'lstm':
            if training:
                torch.save([self.model.state_dict()],
                       model_path + '/model_demo_encoder_training_{}.pt'.format(epoch))
            else:
                torch.save([self.model.state_dict()],
                       model_path + '/model_demo_encoder_{}.pt'.format(epoch))
        else:
            raise NotImplementedError
        return
        

    def load(self, model_path, args):
        # Load model
        if self.args.architecture_demo_encoder == 'lstm':
            demo_encoder_model = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(demo_encoder_model)
        else:
            raise NotImplementedError
        return

    def sync_demo_encoders(self, model_path, epoch):

        if self.args.architecture_demo_encoder == 'lstm':
            demo_encoder_model = torch.load(model_path + '/model_demo_encoder_training_{}.pt'.format(epoch), 
                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(demo_encoder_model[0])

        '''demo_encoder_model = torch.load('/home/gohu/workspace/postdoc/decstr/decstr_gangstr/output/2021-11-07 18:06:43_FetchManipulate3Objects-v0_gnn_per_object/models/model_demo_encoder_training_58.pt', 
                map_location=lambda storage, loc: storage)
        self.model.load_state_dict(demo_encoder_model[0])

        print('model loadedddd')'''



        return True

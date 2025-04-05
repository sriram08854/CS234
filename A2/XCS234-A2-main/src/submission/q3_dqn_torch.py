import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger, join
from utils.test_env import EnvTest
from .q1_schedule import LinearExploration, LinearSchedule
from .q2_linear_torch import Linear

import yaml

yaml.add_constructor("!join", join)

config_file = open("config/q3_dqn.yml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

############################################################
# Problem 3: Implementing DeepMind's DQN
############################################################

class NatureQN(Linear):
    """
    Implementation of DeepMind's Nature paper, please consult the methods section
    of the paper linked below for details on model configuration.
    (https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
    """

    ############################################################
    # Problem 3a: initialize_models

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The in_channels 
        to Conv2d networks will be n_channels * self.config["hyper_params"]["state_history"]

        Args:
            q_network (torch model): variable to store our q network implementation
            target_network (torch model): variable to store our target network implementation

        TODO:
             (1) Set self.q_network to the architecture defined in the Nature paper associated to this question.
                Padding isn't addressed in the paper but here we will apply padding of size 2 to each dimension of
                the input to the first conv layer (this should be an argument in nn.Conv2d).
             (2) Set self.target_network to be the same configuration as self.q_network but initialized from scratch.
             (3) Be sure to use nn.Sequential in your implementation.
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        ### START CODE HERE ###
        # Determine the number of input channels (n_channels * state_history)
        state_history = self.config["hyper_params"]["state_history"]
        in_channels = n_channels * state_history

        # Create a dummy input to determine the flattened size after convolutions.
        dummy_input = torch.zeros(1, in_channels, img_height, img_width)
        conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out = conv(dummy_input)
        flattened_size = conv_out.view(1, -1).shape[1]

        # Build the Q-network as described in the Nature paper.
        self.q_network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Create the target network with the same architecture, but with independent parameters.
        self.target_network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        ### END CODE HERE ###

    ############################################################
    # Problem 3b: get_q_values

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state (torch tensor): shape = (batch_size, img height, img width,
                                            n_channels x config["hyper_params"]["state_history"])
            network (str): The name of the network, either "q_network" or "target_network"

        Returns:
            out (torch tensor): shape = (batch_size, num_actions)

        TODO:
            Perform a forward pass of the input state through the selected network
            and return the output values.

        Hints:
            (1) You can forward a tensor through a network by simply calling it (i.e. network(tensor))
            (2) Look up torch.permute (https://pytorch.org/docs/stable/generated/torch.permute.html)
        """
        ### START CODE HERE ###
        # The state tensor is expected to have shape:
        # (batch_size, img_height, img_width, n_channels * state_history)
        # We need to rearrange it to (batch_size, n_channels * state_history, img_height, img_width)
        state = state.permute(0, 3, 1, 2).contiguous()

        if network == "q_network":
            net = self.q_network
        elif network == "target_network":
            net = self.target_network
        else:
            raise ValueError("Network must be either 'q_network' or 'target_network'")
        
        out = net(state)
        ### END CODE HERE ###
        return out

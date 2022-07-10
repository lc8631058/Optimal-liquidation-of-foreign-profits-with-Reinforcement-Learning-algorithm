# This is a environment for optimal FX liquidation problem,
# the states generated by this environment only contain historical FX rates,
# not the FX rates from a Generalized OU model

import random
import numpy as np

class MarketEnvironment():
    def __init__(self,
                 random = 0,
                 liquid_time = 100,
                 total_vol = 100,
                 v_min = 0,
                 v_max = 10):
        """
        output states for agent
        :param random: random seed
        :param liquid_time: period of liquidation
        :param total_vol: total volume (currency) of liquidation, unit is million
        :param num_trades: number of trades with liquid_time
        :n_min: the minimum volume of execution at each step,
        :n_max: the maximum volume of execution at each step,
        """
        self.random = random
        self.total_volume = total_vol
        self.liquid_time = liquid_time
        # self.num_trades = num_trades
        self.v_min = v_min
        self.v_max = v_max
        # self.naive_volume = self.total_volume / self.num_trades

        # Set the variables for the initial stat
        self.volume_remain_x = self.total_volume
        self.time_remain_mu = self.liquid_time

        # Set the initial transaction state to False
        self.transacting = False

    def reset(self,
              random = 0,
              liquid_time = None,
              total_vol = None,
              # num_trades = None,
              v_min = None,
              v_max = None):
        self.__init__(random = random,
                     liquid_time = liquid_time,
                     total_vol = total_vol,
                     # num_trades = num_trades,
                     v_min = v_min,
                     v_max = v_max)

    def run_newEpisode(self, price_traj):
        """
        put a new episode to the Env for another period of liquidation
        :params price_traj: new FX rates trajectory
        :return: Env with new episode
        """
        # FX rates trajectory, each with length liquid_time
        self.price_trajectory = price_traj
        # initialize price
        self.prevPrice_S = self.price_trajectory[:, 0].reshape([-1, 1])

    def get_initial(self):
        """
        get the initial state for trading start
        :return: initial state
        """
        # Set time step variable k to keep track of the trade number
        self.k = np.zeros([self.prevPrice_S.shape[0], 1], dtype=np.int32)
        # Set the initial state to [x_0, v_0, T]
        self.initial_state = np.concatenate([self.prevPrice_S,
                                       np.repeat(self.total_volume, self.prevPrice_S.shape[0]).reshape([-1, 1]),
                                       np.repeat(self.liquid_time, self.prevPrice_S.shape[0]).reshape([-1, 1])], 1)
        return self.initial_state

    def start_transactions(self):
        """
        start transaction
        :return:
        """
        # Set transactions on
        self.transacting = np.ones([self.prevPrice_S.shape[0], 1], dtype=bool)

        # action collection
        self.action_list = []

        # reward collection
        self.reward_list = []

        # score
        self.score = np.zeros([self.prevPrice_S.shape[0], 1], dtype=np.float32)

    def Naive_reward(self):
        """
        calculate the culmulative reward using Naive strategy
        """
        n = self.total_volume // self.v_min
        remain = self.total_volume - self.v_min * n
        interval = self.liquid_time // n
        total_reward = np.zeros([self.prevPrice_S.shape[0], 1], dtype=np.float32)
        cnt = 0
        for t in range(self.liquid_time + 1):
            if t == 0 or (cnt > 0 and cnt % interval == 0):
                total_reward += (self.v_min / np.exp(self.price_trajectory[:, t])).reshape([-1, 1])
                cnt = 0
            else:
                cnt += 1
        total_reward += (remain / np.exp(self.price_trajectory[:, self.liquid_time])).reshape([-1, 1])
        return total_reward

    def step(self, action, cur_state):
        """
        trading for one step
        :param action: percentage, indicates trade [action] percente of N_MAX
        :return:
        """
        action_mask = action != 0
        trade_vol = np.zeros_like(action_mask, dtype=np.int32)
        trade_vol[action_mask] = action[action_mask] + (self.v_min - 1)
        # if action == 0:
        #     trade_vol = action
        # else:
        #     trade_vol = action + (self.v_min - 1)

        # Create a class that will be used to keep track of information about the transaction
        class Info(object):
            pass

        info = Info()

        # Set the done flag to False to indicate the liquidation is not finished.
        info.done = np.zeros([self.prevPrice_S.shape[0], 1], dtype=bool)

        # current price
        info.cur_price = cur_state[:, 0].reshape([-1, 1])
        self.volume_remain_x = cur_state[:, 1].reshape([-1, 1])
        self.time_remain_mu = cur_state[:, 2].reshape([-1, 1])

        # Start trading
        # If it's the last transaction step, i.e. at time T, v = v_remaining
        # if self.time_remain_mu == 0 or self.volume_remain_x < trade _vol:
        # Mask for condition: if self.time_remain_mu == 0: trade_vol = self.volume_remain_x
        time_mask_1 = self.time_remain_mu == 0
        if time_mask_1.any():
            trade_vol[self.transacting * time_mask_1] = \
                self.volume_remain_x[self.transacting * time_mask_1]
        # Mask for condition: if self.time_remain_mu > 0 and self.volume_remain_x < trade_vol: trade_vol = 0
        time_mask_2 = np.logical_and(self.time_remain_mu > 0, self.volume_remain_x < trade_vol)
        if time_mask_2.any():
            trade_vol[self.transacting * time_mask_2] = 0

        # if self.time_remain_mu == 0:
        #     trade_vol = self.volume_remain_x
        # elif self.time_remain_mu > 0 and self.volume_remain_x < trade_vol:
        #     trade_vol = 0

        ##### Update the variable in state for the next step #####
        # Update the number of volumes remaining
        self.volume_remain_x -= trade_vol * self.transacting

        # Update time left mu
        self.time_remain_mu -= 1

        # Update new price S_euro,c
        # Calculate the reward and append it to the reward collection
        info.reward_r = trade_vol / np.exp(info.cur_price) * self.transacting

        # During training, if the agent does not sell all the volume before the given number of trades or
        # if the total  number of shares remaining is equal to 0, then transaction stops, set the done flag
        # to true.

        # check the state END condition
        end_mask = np.logical_and(self.transacting, np.logical_or(self.time_remain_mu == -1, abs(self.volume_remain_x) == 0))
        # if self.transacting and (self.time_remain_mu == -1 or abs(self.volume_remain_x) == 0):
        self.transacting[end_mask] = False
        info.done[end_mask] = True

        self.k += 1
        # if self.k[0] == 100:
        #     a = 1
        # self.price_collect.append(info.price)
        # Set the new state s_k+1
        if self.k[0] == self.liquid_time + 1:
            info.next_price = np.tile(-1., [self.prevPrice_S.shape[0], 1])
        else:
            info.next_price = self.price_trajectory[:, self.k[0]]

        # the pred_params can be fixed on the first step or changing over time k
        next_state_s = np.concatenate([info.next_price,
                                      self.volume_remain_x,
                                      self.time_remain_mu], axis=1)

        # # add (s_k, a_k, r_k, s_k+1) to replay buffer
        # self.pool.append([state_s[:], action, self.pred_params])

        return (next_state_s, info.reward_r, info.done, info)

    def stop_transaction(self):
        self.transacting = False














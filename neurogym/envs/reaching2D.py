#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ADAPDET FROM ReachingDelayResponse

import numpy as np

import neurogym as ngym
from neurogym import spaces

#%% helper functions
def gauss(x,sd=1,mu=0):
    '''
    Gaussian function

    Parameters
    ----------
    x : numpy array 
        Axis for the gaussian function.
    sd : float, optional
        Standard deviation. The default is 1.
    mu : float, optional
        Mean. The default is 0.

    Returns
    -------
    numpy array
        Gaussian.

    '''
    return np.exp(-((x-mu)/sd)**2/2 )


def accuracy_function(gt, act, output_mask=None):
     gt = np.squeeze(np.array(gt))
     act = np.squeeze(np.array(act))
     end_point = np.cumsum(act,axis=0)
     gt_end_point = np.cumsum(gt,axis=0)
     if len(gt.shape) > 1:
         end_point_acc = np.sqrt(np.sum((end_point-gt_end_point)**2,axis=1))
         acc = np.sum(np.sqrt(np.sum((act-gt)**2,axis=1)))
     else:
         end_point_acc = np.sqrt(np.sum((end_point-gt_end_point)**2))
         acc = np.sum(np.sqrt(np.sum((act-gt)**2)))
         
     return end_point_acc, acc


#%% THE TASK
class Reaching2D(ngym.TrialEnv):
    r"""Reaching task with a delay period.

    A reaching direction is presented by the stimulus during the stimulus
    period. Followed by a delay period, the agent needs to respond to the
    direction of the stimulus during the decision period.
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': ['delayed response', 'continuous action space',
                 'multidimensional action space', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None,
                 box_size=1.,mem=True,reset_pos=True):
        super().__init__(dt=dt)
        # we use box_size define the bounding box
        self.box_size = box_size 
        # whether or not the stim stays on
        self.mem = mem
        # tracking the current position
        self.pos = 0
        self.vel = np.zeros(int(self.tmax/self.dt))
        self.reset_pos=reset_pos

        # Rewards (different kinds possible)
        self.rewards = {'fail': -0., 'correct': +1., 'ongoing': +0.} 
        # experiment logic: we only have rewards no punishments
        # technically wait time and the no-reward-sound can be seen as a punishment
        # it might be conceivable to add different reward styles here
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'stimulus': 500,
            'delay': (0, 1000, 2000),
            'movement': 500}
        if timing:
            self.timing.update(timing)

        #self.r_tmax = self.rewards['fail'] #necessary??
        # self.r_tmax would be added to reward whenever trial has finished.
        self.abort = False

        # define spaces
        name = {'fix': 0, 'x': 1}
        self.observation_space = spaces.Box(low=np.array([0.0, -box_size]),
                                            high=np.array([1.0, box_size]),
                                            dtype=np.float32, name=name)
        
        # action space has one dimension less bc it does not produce fix
        # action will be speed
        self.action_space = spaces.Box(low=-box_size,
                                       high=box_size,
                                       shape=(1,),
                                       dtype=np.float32)

    def _new_trial(self, **kwargs):
        # Trial
        
        # generate the random parameter
        # this would be the position x,y
        # for now only x = ground_truth
        trial = {
            'ground_truth': self.rng.uniform(-self.box_size, self.box_size)
        }
        trial.update(kwargs)
        stim_pos = trial['ground_truth']

        # Setting time periods for this trial
        # Will add stimulus, delay and decision periods sequentially  
        # unsing self.timing info
        self.add_period(['stimulus', 'delay', 'movement'])

        # set obersvation space
        ## Add ground_truth_stim to stimulus period at stimulus location
        ##self.add_ob(ground_truth_stim, 'stimulus', where='x')
        # we could do adding but we just set a new position everytime
        
        # does the stimulus stay on ruding memory period?
        if self.mem:
            mem_stim_pos = 0
        else:
            mem_stim_pos = stim_pos
                        
        # set obesrvations
        self.set_ob([1, stim_pos], 'stimulus')
        self.set_ob([1, mem_stim_pos], 'delay')
        self.set_ob([0, mem_stim_pos], 'movement')

        # set ground truth
        self.set_groundtruth(0, ['stimulus', 'delay'])
        
        # speed profile from current pos to new pos
        # now as a test just direction +-1
        move_i_dur = self.end_ind['movement'] - self.start_ind['movement']
        
        #dist_to_move = stim_pos - self.pos # does not work, stim pos is always 0
        mu_gauss = move_i_dur/2.0
        sd_gauss = move_i_dur/8.0
        gauss_dist = gauss(np.arange(move_i_dur),mu=mu_gauss,sd=sd_gauss)
        move_speed = stim_pos*gauss_dist/sum(gauss_dist)
        move_speed = np.expand_dims(move_speed, 1)
        self.set_groundtruth(move_speed, 'movement')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0 # default reward 0 unless something happens
        gt = self.gt_now  # current ground truth
        ob = self.ob_now # current observation
        # gt and action will be 2D but now are 1D
        
        
        # if we really want to change start positon than we would need to do that
        # in a smart way, maybe we need to update gt here itself?

        # track velocity
        i_now = int(self.t/self.dt)
        self.vel[i_now] = action
        

        # track position
        if (self.t == 0) & self.reset_pos:
            self.pos = 0
        else:
            self.pos += action
            
        # reward per step
        _, acc = accuracy_function(gt, action)
        reward = self.rewards['ongoing']/(1+acc)
                                          
        # this should be checked at the end of the trial
        if self.t + self.dt > self.tmax:
            new_trial = True 
            # here we could do endpoint reward
            acc_end_point, acc = accuracy_function(self.gt, np.array(self.vel))
            reward = self.rewards['correct']/(1+acc)
            self.performance = reward/self.rewards['correct']
        
        if ob[0] == 1:
            # not in movement period
            # abort if movemet hallens
            if action > 0.1:
                new_trial = self.abort
                reward = self.rewards['fail']

        # return 
        # - current observation (predefined in _new_trial here)
        # - current reward
        # - task ends (never, since there is no end to this task)
        # - dict with additional info
        # -- here we store ground truth (position) and 
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}

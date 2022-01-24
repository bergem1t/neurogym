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
   
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.stack([rho,phi],axis=0)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.stack([x,y],axis=0)


#%% THE TASK
class Reaching2D(ngym.TrialEnv):
    r"""Reaching task with a delay period.

    A reaching direction is presented by the stimulus during the stimulus
    period. Followed by a delay period, the agent needs to respond to the
    direction of the stimulus during the decision period. The response is the 
    2D velocity to reach the stimulus from the agent's starting position. The
    task can be in a memory version (mem=True) in which case the stimulus 
    dissapears after the stimulus period.
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': ['delayed response', 'continuous action space',
                 'multidimensional action space', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None,
                 box_size=1.,mem=True,context=None,reset_pos=True,target_pos=None):
        super().__init__(dt=dt)
        # we use box_size to define the bounding box
        self.box_size = box_size 
        # whether or not the stim stays on
        self.mem = mem
        # context cue with stim cue. The input context defines what kind of context in a str
        if isinstance(context,str):
          context = context.lower()
        assert context in [None,'andback','anti'], 'context must be a valid string defining the context'
        self.context = context
        # tracking the current position
        # Position is not implemented and probaly not necessary if we consider
        # hand-centered coordinates
        self.pos = [0, 0]
        self.vel = np.zeros([2,int(self.tmax/self.dt)])
        self.reset_pos=reset_pos
        
        # check the target pos arrangement
        # Not sure if I'm doing it right
        # It might be reasonable to have a discrete observation space for defined
        # target positions!? 
        # Do I even use the spaces correctly??
        if isinstance(target_pos,list):
          target_pos = np.array(target_pos)
        
        if isinstance(target_pos,int):
          # center out reach task and target pos indicates number of directions
          tgt_pos_angle=np.arange(0,2*np.pi,2*np.pi/target_pos)
          # eccentricity is fix at 80% box_size
          self.target_pos = pol2cart(0.8, tgt_pos_angle)
          
        elif isinstance(target_pos,np.ndarray):
          # possible target positions are predefinde
          assert (target_pos.shape[0] == 2) and (target_pos.ndim==2), 'If target_pos has predefined target positions, it must be a 2xN array'
          assert (target_pos.max() <= 1) and (target_pos.min() >= -1), 'Target positions must be defined between -1 and 1 relative to box_size'
          self.target_pos = target_pos
          
        else:
          # positions will be random within the box
          self.target_pos = None
          
        # Rewards (different kinds possible)
        self.rewards = {'fail': -0., 'correct': +1., 'ongoing': +0.} 
        # experiment logic: we only have rewards no punishments
        # technically wait time and the no-reward-sound can be seen as a punishment
        # it might be conceivable to add different reward styles here
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'baseline': 500,
            'stimulus': 500,
            'delay': (0, 500, 1000),
            'movement': 500}
        if timing:
            self.timing.update(timing)

        #self.r_tmax = self.rewards['fail'] #necessary??
        # self.r_tmax would be added to reward whenever trial has finished.
        self.abort = False

        # define spaces
        name = {'fixation': 0, 'target':[1, 2], 'context': 3}
        self.observation_space = spaces.Box(low=np.array([0.0, -box_size, -box_size, 0.0]),
                                            high=np.array([1.0, box_size, box_size, 1.0]),
                                            name=name)#,
                                            #dtype=np.float32, name=name)
        
        # action space has one dimension less bc it does not produce fix
        # action will be speed
        self.action_space = spaces.Box(low=-box_size,
                                       high=box_size,
                                       shape=(2,),name={'target':[0,1]})#,
                                       #dtype=np.float32)

    def _new_trial(self, **kwargs):
        """
        Generate data and ground-truth for a new trial.
  
        Parameters
        ----------
        **kwargs : TYPE
          DESCRIPTION.
  
        Returns
        -------
        trial : TYPE
          DESCRIPTION.
  
        """
      
        # Trial
        
        # generate the random parameter
        # this would be the position x,y
        if isinstance(self.target_pos,np.ndarray):
          # slect predefined target
          #TODO: make sure that the selected targets are balanced?!
          tgts = self.target_pos
          i_target = self.rng.randint(tgts.shape[1])
          trial = {
              'stim_pos': np.array(tgts[:,i_target]*self.box_size),
              'context': 0,
              'context_type': ''
          }
        
        else:
          # randomly generate a position within the defined box space
          trial = {
              'stim_pos': np.array([self.rng.uniform(-self.box_size, self.box_size),
                                    self.rng.uniform(-self.box_size, self.box_size)]),
              'context': 0,
              'context_type': ''
          }
        trial.update(kwargs)
        stim_pos = trial['stim_pos']
        # context
        if not self.context == None:
          trial.update({'context': self.rng.choice([0,1]),
                        'context_type': self.context})
          context_state = trial['context']
        else:
          context_state = 0  


        # Setting time periods for this trial
        # Will add stimulus, delay and decision periods sequentially  
        # unsing self.timing info
        self.add_period(['baseline','stimulus', 'delay', 'movement'])

        # set obersvation space
        ## Add ground_truth_stim to stimulus period at stimulus location
        ##self.add_ob(ground_truth_stim, 'stimulus', where='x')
        # we could do adding but we just set a new position everytime
        
        # does the stimulus stay on during memory period?
        if self.mem:
            mem_stim_pos = [0, 0]
            mem_context = 0
        else:
            mem_stim_pos = stim_pos
            mem_context = context_state
                        
        # set obesrvations
        self.set_ob([1, 0, 0, 0], 'baseline')
        self.set_ob([1, stim_pos[0], stim_pos[1], context_state], 'stimulus')
        self.set_ob([1, mem_stim_pos[0], mem_stim_pos[1], mem_context], 'delay')
        self.set_ob([0, mem_stim_pos[0], mem_stim_pos[1], mem_context], 'movement')

        # set ground truth
        self.set_groundtruth([0,0], ['baseline','stimulus', 'delay'])
        
        # generate the straight reaches velocity profiles = gaussian
        # movement duration
        move_i_dur = self.end_ind['movement'] - self.start_ind['movement']
        
        #ANDBACK: set move duration dependent on context. For andback context
        # the movement is twice as fast and goes back and forth
        if (trial['context_type'] == 'andback') and context_state:
          move_i_dur = round(move_i_dur/2)
        
        # ANTI: set move position dependent on context
        if (trial['context_type'] == 'anti') and context_state:
          move_pos = -stim_pos
        else:
          move_pos = stim_pos
            
        
        #dist_to_move = stim_pos - self.pos # does not work, stim pos is always 0
        mu_gauss = move_i_dur/2.0
        sd_gauss = move_i_dur/8.0
        gauss_dist = gauss(np.arange(move_i_dur),mu=mu_gauss,sd=sd_gauss)
        move_speed = np.matmul(np.array(move_pos).reshape((len(move_pos),1)),
                               gauss_dist.reshape((1,len(gauss_dist)))/sum(gauss_dist))
        #move_speed = np.expand_dims(move_speed, 1)
        if (trial['context_type'] == 'andback') and context_state:
          self.set_groundtruth(np.append(move_speed.T,-move_speed.T,axis=0),'movement')
        else:
          self.set_groundtruth(move_speed.T, 'movement')

        return trial

    def _step(self, action):
        """
        Evaluate new step. 
  
        Parameters
        ----------
        action : TYPE
          DESCRIPTION.
  
        Returns
        -------
        TYPE
          current observation (predefined in _new_trial here).
        reward : TYPE
          current reward.
        bool
          task ends (never, since there is no end to this task).
        dict
          dict with additional info
          - here we store ground truth (position) and.
  
        """
        
        new_trial = False
        # rewards
        reward = 0 # default reward 0 unless something happens
        gt = self.gt_now  # current ground truth
        ob = self.ob_now # current observation
        # gt and action will be 2D but now are 1D
        
        # The trial dict established in '_new_trial' can be accessed by self.trial
        
        
        # if we really want to change start positon than we would need to do that
        # in a smart way, maybe we need to update gt here itself?

        # track velocity
        i_now = int(self.t/self.dt)
        self.vel[:,i_now] = action
        

        # track position
        if (self.t == 0) & self.reset_pos:
            self.pos = [0,0]
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
            # abort if movemet happens
            if action > 0.001:
                new_trial = self.abort
                reward = self.rewards['fail']

        # return 
        # - current observation (predefined in _new_trial here)
        # - current reward
        # - task ends (never, since there is no end to this task)
        # - dict with additional info
        # -- here we store ground truth (position) and 
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}

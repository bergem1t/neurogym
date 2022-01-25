#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ADAPTED FROM ReachingDelayResponse

import numpy as np

import neurogym as ngym
from neurogym import spaces

#%% helper functions
def gauss(x,sd=1,mu=0):
    '''
    Gaussian function, result = exp(-((x-mu)/sd)**2/2)

    Parameters
    ----------
    x : numpy array-like
        input space
    sd : float, optional
        Standard deviation. The default is 1.
    mu : float, optional
        Mean. The default is 0.

    Returns
    -------
    numpy array
        Gaussian.

    '''
    x = np.array(x)
    return np.exp(-((x-mu)/sd)**2/2 )
  
  
def sigmoid(x,a=1):
  """
  Sigmoid function, result = 1/(1+exp(-a*x))

  Parameters
  ----------
  x : numpy array-like
    input space
  a : float, optional
    slope. The default is 1.

  Returns
  -------
  numpy array
    sigmoid.

  """
  x = np.array(x)
  return 1/(1+np.exp(-a*x))


class movement:
  
  
  def __init__(self,move_params=None,dt=1,seed=None):
    """
    Class that can generate various movement sequences

    Parameters
    ----------
    move_params : dict, optional
      The parameter dictionary. Must have specific form. The default is None.
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
      seed to be parsed to np.random.default_rng(seed). The default is None.

    Returns
    -------
    None.

    """
    
    # the defined sequence
    self.param = {'type': [['g']],
            'p': [0.2],
            'scale': [[1]],
            'delay': [0],
            'mt': [1000],
            'linked': False}
    if move_params:
      self.param.update(move_params)
      
    # check which parameters are fix and which are random
    self.param_isvar = {'p'    : isinstance(self.param['p'][0]    ,(list, np.ndarray) ),
                        'scale': isinstance(self.param['scale'][0],(list, np.ndarray) ),
                        'delay': isinstance(self.param['delay'][0],(list, np.ndarray) ),
                        'mt'   : isinstance(self.param['mt'][0],(list, np.ndarray) )}
                        
    
    # output dimensions
    self.ndim = len(self.param['type'])
    # dt
    self.dt = dt
    # movement length = delay + mt
    if self.param_isvar['mt']:
      mt_max = max([max(i) for i in self.param['mt']])
    else:
      mt_max = max(self.param['mt'])
    if self.param_isvar['delay']:
      delay_max = max([max(i) for i in self.param['delay']])
    else:
      delay_max = max(self.param['delay'])
    self.nt_max = mt_max + delay_max
    
   

    # max box size
    max_scale = self.param['scale']
    if self.param_isvar['scale']:
      max_scale = [max(abs(np.array(i))) for i in max_scale]
    max_size = np.zeros_like(max_scale)
    for d in range(self.ndim):
      s = (np.array(self.param['type'][d]) == 's').astype(float)
      sm = (np.array(self.param['type'][d]) == '-s').astype(float)
      c = np.array(self.param['type'][d]) == 'c'
      g = np.array(self.param['type'][d]) == 'g'
      
      s_traj = abs(np.cumsum(s-sm)) # this shows elevation due to sigmoid
      cur_size = max(s_traj*max_scale[d]) # max size due to sigmoids
      if sum(g*s_traj)+sum(c*s_traj):
        cur_size += max_scale[d] # +1 if sine or gauss on top
      
      max_size[d] = max(max_scale[d],cur_size) # max is necessary if no sigmoid is present
    self.max_size = max_size
    
    self.seed(seed)
    
    # store current sequence
    self.sequence = None
    self.trial = None
    
    
  def seed(self, seed=None):
    """Set random seed."""
    #self.rng = np.random.RandomState(seed) # old but used by ngym
    self.rng = np.random.default_rng(seed) # new np recommended function
    return [seed]
    
  def generate(self):
    """
    Generate new movement sequence.

    Raises
    ------
    ValueError
      DESCRIPTION.

    Returns
    -------
    out : TYPE
      DESCRIPTION.

    """
    
    var_param_list = [k for k,i in self.param_isvar.items() if i]
    trial = self.param.copy()
    
    if len(var_param_list) == 0:
      # no variable parameter
      return self.make_sequence(trial)
    
    sel_i = None
    if self.param['linked']:
      # the variable params in the dict are paired acoording to the current order
      # It is assumed that all varaoble parameters have the same lengths 
      #TODO: test this and raise an error if this is not correct
      
      # number of vars
      n_vars = len(self.param[var_param_list[0]][0])
      # select the index vor all variabled
      sel_i = int(self.rng.integers(n_vars))
    
    # generate the trial dict by selecting random parameter
    for p in var_param_list:#['p','scale','delay','mt']:
      #if self.param_isvar[p]:
      trial[p] = []
      for i in range(self.ndim):
        if sel_i != None:
          x = self.param[p][i][sel_i] #linked
        else:
          x = self.rng.choice( self.param[p][i] ) # not linked
        if p in ['delay','mt']:
          trial[p].append(round(x))
        else:
          trial[p].append(x)
    
    # make_sequence() will generate the sequence out of the trial
    return self.make_sequence(trial)
  
  def make_sequence(self,trial):
    
    # overwrite current trial
    self.trial = trial
    
    out = np.zeros([self.ndim,round(self.nt_max/self.dt)])
    
    for i in range(self.ndim):
      # pick parameter from the param dict
      ele_list = trial['type'][i]
      mt = round(trial['mt'][i]/self.dt)
      delay = round(trial['delay'][i]/self.dt)
      p = trial['p'][i]
      scale = trial['scale'][i]
      
      # setup sequence element matrix
      ne = len(ele_list)
      seq_len = int(np.floor(mt/ne))
      seq_ele = np.zeros([ne,seq_len])
      x = np.linspace(-1, 1, seq_len)
      
      # generate each element
      for j,e in enumerate(ele_list):
        #start where we left in the last sequence
        if j > 0:
          seq_ele[j,:] = np.ones(seq_len)*seq_ele[j-1,-1]
      
        if 'g' in e:
          add_seq = gauss(x,p)
        elif 's' in e:
          add_seq = sigmoid(x,p)
        elif 'l' in e:
          add_seq = np.zeros(seq_len)
        elif 'c' in e:
          add_seq = np.sin(np.pi*x*p)
        else:
          raise ValueError('types must be "g" "s" "c" and/or "l"')
          
        # start at 0
        add_seq -= add_seq[0]
        
        # scale to abs max = 1
        add_seq /= max(abs(add_seq))
        
        # invert if requested
        if '-' in e:
          add_seq *= -1
        
        # scale
        add_seq *=  scale
        
        seq_ele[j,:] += add_seq
      
      # flatten the element matrix to the output array
      out[i,delay:(delay+seq_ele.size)] = seq_ele.flatten()
      out[i,delay+seq_ele.size:] = out[i,seq_ele.size-1]  
    
    # store the sequence
    self.sequence = out
    
    return out

    
    
    


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
class Movementsequence(ngym.TrialEnv):
    r"""Movement task with a delay period.

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

    def __init__(self, dt=100, rewards=None, timing=None, move_param=None,
                 mem=True,context=None,reset_pos=True,):
        super().__init__(dt=dt)

        # whether or not the stim stays on
        self.mem = mem
        # context cue with stim cue. The input context defines what kind of context in a str
        if isinstance(context,str):
          context = context.lower()
        assert context in [None,'anti'], 'context must be a valid string defining the context'
        self.context = context
          
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
            'delay': (0, 500, 1000)}
        if timing:
            self.timing.update(timing)
            
        # this class defines the output space
        self.movement = movement(move_param,dt=dt)
        self.timing['movement'] = self.movement.nt_max
          
        # output dimensions  
        n_action_dim = self.movement.ndim
        
        # stimulus dimensions (without fix and context)
        n_stim_dim = sum([i for _,i in self.movement.param_isvar.items()])*n_action_dim
        
        # we normalize the stimuli that it is between [-box_size, box-size]
        self.stim_vars = [k for k,i in self.movement.param_isvar.items() if i]
        self.stim_absmax = [np.abs(self.movement.param[k]).max() for k in self.stim_vars]
        
        # tracking the current position
        # Position is not implemented and probaly not necessary if we consider
        # hand-centered coordinates
        self.x_cumsum = [0 for i in range(n_action_dim)]
        self.x = np.zeros([n_action_dim,int(self.tmax/self.dt)])
        self.reset_pos=reset_pos
        
        # calculate the bounding box
        self.box_size = np.float32(max(self.movement.max_size))


        #self.r_tmax = self.rewards['fail'] #necessary??
        # self.r_tmax would be added to reward whenever trial has finished.
        self.abort = False

        # define spaces
        name = {'fixation': 0}
        i_stim_dim = 1
        stim_low = [0.0]
        stim_high = [1.0]
        if context:
          name['context'] = i_stim_dim
          i_stim_dim += 1  
          stim_low += [0.0]
          stim_high += [1.0]
        for k in self.stim_vars:
          name[k] = list(np.arange(n_action_dim) + i_stim_dim)
          i_stim_dim += n_action_dim
          #stim_low += [min(p) for p in self.movement.param[k]]
          #stim_high += [max(p) for p in self.movement.param[k]]
        stim_low += [-1.0]*n_stim_dim
        stim_high += [1.0]*n_stim_dim
        self.obs_name = name
        self.observation_space = spaces.Box(low=np.array(stim_low,dtype=np.float32),
                                            high=np.array(stim_high,dtype=np.float32),
                                            name=name)#,
                                            #dtype=np.float32, name=name)
        
        # action space has one dimension less bc it does not produce fix
        # action will be speed
        self.action_space = spaces.Box(low=-self.box_size,
                                       high=self.box_size,
                                       shape=(n_action_dim,),name={'target':[0,1]})#,
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
        
        # the movement
        self.movement.generate()
        trial = self.movement.trial.copy()

        # context
        # trial.update({
        #     'stim_pos': np.array([self.rng.uniform(-self.box_size, self.box_size),
        #                           self.rng.uniform(-self.box_size, self.box_size)]),
        #     'context': 0,
        #     'context_type': ''
        # })
        
        
        # stim_pos = trial['stim_pos']
        # context
        trial.update({'context': 0,
                      'context_type': ''})
        if not self.context == None:
          trial.update({'context': self.rng.choice([0,1]),
                        'context_type': self.context})

        trial.update(kwargs) 
        
        # the movement
        move_seq = self.movement.make_sequence(trial)
        context_state = trial['context']
        stim_state = []
        for i,k in enumerate(self.stim_vars):
          cur_state = np.array(trial[k])/self.stim_absmax[i] if self.stim_absmax[i]>0 else [0]*self.movement.ndim
          stim_state.extend(list(cur_state))

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
            mem_stim_state = [0 for i in stim_state]
            mem_context = 0
        else:
            mem_stim_state = stim_state
            mem_context = context_state
        
        fix_state = [[1],[1],[1],[0]]
        if self.context:
          fix_state = [[1,0],[1,context_state],[1,mem_context],[0,mem_context]]
                        
        # set obesrvations
        self.set_ob(fix_state[0] + [0 for i in stim_state], 'baseline')
        self.set_ob(fix_state[1] + stim_state, 'stimulus')
        self.set_ob(fix_state[2] + mem_stim_state, 'delay')
        self.set_ob(fix_state[3] + mem_stim_state, 'movement')

        # set ground truth
        self.set_groundtruth([0 for i in move_seq], ['baseline','stimulus', 'delay'])
        
        # add the sequence        
        # ANTI: invert movement
        if (trial['context_type'] == 'anti') and context_state:
          move_seq = -move_seq            

        self.set_groundtruth(move_seq.T, 'movement')

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
        
        # The trial dict established in '_new_trial' can be accessed by self.trial
        
        # if we really want to change start positon than we would need to do that
        # in a smart way, maybe we need to update gt here itself?

        # track velocity
        i_now = int(self.t/self.dt)
        self.x[:,i_now] = action
        

        # track position
        if (self.t == 0) & self.reset_pos:
            self.x_cumsum = [0 for i in range(self.movement.ndim)]
        else:
            self.x_cumsum += action
            
        # reward per step - This is only relevant for supervised learning??
        _, acc = accuracy_function(gt, action)
        reward = self.rewards['ongoing']/(1+acc)
                                          
        # this should be checked at the end of the trial
        if self.t + self.dt > self.tmax:
            new_trial = True 
            # here we could do endpoint reward
            acc_end_point, acc = accuracy_function(self.gt, np.array(self.x))
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

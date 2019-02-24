# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:00:32 2019

@author: MOLANO

A parametric working memory task, based on

  Neuronal population coding of parametric working memory.
  O. Barak, M. Tsodyks, & R. Romo, JNS 2010.

  http://dx.doi.org/10.1523/JNEUROSCI.1875-10.2010

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

import ngym
from gym import spaces
from gym.utils import seeding


class romo(ngym.ngym):

    # Inputs
    inputs = tasktools.to_map('FIXATION', 'F-POS', 'F-NEG')

    # Actions
    actions = tasktools.to_map('FIXATE', '>', '<')

    # Trial conditions
    gt_lts = ['>', '<']
    fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]
    n_conditions = len(gt_lts) * len(fpairs)

    # Training
    n_gradient = n_conditions
    n_validation = 20*n_conditions

    # Slow down the learning
    lr = 0.002
    baseline_lr = 0.002

    # Input noise
    sigma = np.sqrt(2*100*0.001)

    # Epoch durations
    fixation = 750
    f1 = 500
    delay_min = 3000 - 300
    delay_max = 3000 + 300
    f2 = 500
    decision = 500
    tmax = fixation + f1 + delay_max + f2 + decision

    # Rewards
    R_ABORTED = -1
    R_CORRECT = +1

    # Input scaling
    fall = np.ravel(fpairs)
    fmin = np.min(fall)
    fmax = np.max(fall)

    def __init__(self, dt=0.1):
        # call ngm __init__ function
        super().__init__(dt=dt)
        high = np.array([1])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None

        self.rng = np.random.RandomState(seed=0)  # TODO: move to superclass?
        self.trial = self.get_condition(self.rng, self.dt)

    def get_condition(self, rng, dt, context={}):
        # -------------------------------------------------------------------------
        # Epochs
        # --------------------------------------------------------------------------

        delay = context.get('delay')
        if delay is None:
            delay = tasktools.uniform(rng, dt, self.delay_min, self.delay_max)

        durations = {
            'fixation':   (0, self.fixation),
            'f1':         (self.fixation, self.fixation + self.f1),
            'delay':      (self.fixation + self.f1,
                           self.fixation + self.f1 + delay),
            'f2':         (self.fixation + self.f1 + delay,
                           self.fixation + self.f1 + delay + self.f2),
            'decision':   (self.fixation + self.f1 + delay + self.f2,
                           self.tmax),
            'tmax':       self.tmax
            }
        time, epochs = tasktools.get_epochs_idx(dt, durations)

        gt_lt = context.get('gt_lt')
        if gt_lt is None:
            gt_lt = tasktools.choice(rng, self.gt_lts)

        fpair = context.get('fpair')
        if fpair is None:
            fpair = tasktools.choice(rng, self.fpairs)

        return {
            'durations': durations,
            'time':      time,
            'epochs':    epochs,
            'gt_lt':     gt_lt,
            'fpair':     fpair
            }

    def seed(self, seed=None):  # TODO: move to superclass?
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def scale(self, f):
        return (f - self.fmin)/(self.fmax - self.fmin)

    def scale_p(self, f):
        return (1 + self.scale(f))/2

    def scale_n(self, f):
        return (1 - self.scale(f))/2

    def step(self, action):
        trial = self.trial

        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        epochs = trial['epochs']
        status = {'continue': True}
        reward = 0
        if self.t-1 not in epochs['decision']:
            if action != self.actions['FIXATE']:
                status['continue'] = False
                status['choice'] = None
                reward = self.R_ABORTED
        elif self.t-1 in epochs['decision']:
            if action == self.actions['>']:
                status['continue'] = False
                status['choice'] = '>'
                status['correct'] = (trial['gt_lt'] == '>')
                if status['correct']:
                    reward = self.R_CORRECT
            elif action == self.actions['<']:
                status['continue'] = False
                status['choice'] = '<'
                status['correct'] = (trial['gt_lt'] == '<')
                if status['correct']:
                    reward = self.R_CORRECT

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        if trial['gt_lt'] == '>':
            f1, f2 = trial['fpair']
        else:
            f2, f1 = trial['fpair']

        u = np.zeros(len(self.inputs))
        if self.t not in epochs['decision']:
            u[self.inputs['FIXATION']] = 1
        if self.t in epochs['f1']:
            u[self.inputs['F-POS']] = self.scale_p(f1) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            u[self.inputs['F-NEG']] = self.scale_n(f1) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
        if self.t in epochs['f2']:
            u[self.inputs['F-POS']] = self.scale_p(f2) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            u[self.inputs['F-NEG']] = self.scale_n(f2) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)

        # -------------------------------------------------------------------------

        return u, reward, status

    def reset(self):
        self.trial = self.get_condition(self.rng, self.dt)
        self.t = 0

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.97

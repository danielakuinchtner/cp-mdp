# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox: ``mdp`` module
=====================================================

The ``mdp`` module provides classes for the resolution of discrete-time Markov
Decision Processes.

Available classes
-----------------
:class:`~mdptoolbox.mdp.MDP`
    Base Markov decision process class
:class:`~mdptoolbox.mdp.ValueIteration`
    Value iteration MDP
:class:`~mdptoolbox.mdp.ValueIterationGS`
    Gauss-Seidel value iteration MDP

"""

# Copyright (c) 2011-2015 Steven A. W. Cordwell
# Copyright (c) 2009 INRA
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the <ORGANIZATION> nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math as _math
import time as _time
import numpy as _np
import scipy.sparse as _sp
import mdptoolbox.util as _util

_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations " \
    "condition."
_MSG_STOP_EPSILON_OPTIMAL_POLICY = "Iterating stopped, epsilon-optimal " \
    "policy found."
_MSG_STOP_EPSILON_OPTIMAL_VALUE = "Iterating stopped, epsilon-optimal value " \
    "function found."
_MSG_STOP_UNCHANGING_POLICY = "Iterating stopped, unchanging policy found."


def _computeDimensions(transition):
    A = len(transition)
    try:
        if transition.ndim == 3:
            S = transition.shape[1]
        else:
            S = transition[0].shape[0]
    except AttributeError:
        S = transition[0].shape[0]
    return S, A


def _printVerbosity(iteration, variation):
    if isinstance(variation, float):
        print("{:>10}{:>12f}".format(iteration, variation))
    elif isinstance(variation, int):
        print("{:>10}{:>12d}".format(iteration, variation))
    else:
        print("{:>10}{:>12}".format(iteration, variation))


class MDP(object):

    def __init__(self, transitions, reward, discount, epsilon, max_iter,
                 skip_check=False):
        # Initialise a MDP based on the input parameters.

        # if the discount is None then the algorithm is assumed to not use it
        # in its computations
        if discount is not None:
            self.discount = float(discount)
            assert 0.0 < self.discount <= 1.0, (
                "Discount rate must be in ]0; 1]"
            )
            #if self.discount == 1:
                #print("WARNING: check conditions of convergence. With no "
                      #"discount, convergence can not be assumed.")

        # if the max_iter is None then the algorithm is assumed to not use it
        # in its computations
        if max_iter is not None:
            self.max_iter = int(max_iter)
            assert self.max_iter > 0, (
                "The maximum number of iterations must be greater than 0."
            )

        # check that epsilon is something sane
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, "Epsilon must be greater than 0."

        if not skip_check:
            # We run a check on P and R to make sure they are describing an
            # MDP. If an exception isn't raised then they are assumed to be
            # correct.
            _util.check(transitions, reward)

        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)
        self.R = self._computeReward(reward, transitions)
        #print(reward)

        # the verbosity is by default turned off
        self.verbose = False
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector ie shape of (S,) or (1, S)
        self.V = None
        # policy can also be stored as a vector
        self.policy = None

    def __repr__(self):
        P_repr = "P: \n"
        R_repr = "R: \n"
        for aa in range(self.A):
            P_repr += repr(self.P[aa]) + "\n"
            R_repr += repr(self.R[aa]) + "\n"
        return(P_repr + "\n" + R_repr)

    def _bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (1, self.S)), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = _np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.R[aa] + self.discount * self.P[aa].dot(V)
        #print("QAA", Q[aa])

        #print(V.shape)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0), Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)

    def _computeTransition(self, transition):
        return tuple(transition[a] for a in range(self.A))

    def _computeReward(self, reward, transition):
        # Compute the reward for the system in one state chosing an action.
        # Arguments
        # Let S = number of states, A = number of actions
        # P could be an array with 3 dimensions or  a cell array (1xA),
        # each cell containing a matrix (SxS) possibly sparse
        # R could be an array with 3 dimensions (SxSxA) or  a cell array
        # (1xA), each cell containing a sparse matrix (SxS) or a 2D
        # array(SxA) possibly sparse
        try:
            if reward.ndim == 1:
                return self._computeVectorReward(reward)
            elif reward.ndim == 2:
                return self._computeArrayReward(reward)
            else:
                r = tuple(map(self._computeMatrixReward, reward, transition))
                return r
        except (AttributeError, ValueError):
            if len(reward) == self.A:
                r = tuple(map(self._computeMatrixReward, reward, transition))
                return r
            else:
                return self._computeVectorReward(reward)

    def _computeVectorReward(self, reward):
        if _sp.issparse(reward):
            raise NotImplementedError
        else:
            r = _np.array(reward).reshape(self.S)
            return tuple(r for a in range(self.A))

    def _computeArrayReward(self, reward):
        if _sp.issparse(reward):
            raise NotImplementedError
        else:
            def func(x):
                return _np.array(x).reshape(self.S)

            return tuple(func(reward[:, a]) for a in range(self.A))

    def _computeMatrixReward(self, reward, transition):
        if _sp.issparse(reward):
            # An approach like this might be more memory efficeint
            # reward.data = reward.data * transition[reward.nonzero()]
            # return reward.sum(1).A.reshape(self.S)
            # but doesn't work as it is.
            return reward.multiply(transition).sum(1).A.reshape(self.S)
        elif _sp.issparse(transition):
            return transition.multiply(reward).sum(1).A.reshape(self.S)
        else:
            return _np.multiply(transition, reward).sum(1).reshape(self.S)

    def _startRun(self):
        if self.verbose:
            _printVerbosity('Iteration', 'Variation')

        self.time = _time.time()

    def _endRun(self):
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())

        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)

        self.time = _time.time() - self.time

    def run(self):
        """Raises error because child classes should implement this function.
        """
        raise NotImplementedError("You should create a run() method.")

    def setSilent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False

    def setVerbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True



class ValueIteration(MDP):

    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=1000, initial_value=0, skip_check=False):
        # Initialise a value iteration MDP.

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter,
                     skip_check=skip_check)
        self.iterations_list = []
        self.v_list = []
        # initialization of optional arguments
        if initial_value == 0:
            self.V = _np.zeros(self.S)
        else:
            assert len(initial_value) == self.S, "The initial value must be " \
                "a vector of length S."
            self.V = _np.array(initial_value).reshape(self.S)
        if self.discount < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
            self._boundIter(epsilon)
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:  # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon

    def _boundIter(self, epsilon):
        # Compute a bound for the number of iterations.
        #
        # for the value iteration
        # algorithm to find an epsilon-optimal policy with use of span for the
        # stopping criterion
        #
        # Arguments -----------------------------------------------------------
        # Let S = number of states, A = number of actions
        #    epsilon   = |V - V*| < epsilon,  upper than 0,
        #        optional (default : 0.01)
        # Evaluation ----------------------------------------------------------
        #    max_iter  = bound of the number of iterations for the value
        #    iteration algorithm to find an epsilon-optimal policy with use of
        #    span for the stopping criterion
        #    cpu_time  = used CPU time
        #
        # See Markov Decision Processes, M. L. Puterman,
        # Wiley-Interscience Publication, 1994
        # p 202, Theorem 6.6.6
        # k =    max     [1 - S min[ P(j|s,a), p(j|s',a')] ]
        #     s,a,s',a'       j
        k = 0
        h = _np.zeros(self.S)

        for ss in range(self.S):
            PP = _np.zeros((self.A, self.S))
            for aa in range(self.A):
                try:
                    PP[aa] = self.P[aa][:, ss]
                    # print("PP[aa]", PP[aa])
                except ValueError:
                    PP[aa] = self.P[aa][:, ss].todense().A1
            # minimum of the entire array.
            h[ss] = PP.min()

        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        # p 201, Proposition 6.6.5
        span = _util.getSpan(value - Vprev)
        max_iter = (_math.log((epsilon * (1 - self.discount) / self.discount) /
                    span) / _math.log(self.discount * k))
        # self.V = Vprev

        self.max_iter = int(_math.ceil(max_iter))

    def run(self):
        # Run the value iteration algorithm.
        self._startRun()

        while True:
            self.iter += 1

            Vprev = self.V.copy()

            # Bellman Operator: compute policy and value functions
            self.policy, self.V = self._bellmanOperator()

            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = _util.getSpan(self.V - Vprev)
            self.iterations_list.append(variation)
            self.v_list.append(self.V.copy())
            if self.verbose:
                _printVerbosity(self.iter, variation)

            if variation < self.thresh:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        self._endRun()


class ValueIterationGS(ValueIteration):


    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10, initial_value=0, skip_check=False):
        # Initialise a value iteration Gauss-Seidel MDP.

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter,
                     skip_check=skip_check)
        self.iterations_list = []
        self.v_list = []
        # initialization of optional arguments
        if initial_value == 0:
            self.V = _np.zeros(self.S)
        else:
            if len(initial_value) != self.S:
                raise ValueError("The initial value must be a vector of "
                                 "length S.")
            else:
                try:
                    self.V = initial_value.reshape(self.S)
                except AttributeError:
                    self.V = _np.array(initial_value)
                except:
                    raise
        if self.discount < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
            self._boundIter(epsilon)
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:  # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon

    def run(self):
        # Run the value iteration Gauss-Seidel algorithm.

        self._startRun()
        self.v_list.append(self.V.copy())
        while True:
            self.iter += 1

            Vprev = self.V.copy()

            for s in range(self.S):

                Q = [float(self.R[a][s] +
                           self.discount * self.P[a][s, :].dot(self.V))
                     for a in range(self.A)]

                self.V[s] = max(Q)

            variation = _util.getSpan(self.V - Vprev)
            self.iterations_list.append(variation)
            self.v_list.append(self.V.copy())
            if self.verbose:
                _printVerbosity(self.iter, variation)

            if variation < self.thresh:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        self.policy = []
        for s in range(self.S):
            Q = _np.zeros(self.A)
            for a in range(self.A):

                Q[a] = (self.R[a][s] +
                        self.discount * self.P[a][s, :].dot(self.V))


            self.V[s] = Q.max()

            print(self.V[s])

            self.policy.append(int(Q.argmax()))

        self._endRun()

"""Utility functions for training tutorial."""

import pprint
from copy import deepcopy
from functools import reduce
from itertools import accumulate

import homomorph
import numpy as np
from numpy import exp, log


class ArrayRV:
    """Class that allows HMM class to interface with a pre-computed probability array.

    The rvs method is only defined so the HMM recognizes it as a proper
    random variable. Since random variates are not needed in this script, the
    body of the method is left blank.
    """
    def __init__(self, array):
        self.array = array

    def pmf(self, x):
        return self.array[x]

    def rvs(self, random_state=None):
        pass


class HMM(homomorph.HMM):
    """A class that modifies the default HMM to accept pre-calculated intermediates.

    Discriminative training requires decoding of both states and transitions.
    Both of these calculations rely on the forward and backward variables.
    Since these are the most computationally intensive steps in decoding and
    both are used in state and transition decoding, this modified class accepts
    the forward and backward variables as arguments rather than calculating
    them from scratch each time.
    """
    def forward_backward1(self, emits, fs, ss_f, bs, ss_b):
        """Posterior decoding of states."""
        p = reduce(lambda x, y: x+y, map(log, ss_f))
        ss_f = list(accumulate(map(log, ss_f)))
        ss_b = list(accumulate(map(log, ss_b[::-1])))[::-1]

        fbs = {state: [] for state in self.states}
        for i in range(len(emits)):
            for state, fb in fbs.items():
                fbs[state].append(fs[state][i]*bs[state][i]*exp(ss_f[i]+ss_b[i]-p))

        return fbs

    def forward_backward2(self, emits, fs, ss_f, bs, ss_b):
        """Posterior decoding of transitions."""
        p = reduce(lambda x, y: x+y, map(log, ss_f))
        ss_f = list(accumulate(map(log, ss_f)))
        ss_b = list(accumulate(map(log, ss_b[::-1])))[::-1]

        pairs = [(state0, state1) for state0, t_dist in self.t_dists.items() for state1 in t_dist]
        fbs = {pair: [] for pair in pairs}
        for i in range(len(emits)-1):
            for (state0, state1), fb in fbs.items():
                fb.append(fs[state0][i]*self.t_dists[state0][state1]*self.e_dists[state1].pmf(emits[i+1])*bs[state1][i+1]*exp(ss_f[i]+ss_b[i+1]-p))

        return {pair: sum(fb) for pair, fb in fbs.items()}

    def joint_likelihood(self, emits, states):
        """Log-likelihood of emission and state sequence."""
        states = [self._state2idx[state] for state in states]
        p, state0 = log(self._e_dists_rv[states[0]].pmf(emits[0]) * self._start_dist_rv.pmf(states[0])), states[0]
        for emit, state1 in zip(emits[1:], states[1:]):
            p += log(self._e_dists_rv[state1].pmf(emit) * self._t_dists_rv[state0].pmf(state1))
            state0 = state1
        return p


def fit_CML(data, t_dists, e_params, e_funcs, e_primes, e_param2aux, e_aux2param, start_dist,
            maxiter=100, epsilon=1E-4, eta=1, verbose=True):
    """Fit data to HMM using conditional maximum likelihood training."""
    # Copy parameters and convert to auxiliary variables
    t_dists = deepcopy(t_dists)
    e_params = deepcopy(e_params)
    start_dist = start_dist.copy()
    t_dists_aux, e_params_aux, start_dist_aux = param2aux(t_dists, e_params, e_param2aux, start_dist)

    # Make record object from each state-emission pair
    state_set = set(t_dists)
    t_sets = {state1: {state2 for state2 in t_dist} for state1, t_dist in t_dists.items()}
    records = []
    for example in data:
        xs, ys = np.array(example[0]), np.array(example[1])
        mis = state_indicators(xs, state_set)
        mijs = transition_counts(xs, t_sets)
        records.append({'state_seq': xs, 'emit_seq': ys, 'mis': mis, 'mijs': mijs})

    ll0 = None
    for numiter in range(maxiter):
        # Calculate gradients
        gradients = []
        for record in records:
            # Unpack record fields
            state_seq, emit_seq = record['state_seq'], record['emit_seq']
            mis, mijs = record['mis'], record['mijs']

            # Pre-calculate probabilities for each state as array
            e_dists_rv = {}
            for state, e_param in e_params.items():
                e_func = e_funcs[state]
                e_dists_rv[state] = ArrayRV(e_func(emit_seq, **e_param))

            # Instantiate model and get expectations
            model = HMM(t_dists, e_dists_rv, start_dist)
            idx_seq = list(range(len(state_seq)))  # Everything is pre-calculated, so emit_seq is the emit index
            fs, ss_f = model.forward(idx_seq)
            bs, ss_b = model.backward(idx_seq)
            nis = model.forward_backward1(idx_seq, fs, ss_f, bs, ss_b)
            nijs = model.forward_backward2(idx_seq, fs, ss_f, bs, ss_b)

            # Calculate likelihood
            px = reduce(lambda x, y: x + y, map(log, ss_f))
            pxy = model.joint_likelihood(idx_seq, state_seq)
            ll = pxy - px

            # Get t_dists_aux gradients
            t_grads_aux = {}
            for state1, t_dist in t_dists.items():
                mn = sum([mijs[(state1, state2)] - nijs[(state1, state2)] for state2 in t_dist])
                t_grad_aux = {}
                for state2, p in t_dist.items():
                    t_grad_aux[state2] = -(mijs[(state1, state2)] - nijs[(state1, state2)] - p * mn)
                t_grads_aux[state1] = t_grad_aux

            # Get e_dists_aux gradients
            e_grads_aux = {}
            for state, e_param in e_params.items():
                mn = np.array(mis[state]) - np.array(nis[state])
                e_grad_aux = {}
                for param in e_param:
                    e_prime = e_primes[state][param]
                    e_param_aux = e_params_aux[state]
                    e_grad_aux[f'{param}_aux'] = -mn * e_prime(emit_seq, **e_param, **e_param_aux)
                e_grads_aux[state] = e_grad_aux

            # Get start_dist gradients
            mn = sum([mis[state][0] - nis[state][0] for state in start_dist])
            start_grad_aux = {}
            for state, p in start_dist.items():
                start_grad_aux[state] = -(mis[state][0] - nis[state][0] - p * mn)

            gradients.append({'ll': ll, 't_grads_aux': t_grads_aux, 'e_grads_aux': e_grads_aux, 'start_grad_aux': start_grad_aux})

        # Format parameters for display
        t_string = pprint.pformat(t_dists).replace('\n', '\n' + len('t_dists: ') * ' ')
        e_string = pprint.pformat(e_params).replace('\n', '\n' + len('e_dists: ') * ' ')
        start_string = pprint.pformat(start_dist)

        # Check stop condition
        # Don't want to repeat calculations, so ith iterate checks previous update
        # For example, 0th iterate shows initial parameters and 1st iterate shows results of first update
        ll = sum([gradient['ll'] for gradient in gradients])
        if ll0 is not None and abs(ll - ll0) < epsilon:
            if verbose:
                print(f'FINAL VALUES')
                print('log-likelihood:', ll)
                print('Δlog-likelihood:', ll - ll0 if ll0 is not None else None)
                print('t_dists:', t_string)
                print('e_params:', e_string)
                print('start_dist:', start_string)
            break

        # Print results
        if verbose:
            print(f'ITERATION {numiter}')
            print('log-likelihood:', ll)
            print('Δlog-likelihood:', ll - ll0 if ll0 is not None else None)
            print('t_dists:', t_string)
            print('e_params:', e_string)
            print('start_dist:', start_string)
            print()

        # Accumulate and apply gradients
        for state1, t_dist_aux in t_dists_aux.items():
            for state2 in t_dist_aux:
                grad_stack = np.hstack([gradient['t_grads_aux'][state1][state2] for gradient in gradients])
                dz = eta * grad_stack.sum() / len(grad_stack)
                t_dist_aux[state2] -= dz

        for state, e_param_aux in e_params_aux.items():
            for aux in e_param_aux:
                grad_stack = np.hstack([gradient['e_grads_aux'][state][aux] for gradient in gradients])
                dz = eta * grad_stack.sum() / len(grad_stack)
                e_param_aux[aux] -= dz

        for state in start_dist_aux:
            grad_stack = np.hstack([gradient['start_grad_aux'][state] for gradient in gradients])
            dz = eta * grad_stack.sum() / len(grad_stack)
            start_dist_aux[state] -= dz

        # Convert auxiliary variables to primary
        ll0 = ll
        t_dists, e_params, start_dist = aux2param(t_dists_aux, e_params_aux, e_aux2param, start_dist_aux)

    return t_dists, e_params, start_dist


def transition_counts(state_seq, t_sets):
    """Return counts of transitions between states."""
    mijs = {}
    for state1, t_set in t_sets.items():
        s1 = state_seq[:-1] == state1
        for state2 in t_set:
            s2 = state_seq[1:] == state2
            mijs[(state1, state2)] = (s1 & s2).sum()
    return mijs


def state_indicators(state_seq, state_set):
    """Return indicators of states."""
    mis = {}
    for state in state_set:
        mis[state] = state_seq == state
    return mis


def param2aux(t_dists, e_params, e_param2aux, start_dist):
    """Transform primary parameters to their corresponding auxiliary variables."""
    t_dists_aux = {}
    for state1, t_dist in t_dists.items():
        t_dists_aux[state1] = {state2: log(p) for state2, p in t_dist.items()}

    e_params_aux = {}
    for state, e_param in e_params.items():
        e_param_aux = {}
        for param in e_param:
            aux = f'{param}_aux'
            f = e_param2aux[state][param]
            z = f(**e_param)
            e_param_aux[aux] = z
        e_params_aux[state] = e_param_aux

    start_dist_aux = {state: log(p) for state, p in start_dist.items()}

    return t_dists_aux, e_params_aux, start_dist_aux


def aux2param(t_dists_aux, e_params_aux, e_aux2param, start_dist_aux):
    """Transform auxiliary variables to their corresponding primary parameters."""
    t_dists = {}
    for state1, t_dist_aux in t_dists_aux.items():
        z_sum = sum([exp(z) for z in t_dist_aux.values()])
        t_dists[state1] = {state2: exp(z)/z_sum for state2, z in t_dist_aux.items()}

    e_params = {}
    for state, e_param_aux in e_params_aux.items():
        e_param = {}
        for aux in e_param_aux:
            param = aux.removesuffix('_aux')
            f = e_aux2param[state][param]
            p = f(**e_param_aux)
            e_param[param] = p
        e_params[state] = e_param

    z_sum = sum([exp(z) for z in start_dist_aux.values()])
    start_dist = {state: exp(z) / z_sum for state, z in start_dist_aux.items()}

    return t_dists, e_params, start_dist

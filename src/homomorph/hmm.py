"""Classes and functions which implement HMM algorithms."""

from functools import reduce
from itertools import chain, accumulate

import scipy.stats as stats
from numpy import exp, log


class HMM:
    """A class for storing HMM model parameters and computing inferences from sequences of emissions.

    In this implementation, the hidden states are simply called states and the
    outputs associated with each state are called emissions (or emits for
    short). The recommended method for instantiating a particular model is by
    passing all its parameters in the form of dictionaries, e.g. for the
    transition matrix a nested dictionary where the first and second levels
    correspond to rows and columns, respectively. Under the hood, however,
    scipy rv_discrete objects are used as a convenient API for storing the pmf
    and generating random variates. This allows for more flexible emission
    distributions (such as those over a countably infinite support) to be
    passed directly as rv_discrete objects. Currently only discrete random
    variables are supported for emission distributions, largely since in the
    scipy implementation of continuous random variables the pdf is accessed via
    the pdf method (as opposed to the pmf method). Additionally, the
    transition matrix can only be passed as a nested dictionary to ensure a
    clear labeling of all states in the model.

    Parameters
    ----------
    t_dists: dict of dicts
        The outer dict selects a row in the transition matrix and the inner
        dict selects a column in that row. Each possible state must have an
        entry in the outer dict. The probability masses of each distributions
        given as an inner dict must sum to 1.
    e_dists: dict of dicts or dict of discrete_rvs
        Outer dict is keyed by state. If dict of dicts, inner dict is a mapping
        of emission labels to probabilities. Combinations of inner dicts and
        discrete_rvs are disallowed.
    start_dist: dict
        Dictionary mapping state labels to their probability of occurring as an
        initial state.
    stop_states: list of hashable objects, optional
        Labels of dictionary designated as stop state. Simulations will
        terminate early if a stop state is encountered.

    Methods
    -------
    simulate
    forward
    backward
    forward_backward
    """
    def __init__(self, t_dists, e_dists, start_dist, stop_states=None, name='hmm'):
        # Check t_dists
        states = set(t_dists)
        for state, t_dist in t_dists.items():
            if not set(t_dist) <= states:
                raise ValueError(f'{state} t_dist contains a transition to an unknown state.')

        # Check e_dists
        if set(e_dists) != set(states):
            raise ValueError('States in e_dists do not match those in t_dists.')
        all_dicts = all([isinstance(e_dist, dict) for e_dist in e_dists.values()])
        all_pmfs = all([getattr(e_dist, 'pmf', None) and getattr(e_dist, 'rvs', None) for e_dist in e_dists.values()])
        all_pdfs = all([getattr(e_dist, 'pdf', None) and getattr(e_dist, 'rvs', None) for e_dist in e_dists.values()])
        if not (all_dicts or all_pmfs or all_pdfs):
            raise ValueError('e_dists are not either all dicts or all rvs of the same type.')

        # Check start_dist
        if not set(start_dist) <= states:
            raise ValueError('start_dist contains a transition to an unknown state.')

        # Create random variates from t_dists
        state2idx = {}
        idx2state = {}
        for idx, state in enumerate(sorted(states)):
            state2idx[state] = idx
            idx2state[idx] = state
        t_dists_rv = {}
        for state, t_dist in t_dists.items():
            idx = state2idx[state]
            t_dists_rv[idx] = rv_from_dict(t_dist, state2idx)

        # Create random variates from e_dists if dictionaries
        if all_dicts:
            emits = {emit for e_dist in e_dists.values() for emit in e_dist}
            emit2idx = {}
            idx2emit = {}
            _emits = set()
            for idx, emit in enumerate(sorted(emits)):
                emit2idx[emit] = idx
                idx2emit[idx] = emit
                _emits.add(idx)
            e_dists_rv = {}
            for state, e_dist in e_dists.items():
                idx = state2idx[state]
                e_dists_rv[idx] = rv_from_dict(e_dist, emit2idx)
        else:
            emits = None
            emit2idx = IdentityGetter()
            idx2emit = IdentityGetter()
            _emits = None
            e_dists_rv = {state2idx[state]: e_dist for state, e_dist in e_dists.items()}

        # Create random variate from start_dist
        start_dist_rv = rv_from_dict(start_dist, state2idx)

        # Extract probability functions from e_dists
        attr = 'pdf' if all_pdfs else 'pmf'
        e_dists_pf = {state: getattr(e_dist, attr) for state, e_dist in e_dists_rv.items()}

        self.name = name
        self.states = states
        self._states = set(idx2state)
        self._state2idx = state2idx
        self._idx2state = idx2state
        self.emits = emits
        self._emit2idx = emit2idx
        self._idx2emit = idx2emit
        self.t_dists = t_dists
        self._t_dists_rv = t_dists_rv
        self.e_dists = e_dists
        self._e_dists_rv = e_dists_rv
        self._e_dists_pf = e_dists_pf
        self.start_dist = start_dist
        self._start_dist_rv = start_dist_rv
        self.stop_states = stop_states
        self._stop_states = [state2idx[state] for state in stop_states] if stop_states is not None else []

    def __repr__(self):
        pad = 4 * ' '
        return (f"HMM(states={self.states},\n"
                f"{pad}stop_states={self.stop_states},\n"
                f"{pad}name='{self.name}')")

    def simulate(self, step_max):
        """Simulate progression of states up to a maximum number of steps.

        Parameters
        ----------
        step_max: int
            Maximum number of steps to simulate. Simulation is terminated early
            if a stop state is encountered.

        Returns
        -------
        steps: list of tuples
            List of tuples as (state, emission)
        """
        if step_max == 0:
            return []
        state0 = self._start_dist_rv.rvs()
        emit0 = self._e_dists_rv[state0].rvs()
        steps = [(self._idx2state[state0], self._idx2emit[emit0])]
        for i in range(step_max-1):
            if state0 in self._stop_states:
                return steps
            state1 = self._t_dists_rv[state0].rvs()
            emit1 = self._e_dists_rv[state1].rvs()
            steps.append((self._idx2state[state1], self._idx2emit[emit1]))
            state0 = state1
        return steps

    def viterbi(self, emits):
        """Infer the most likely sequence of states yielding the given sequence of emissions.

        Parameters
        ----------
        emits: list
            List of emission labels.

        Returns
        -------
        tbs: list of lists
            List of sequences of states. While unlikely, it is possible for a
            emission sequence to have multiple distinct maximum probability
            state sequences.
        """
        # Forward pass
        emits = [self._emit2idx[emit] for emit in emits]  # Convert emits to internal labels
        vs = {state: [(log(self._e_dists_pf[state](emits[0])) + log(self._start_dist_rv.pmf(state)), [None])] for state in self._states}
        for i, emit in enumerate(emits[1:]):
            for state0 in self._states:
                # Get probabilities
                t_probs = {state1: vs[state1][i][0] + log(self._t_dists_rv[state1].pmf(state0)) for state1 in self._states}
                t_prob = max(t_probs.values())  # Probability of most likely path to state
                e_prob = log(self._e_dists_pf[state0](emit))

                # Get traceback states
                tb_states = [s for s, p in t_probs.items() if p == t_prob]
                vs[state0].append((e_prob+t_prob, tb_states))

        # Compile traceback states (taking care to allow for multiple paths)
        v_max = max([v[-1][0] for v in vs.values()])
        tbs = [[state] for state, v in vs.items() if v[-1][0] == v_max]
        for i in range(len(emits) - 1, 0, -1):
            new_tbs = []
            for tb in tbs:
                states = vs[tb[-1]][i][1]
                new_tbs.extend(tb + [state] for state in states)
            tbs = new_tbs
        tbs = [[self._idx2state[state] for state in tb[::-1]] for tb in tbs]  # Convert states to external labels

        return tbs

    def forward(self, emits):
        """Compute the forward variable for each state at each time point.

        To prevent numerical underflow, the forward variables are scaled, so
        the sum over all states at each time point is 1. The unscaled value at
        time i is given by s_0i*fs[state][i] where s_0i is the product of all
        scaling factors from 0 to i, inclusive. See section 3.6 of Durbin's
        Biological Sequence Analysis for more details.

        Parameters
        ----------
        emits: list
            List of emission labels.

        Returns
        -------
            fs: dict of lists
                Forward variables keyed by state label.
            ss: list
                Scaling factors at each time point.
        """
        if not emits:  # Catch empty inputs
            return {state: [] for state in self.states}, []

        # Initialize
        emits = [self._emit2idx[emit] for emit in emits]  # Convert emits to internal labels
        fs = {state: [self._e_dists_pf[state](emits[0]) * self._start_dist_rv.pmf(state)] for state in self._states}
        s = sum([fs[state][0] for state in self._states])
        for state in self._states:
            fs[state][0] /= s
        ss = [s]

        # Forward pass
        for i, emit in enumerate(emits[1:]):
            # Get probabilities
            for state0 in self._states:
                t_probs = [fs[state1][i] * self._t_dists_rv[state1].pmf(state0) for state1 in self._states]
                t_prob = sum(t_probs)  # Probability of all paths to state
                e_prob = self._e_dists_pf[state0](emit)
                fs[state0].append(e_prob*t_prob)

            # Scale probabilities
            s = sum([fs[state][i+1] for state in self._states])
            for state in self._states:
                fs[state][i+1] /= s
            ss.append(s)

        return {self._idx2state[state]: f for state, f in fs.items()}, ss  # Convert to external labels

    def backward(self, emits):
        """Compute the backward variable for each state at each time point.

        To prevent numerical underflow, the backward variables are scaled, so
        the sum over all states at each time point is 1. The unscaled value at
        time i is given by s_iN*bs[state][i] where s_iN is the product of all
        scaling factors from i to N, inclusive. See section 3.6 of Durbin's
        Biological Sequence Analysis for more details.

        Parameters
        ----------
        emits: list
            List of emission labels.

        Returns
        -------
            bs: dict of lists
                Backward variables keyed by state label.
            ss: list
                Scaling factors at each time point.
        """
        if not emits:  # Catch empty inputs
            return {state: [] for state in self.states}, []

        # Initialize
        emits = [self._emit2idx[emit] for emit in emits]  # Convert emits to internal labels
        bs = {state: [1] for state in self._states}
        s = sum([bs[state][0] for state in self._states])
        for state in self._states:
            bs[state][0] /= s
        ss = [s]

        # Backward pass
        for i, emit in enumerate(emits[:0:-1]):  # Reverse sequence starting from last emit excluding first
            # Get probabilities
            for state0 in self._states:
                probs = [bs[state1][i] * self._t_dists_rv[state0].pmf(state1) * self._e_dists_pf[state1](emit) for state1 in self._states]
                prob = sum(probs)  # Probability of all paths to state
                bs[state0].append(prob)

            # Scale probabilities
            s = sum([bs[state][i+1] for state in self._states])
            for state in self._states:
                bs[state][i+1] /= s
            ss.append(s)

        return {self._idx2state[state]: b[::-1] for state, b in bs.items()}, ss[::-1]  # Convert to external labels and undo reversal

    def forward_backward(self, emits):
        """Infer state probabilities at each time point.

        The probabilities are not re-normalized, so the sums of posterior
        probabilities across states at a given time point will not necessarily
        equal one. This is done to not give false confidence in the numerical
        precision of the results.

        Parameters
        ----------
        emits: list
            List of emission labels.

        Returns
        -------
        fb: dict of lists
            State probabilities at each time point keyed by the state labels.
        """
        fs, ss_f = self.forward(emits)
        bs, ss_b = self.backward(emits)
        p = reduce(lambda x, y: x+y, map(log, ss_f))
        ss_f = list(accumulate(map(log, ss_f)))
        ss_b = list(accumulate(map(log, ss_b[::-1])))[::-1]

        fbs = {state: [] for state in self.states}
        for i in range(len(emits)):
            for state, fb in fbs.items():
                fbs[state].append(fs[state][i]*bs[state][i]*exp(ss_f[i]+ss_b[i]-p))

        return fbs


class ARHMM:
    """A class for storing HMM model parameters and computing inferences from sequences of emissions.

    The implementation is essentially identical to the HMM class with a few
    exceptions to accommodate the emission distribution's dependence on the
    previous emission. First, the pmf or pdf method of the emission
    distributions must accept a tuple of the previous and current emissions,
    respectively. (Consequently, dictionary style emission distributions are
    not supported for ARHMMs.) The rvs method must also accept the previous
    emission as its first argument. Second, an ARHMM requires additional start
    emission distributions for each possible start state, as the behavior of
    the emission distribution is technically undefined when there is no previous
    emission.

    Otherwise the ARHMM class is identical to the HMM class, so please refer to
    it for addtional details.
    """
    def __init__(self, t_dists, e_dists, start_t_dist, start_e_dists, stop_states=None, name='arhmm'):
        # Check t_dists
        states = set(t_dists)
        for state, t_dist in t_dists.items():
            if not set(t_dist) <= states:
                raise ValueError(f'{state} t_dist contains a transition to an unknown state.')

        # Check e_dists
        if set(e_dists) != set(states):
            raise ValueError('States in e_dists do not match those in t_dists.')
        all_pmfs = all([getattr(e_dist, 'pmf', None) and getattr(e_dist, 'rvs', None) for e_dist in chain(e_dists.values(), start_e_dists.values())])
        all_pdfs = all([getattr(e_dist, 'pdf', None) and getattr(e_dist, 'rvs', None) for e_dist in chain(e_dists.values(), start_e_dists.values())])
        if not (all_pmfs or all_pdfs):
            raise ValueError('e_dists and start_e_dists are not all rvs of the same type.')

        # Check start_dists
        start_states = set(start_t_dist)
        if not start_states <= states:
            raise ValueError('start_t_dist contains a transition to an unknown state.')
        if set(start_e_dists) != start_states:
            raise ValueError('States in start_e_dists do not match those in start_t_dist.')

        # Create random variates from t_dists
        state2idx = {}
        idx2state = {}
        for idx, state in enumerate(sorted(states)):
            state2idx[state] = idx
            idx2state[idx] = state
        t_dists_rv = {}
        for state, t_dist in t_dists.items():
            idx = state2idx[state]
            t_dists_rv[idx] = rv_from_dict(t_dist, state2idx)

        # Create random variates from e_dists
        e_dists_rv = {state2idx[state]: e_dist for state, e_dist in e_dists.items()}

        # Create random variate from start_dists
        start_t_dist_rv = rv_from_dict(start_t_dist, state2idx)
        start_e_dists_rv = {state2idx[state]: e_dist for state, e_dist in start_e_dists.items()}

        # Extract probability functions from e_dists
        attr = 'pdf' if all_pdfs else 'pmf'
        e_dists_pf = {state: getattr(e_dist, attr) for state, e_dist in e_dists_rv.items()}
        start_e_dists_pf = {state: getattr(e_dist, attr) for state, e_dist in start_e_dists_rv.items()}

        self.name = name
        self.states = states
        self._states = set(idx2state)
        self._state2idx = state2idx
        self._idx2state = idx2state
        self.t_dists = t_dists
        self._t_dists_rv = t_dists_rv
        self.e_dists = e_dists
        self._e_dists_rv = e_dists_rv
        self._e_dists_pf = e_dists_pf
        self.start_t_dist = start_t_dist
        self._start_t_dist_rv = start_t_dist_rv
        self.start_e_dists = start_e_dists
        self._start_e_dists_rv = start_e_dists_rv
        self._start_e_dists_rv = start_e_dists_pf
        self.start_states = start_states
        self._start_states = {state2idx[state] for state in self.start_states}
        self.stop_states = stop_states
        self._stop_states = [state2idx[state] for state in stop_states] if stop_states is not None else []

    def __repr__(self):
        pad = 4 * ' '
        return (f"ARHMM(states={self.states},\n"
                f"{pad}stop_states={self.stop_states},\n"
                f"{pad}name='{self.name}')")

    def simulate(self, step_max):
        if step_max == 0:
            return []
        state0 = self._start_t_dist_rv.rvs()
        emit0 = self._start_e_dists_rv[state0].rvs()
        steps = [(self._idx2state[state0], emit0)]
        for i in range(step_max-1):
            if state0 in self._stop_states:
                return steps
            state1 = self._t_dists_rv[state0].rvs()
            emit1 = self._e_dists_rv[state1].rvs(emit0)
            steps.append((self._idx2state[state1], emit1))
            state0 = state1
            emit0 = emit1
        return steps

    def viterbi(self, emits):
        # Forward pass
        vs = {state: [(0, [None])] for state in self._states}
        for state in self._start_states:
            vs[state] = [(log(self._start_e_dists_pf[state](emits[0])) + log(self._start_t_dist_rv.pmf(state)), [None])]
        for i, emit in enumerate(emits[1:]):
            for state0 in self._states:
                # Get probabilities
                t_probs = {state1: vs[state1][i][0] + log(self._t_dists_rv[state1].pmf(state0)) for state1 in self._states}
                t_prob = max(t_probs.values())  # Probability of most likely path to state
                e_prob = log(self._e_dists_pf[state0]((emits[i], emit)))

                # Get traceback states
                tb_states = [s for s, p in t_probs.items() if p == t_prob]
                vs[state0].append((e_prob+t_prob, tb_states))

        # Compile traceback states (taking care to allow for multiple paths)
        v_max = max([v[-1][0] for v in vs.values()])
        tbs = [[state] for state, v in vs.items() if v[-1][0] == v_max]
        for i in range(len(emits) - 1, 0, -1):
            new_tbs = []
            for tb in tbs:
                states = vs[tb[-1]][i][1]
                new_tbs.extend(tb + [state] for state in states)
            tbs = new_tbs
        tbs = [[self._idx2state[state] for state in tb[::-1]] for tb in tbs]  # Convert states to external labels

        return tbs

    def forward(self, emits):
        if not emits:  # Catch empty inputs
            return {state: [] for state in self.states}, []

        # Initialize
        fs = {state: [0] for state in self._states}
        for state in self._start_states:
            fs[state] = [self._start_e_dists_pf[state](emits[0]) * self._start_t_dist_rv.pmf(state)]
        s = sum([fs[state][0] for state in self._states])
        for state in self._states:
            fs[state][0] /= s
        ss = [s]

        # Forward pass
        for i, emit in enumerate(emits[1:]):
            # Get probabilities
            for state0 in self._states:
                t_probs = [fs[state1][i] * self._t_dists_rv[state1].pmf(state0) for state1 in self._states]
                t_prob = sum(t_probs)  # Probability of all paths to state
                e_prob = self._e_dists_pf[state0]((emits[i], emit))
                fs[state0].append(e_prob*t_prob)

            # Scale probabilities
            s = sum([fs[state][i+1] for state in self._states])
            for state in self._states:
                fs[state][i+1] /= s
            ss.append(s)

        return {self._idx2state[state]: f for state, f in fs.items()}, ss  # Convert to external labels

    def backward(self, emits):
        if not emits:  # Catch empty inputs
            return {state: [] for state in self.states}, []

        # Initialize
        bs = {state: [1] for state in self._states}
        s = sum([bs[state][0] for state in self._states])
        for state in self._states:
            bs[state][0] /= s
        ss = [s]

        # Backward pass
        for i, emit in enumerate(emits[:0:-1]):  # Reverse sequence starting from last emit excluding first
            # Get probabilities
            for state0 in self._states:
                probs = [bs[state1][i] * self._t_dists_rv[state0].pmf(state1) * self._e_dists_pf[state1]((emits[-(i+2)], emit)) for state1 in self._states]
                prob = sum(probs)  # Probability of all paths to state
                bs[state0].append(prob)

            # Scale probabilities
            s = sum([bs[state][i+1] for state in self._states])
            for state in self._states:
                bs[state][i+1] /= s
            ss.append(s)

        return {self._idx2state[state]: b[::-1] for state, b in bs.items()}, ss[::-1]  # Convert to external labels and undo reversal

    def forward_backward(self, emits):
        fs, ss_f = self.forward(emits)
        bs, ss_b = self.backward(emits)
        p = reduce(lambda x, y: x+y, map(log, ss_f))
        ss_f = list(accumulate(map(log, ss_f)))
        ss_b = list(accumulate(map(log, ss_b[::-1])))[::-1]

        fbs = {state: [] for state in self.states}
        for i in range(len(emits)):
            for state, fb in fbs.items():
                fbs[state].append(fs[state][i]*bs[state][i]*exp(ss_f[i]+ss_b[i]-p))

        return fbs


class IdentityGetter:
    def __getitem__(self, key):
        return key


def rv_from_dict(d, label2idx):
    xs = []
    ps = []
    for x, p in d.items():
        xs.append(label2idx[x])
        ps.append(p)
    return stats.rv_discrete(values=(xs, ps))

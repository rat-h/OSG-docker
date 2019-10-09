COMMENT
Noise current characterized by normal distribution
with user-specified mean and standard deviation.

Borrows from NetStim's code so it can be linked with an external instance 
of the Random class in order to generate output that is independent of 
other instances of NUnif.

User specifies the time at which the noise starts, 
the duration of the noise,
and the interval at which new samples are drawn, 
For fixed dt integration, interval must be > dt (ideally a whole multiple of dt).
Current varies linearly with time between samples.

Note that, with fixed dt, roundoff error may occasionally
cause slight jitter (+-dt) in the actual sample intervals.
This will be most noticeable if the sample interval is
only a small multiple of dt.
With adaptive integration, jitter will happen very rarely, if at all.
ENDCOMMENT

NEURON {
:    POINT_PROCESS InUp : noisy current source, uniform distribution, piecewise linear in time
    POINT_PROCESS InNp : noisy current source, normal distribution, piecewise linear in time
    NONSPECIFIC_CURRENT i
    RANGE delay, dur, per
:    RANGE lo, hi
    RANGE mean, stdev
    RANGE genevent, y1
    THREADSAFE : true only if every instance has its own distinct Random
    POINTER donotuse
}

UNITS {
    (nA) = (nanoamp)
    (mA) = (milliamp)
}

PARAMETER {
    delay (ms) : delayay until noise starts
    dur (ms) <0, 1e9> : duration of noise
    per = 0.1 (ms) <1e-9, 1e9> : period i.e. interval at which new random values are returned
:    lo = 0 (nA)
:    hi = 1 (nA)
    mean = 0 (nA)
    stdev = 1 (nA)
    genevent = 0 (1) : if 1, generates an output event for each new value
      : enables use of NetCon.record to execute hoc code
      : e.g. to capture each new value just once
}

ASSIGNED {
    on
    ival (nA)
    i (nA)
    donotuse
    t0 (ms)
    y0 (nA)
    y1 (nA)
}

INITIAL {
    on = 0
    ival = 0
    i = 0
    net_send(delay, 1)
}

PROCEDURE seed(x) {
    set_seed(x)
}

BEFORE BREAKPOINT {
    if (on==0) {
        i = 0
    } else {
        i = y0 + ((t-t0)/per)*(y1 - y0)
    }
}

BREAKPOINT {
:    i = ival
}

FUNCTION yval() (nA) {
:    yval = (hi-lo)*urand() + lo
    yval = mean + nrand()*stdev : first sample
}

NET_RECEIVE (w) {
    if (dur>0) {
        if (flag==1) {
            if (on==0) { : turn on
                on=1
                net_send(dur,1) : to turn it off
                net_send(per, 2) : prepare for next sample
                t0 = t
                y0 = yval()
                y1 = yval()
                if (genevent==1) {
                    net_event(t) : to trigger recording of the new value
                }
            } else {
                if (on==1) { : turn off
                    on=0
                    y0 = 0
                    y1 = 0
                }
            }
        }
        if (flag==2) {
            if (on==1) {
                net_send(per, 2) : prepare for next sample
                t0 = t
                y0 = y1
                y1 = yval()
                if (genevent==1) {
                    net_event(t) : to trigger recording of the new value
                }
            }
        }
    }
}

VERBATIM
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
ENDVERBATIM

COMMENT
: FUNCTION erand() {
FUNCTION urand() {
VERBATIM
    if (_p_donotuse) {
        /*
         : Supports separate independent but reproducible streams for
         : each instance. However, the corresponding hoc Random
         : distribution MUST be set to Random.uniform(0,1)
         */
//            _lerand = nrn_random_pick(_p_donotuse);
            _lurand = nrn_random_pick(_p_donotuse);
    }else{
        /* only can be used in main thread */
        if (_nt != nrn_threads) {
hoc_execerror("multithread random in InUnif"," only via hoc Random");
        }
ENDVERBATIM
        : the old standby. Cannot use if reproducible parallel sim
        : independent of nhost or which host this instance is on
        : is desired, since each instance on this cpu draws from
        : the same stream
:        erand = exprand(1)
        urand = scop_random()
VERBATIM
    }
ENDVERBATIM
}
ENDCOMMENT

: FUNCTION erand() {
: FUNCTION urand() {
FUNCTION nrand() {
VERBATIM
    if (_p_donotuse) {
        /*
         : Supports separate independent but reproducible streams for
         : each instance. However, the corresponding hoc Random
:         : distribution MUST be set to Random.uniform(0,1)
         : distribution MUST be set to Random.normal(0,1)
         */
//            _lerand = nrn_random_pick(_p_donotuse);
//            _lurand = nrn_random_pick(_p_donotuse);
            _lnrand = nrn_random_pick(_p_donotuse);
    }else{
        /* only can be used in main thread */
        if (_nt != nrn_threads) {
hoc_execerror("multithread random in InUnif"," only via hoc Random");
        }
ENDVERBATIM
        : the old standby. Cannot use if reproducible parallel sim
        : independent of nhost or which host this instance is on
        : is desired, since each instance on this cpu draws from
        : the same stream
:        erand = exprand(1)
:        urand = scop_random()
        nrand = normrand(0,stdev/(1(nA)))
VERBATIM
    }
ENDVERBATIM
}

PROCEDURE noiseFromRandom() {
VERBATIM
 {
    void** pv = (void**)(&_p_donotuse);
    if (ifarg(1)) {
        *pv = nrn_random_arg(1);
    }else{
        *pv = (void*)0;
    }
 }
ENDVERBATIM
}

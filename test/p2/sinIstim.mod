TITLE Sinusoidal IStim


NEURON {
	POINT_PROCESS sinIstim
	NONSPECIFIC_CURRENT i
	RANGE tstart,tstop,bias,amp,per
}

PARAMETER {
	amp  = 7e-7	(mA)
	bias = 0	(mA)
	tstart = 200 (ms)
	tstop  = 2200 (ms)
	per = 2000 (ms)
}

UNITS {
	(mA) = (milliamp)
}

ASSIGNED {
	i (mA)
	amp2 (mA)
	pi2per (1/mc)
}
INITIAL{
	amp2 = amp/2.
	if (per < 1e-9){
		pi2per = 0
	} else {
		pi2per  = 3.14159265359*2./per
	}
}

:STATE {  }

BREAKPOINT {
	if (t > tstart && t <tstop) {
		if (per < 1e-9){
			i = (-1.0)*(amp2/2. + bias )
		} else {
			i = (-1.0)*( amp2*(1.-cos(pi2per*(t-tstart))) + bias) :minus, because it is inward current
		}
	}else{
		i = (-1.0)*bias
	}
}

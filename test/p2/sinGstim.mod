TITLE Sinusoidal conductance modulation


NEURON {
	POINT_PROCESS sinGstim
	NONSPECIFIC_CURRENT i
	RANGE tstart,tstop,bias,gmax,per,E
}

PARAMETER {
	gmax  = 7e-7	(mS)
	E     = -75	(nV)
	bias = 0	(mS)
	tstart = 200 (ms)
	tstop  = 2200 (ms)
	per = 2000 (ms)
}

UNITS {
	(mA) = (milliamp)
}

ASSIGNED {
	i (nA)
	gmax2 (mA)
	pi2per (1/mc)
}
INITIAL{
	gmax2 = gmax/2.
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
			i = 1e3*(v-E)*(gmax/2. + bias )
		} else {
			i = 1e3*(v-E)*( gmax2*(1.-cos(pi2per*(t-tstart))) + bias) 
		}
	}else{
		i = 1e3*(v-E)*bias
	}
}

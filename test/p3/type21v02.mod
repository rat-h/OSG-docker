TITLE type21v02.mod  a second version of a simple planar type-1/type-2 model

COMMENT
Reduce planar model for Hodgkin-Huxley model
  with can be witched between Type-1 and Type-2 dynamics.

It has approximately the same resting potential, input resistance,
  spike shape and frequencies (where F-I curves aren't zero) 
  for both models.

--- Model equations ---
C dv/dt = I + g_L(E_L-v)+ g_{Na}m^3_\infty(v)(a + bn)(E_{Na}-v)-g_Kn^4(E_K-v)\\
  dn/dt = (n_\infty(v) - n)/\tau_n(v)

m_\infty(v) =   1/( 1+e^(-(v+40)/9.5) )
n_\infty(v) = n_0 + (1-n_0)/(1+e^( -(v-v_{1/2})/\theta ) )
\tau_n(v)   = \tau_0+ s_\tau e^( -( (v - v_0)/\sigma  )^2 )

--- A map of the model parameters into parameter names in the mod-file ---

          :      : Type I  : Type II :
----------:------:---------:---------:
g_L       : gl   :   0.3   :   0.1   :
E_L       : el   : -54.3   : -39.0   :
g_{Na}    : gna  :       120.0       :
a         : a    :  0.906483183915   :
b         : b    : -1.10692947808    :
E_{Na}    : ena  :       50.0        :
g_K       : gk   :       36.0        :
E_K       : ek   :      -77.0        :
n_0       : n0   :   0.35  :   0.28  :
v_{1/2}   : v12  : -40.0   : -44.5   :
\theta    : sl   :   4.0   :   9.0   :
\tau_0    : t0   :   0.46  :   0.5   :
s_\tua    : st   :   3.5   :   5.0   :
v_0       : v0   : -60.5   : -60.0   :
\sigma    : sg   :  35.9   :  30.0   :

--- USAGE ---
The type-1/type-2 mode canbe switched by variable type21.
If type21 = 1, the model is set into type 1 dynamics, and
  any other parameters are ignored.
If type21 = 2, the model is set into type 2 dynamics.
To gain access to other parameters, set type21 to zero.


--- Neuron parameters for 1000 um2 membrane and 1 uF/cm2 capacitance. ---
Resting potential (mV)
    Type-1     : -67.78432212370292
    Type-2     : -67.91262149648327
    difference : 0.1282993727803472

Input resistance (MOhm) for 1000 um2 compartment
----------------------: Type-1:Type-2
for positive current  : 174   : 203
for negative current  : 176   : 203

Input resistance (Ohm cm2)
----------------------: Type-1:Type-2
for positive current  : 1741  : 2032
for negative current  : 1761  : 2027

Steady-state values for zero input current
    type21 = 1, vinit = -67.78432212370292, ninit = 0.35062495845399
    type21 = 2, vinit = -67.91262149648327, ninit = 0.32971471805597

-----
Spike Threshold:
    Type-I  : -44.3427953154(mV)
    Type-II : -44.8076699075(mV)
Spike duration:
    Type-I  : 0.39(ms)
    Type-II : 0.4(ms)
Spike Height:
    Type-I  : 88.002566626(mV)
    Type-II : 88.3969740276(mV)
AHP:
    Type-I  : 31.0120287902(mV)
    Type-II : 30.547154198(mV)

--- --- ---
developed by Ruben A. Tikidji-Hamburyan, LSU HSC, 2018-11-21

ENDCOMMENT

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
        (mS) = (millisiemens)
}
 
NEURON {
	SUFFIX type21
	NONSPECIFIC_CURRENT i
	RANGE  ninit            : initial conditions for n
	                        : if negative, it uses stady-state for given
	                        : voltage
	RANGE  type21           : 1 - for type-1
	                        : 2 - for type-2, 
	                        : 0 - to enable parameters below
	:>> THIS PARAMETERS ARE PRESET BY type21 AT INIT
		RANGE gl,el,v12,sl  
		RANGE n0,sn,t0,st,v0,sg
	:<<
	:>> Parameters below have default velues
		RANGE gk,ek,gna,ena,a,b
	:<<
	:GLOBAL minf, ninf, ntau
}
 
PARAMETER {
        v               (mV)
        type21=2        (1)
        gna= 120.       (mS/cm2)
        ena=  50.       (mV)
        gk =  36.       (mS/cm2)
        ek = -77.       (mV)
        gl              (mS/cm2)
        el              (mV)
        n0              (1)
        sn              (1)
        t0              (ms)
        st              (ms)
        v0              (mV)
        sg              (mV)
        v12             (mV)
        sl              (mV)
        a  =  0.906483183915
        b  = -1.10692947808
        ninit = 0.34
	: type21 = 1, vinit = -67.78432212370292, ninit = 0.35062495845399
	: type21 = 2, vinit = -67.91262149648327, ninit = 0.32971471805597
}
 
STATE {
   n
}

ASSIGNED {
        i       (mA/cm2) 
        minf
        ninf
        ntau    (ms)
}
 
BREAKPOINT {
	SOLVE states METHOD cnexp
	:----vvvv-- is needed to convert uA/cm2 to mA/cm2
	i = (1e-3)*( gna*minf*minf*minf*(a+n*b)*(v-ena)+gk*n*n*n*n*(v-ek)+gl*(v-el) )
}
 
DERIVATIVE states { 
	rates(v)
	n'= (ninf- n)/ ntau 
}


INITIAL {
	if ( fabs(type21 - 1.) < 1e-6 ){
		: Paramters for type 1
		gl  =   0.3  (mS/cm2)
		el  = -54.3  (mV)
		n0  =   0.35
		sn  =   1. - n0
		v12 = -40.   (mV)
		sl  =   4.   (mV)
		t0  =    .46 (ms)
		st  =   3.5  (ms)
		v0  = -60.5  (mV)
		sg  =  35.9  (mV)
		:printf("Type - I\n")
	} 
	if ( fabs(type21 - 2.) < 1e-6 ){
		: Paramters for type 2
		gl  =   0.1  (mS/cm2)
		el  = -39.   (mV)
		n0  =   0.28
		sn  =   1. - n0
		v12 = -44.5  (mV)
		sl  =   9.   (mV)
		t0  =    .5  (ms)
		st  =   5.   (ms)
		v0  = -60.   (mV)
		sg  =  30.   (mV)
		:printf("Type - II\n")
	} 
	rates(v)
	if (ninit < 0 || ninit > 1){
		n = ninf
	} else {
		n = ninit
	}
}

PROCEDURE rates(v (mV)) {
UNITSOFF 
	:TABLE minf, ninf, ntau FROM -100 TO 100 WITH 200
	minf =      1./(1.+exp(-(v+40.)/9.5))
	ninf = n0 + sn/(1.+exp(-(v-v12)/sl ))
	ntau = t0 + st*exp(-((v-v0)/sg)*((v-v0)/sg))
	
UNITSON
}

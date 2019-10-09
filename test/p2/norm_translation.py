import numpy as np

def getscale(Atau1,Atau2,Btau1,Btau2):
	Atp = (Atau1*Atau2)/(Atau2 - Atau1) * np.log(Atau2/Atau1)
	Afactor = -np.exp(-Atp/Atau1) + np.exp(-Atp/Atau2)
	Afactor = 1./Afactor

	Asp=1./np.abs(Atau2-Atau1)

	Btp = (Btau1*Btau2)/(Btau2 - Btau1) * np.log(Btau2/Btau1)
	Bfactor = -np.exp(-Btp/Btau1) + np.exp(-Btp/Btau2)
	Bfactor = 1./Bfactor

	Bsp=1./np.abs(Btau2-Btau1)

	nFactor=Afactor*(Atau2-Atau1)/Bfactor/(Btau2-Btau1)
	print "A:Tau1=%g, Tau2=%g"%(Atau1,Atau2)
	print "B:Tau1=%g, Tau2=%g"%(Btau1,Btau2)
	print "nFactor = %g"%nFactor
	return nFactor



if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import sys,os,csv
	from neuron import h

	Atau1,Atau2 = 1.,3.
	Btau1,Btau2 = 2.,5.

	nFactor = getscale(Atau1,Atau2,Btau1,Btau2)
	
	soma1,     soma2	= h.Section(),h.Section()
	soma1.L,   soma1.L	= 1.,1.
	soma1.diam,soma2.diam=1/np.pi,1/np.pi
	soma1.nseg,soma2.nseg=1,1
	soma1.cm,  soma2.cm	=1,1
	soma1.insert('pas')
	soma1(0.5).pas.g = 0.00025
	soma1(0.5).pas.e = -70
	soma2.insert('pas')
	soma2(0.5).pas.g = 0.00025
	soma2(0.5).pas.e = -70
	syn1 = 	h.Exp2Syn(0.5, sec=soma1)
	syn2 = 	h.Exp2Syn(0.5, sec=soma2)
	syn1.e,syn2.e = 0,0
	syn1.tau1,syn2.tau1 = Atau1,Btau1
	syn1.tau2,syn2.tau2 = Atau2,Btau2
	netstims = h.NetStim()
	netstims.start = 255
	netstims.noise = 0

	netcon1 = h.NetCon(netstims,syn1, -10, 0, 5e-7)
	netcon2 = h.NetCon(netstims,syn2, -10, 0, 5e-7*nFactor)

	soma1(0.5).v = soma2(0.5).v  = -70

	t       = h.Vector()
	is1,is2 = h.Vector(),h.Vector()
	gs1,gs2 = h.Vector(),h.Vector()
	vs1,vs2 = h.Vector(sec=soma1),h.Vector(sec=soma2)

	t.record(h._ref_t)
	gs1.record(syn1._ref_g)
	gs2.record(syn2._ref_g)
	is1.record(syn1._ref_i)
	is2.record(syn2._ref_i)
	vs1.record(soma1(0.5)._ref_v)
	vs2.record(soma2(0.5)._ref_v)

	
	h.finitialize()
	h.fcurrent()
	h.frecord_init()
	while(h.t<700.): h.fadvance()

	print np.sum(np.array(gs1)),np.sum(np.array(gs2))
	ax=plt.subplot(311)
	plt.plot(t,is1,"r-")
	plt.plot(t,is2,"b-")
	plt.subplot(312,sharex=ax)
	plt.plot(t,gs1,"r-")
	plt.plot(t,gs2,"b-")
	plt.subplot(313,sharex=ax)
	plt.plot(t,vs1,"r-")
	plt.plot(t,vs2,"b-")
	plt.show()
	


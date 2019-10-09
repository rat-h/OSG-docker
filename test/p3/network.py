"""
/***********************************************************************************************************\

 
 Based on the script for the paper:
 
 Copyright: Ruben Tikidji-Hamburyan <rath@gwu.edu> <rtikid@lsuhsc.edu> Aug.2018 - ....

\************************************************************************************************************/
"""
import numpy as np
from scipy import optimize
from scipy import integrate
import scipy.stats as sps
import sys,os,csv,threading
import random as rnd
from functools import reduce
try:
	import pickle as pickle
except:
	import pickle
from datetime import datetime
import time

from neuron import h
h.load_file("stdgui.hoc")

os.system("nrnivmodl")


###### Abbreviations:
Abbreviations=(
	( 'I',       'I',   "Current"),
	( 'TI',      'TI',  "Total Current"),
	( 'TSI',     'TSI', "Total Synaptic Current"),
	( 'MTI',     'MTI', "Mean Total Curent"),
	( 'MTSI',    'MTSI',"Mean Total Synaptic Current"),
	( 'G',       'G',   "Condunctance"),
	( 'TG',      'TG',  "Total Conductance"),
	( 'MTG',     'MTG', "Mean Total Conducntance"),
	( 'FR',      'FR',  "Firing Rate"),
	( 'NORM',   'NORM',"Normal distribution"),
	( 'LOGN',   'LOGN',"Log Normal Distribution"),
	( 'Mstate', 'm',   "Sodium Activation variable"), 
	( 'Hstate', 'h',   "Sodium Inctivation variable"),
	( 'Nstate', 'n',   "Potassium Activation variable"),
	( 'ucon',   'ucon',"Uniform distribution of number connection per cell"),
	( 'ncon',   'ncon',"Normal distribution of number connection per cell"),
	( 'bcon',   'bcon',"Binomial distribution of number connection per cell"),
	( 'ON',     'True',"Turn ON some parameter"),
	( 'OFF',     'False',"Turn OFF some parameter"),
)

for ab,val,meaning in Abbreviations:	
	print("Applay abbreviation % 8s for % 8s for: "%(ab,val)+meaning, end=' ')
	try:
		exec(ab+'=\''+val+'\'')
	except:
		print("Fail!")
		exit(1)
	print("DONE")


###### Paramters:
methods		= {
	'ncell'		: 300,			# number of neurons in population
	'ncon'		: ('b',0.133),	# number of input connections per neuron
								# constant or uniform distribution(from, to) or normalized uniform distribution (mena, stder, ncon-norm)
								# OR uniform distribution  ('u', from, {to},    {{ncon-norm}})
								# OR normal distribution   ('n', mean, {stdev}, {{ncon-norm}})
								# OR binomial distribution ('b', prob,           {ncon-norm} )
	'neuron'	: {
		'Vinit'		: (-50.,20),#(-51.86007190636312,20),	# Constant or (mean,stdev) or [value for each neuron] or string or file name there values for each neuron are contained.
		'Type'      : 1,
		'Iapp'      : None,
		'Istdev'	: None,						# ---/---/---

	},
	'synapse'	: {
		#'weight'	: 0.75e-2,					# Synaptic conductance
		'weight'	: 0.03e-2,					# Synaptic conductance
		'delay'		: 0.8,						# Axonal Delays
		'gsynscale'	: 1.0,						# Conductance caramelization
		'tau1'		: 1.0,
		'tau2'		: 3.0,
		'Esyn'		: -75.0,					# Synaptic reversal potential
												# Constant or (mean,stdev) or [value for each neuron] or string or file name there values for each neuron are contained.
		'synscaler'	: None,

	},		
	'R2'		: True,
	'maxFreq'	: 200.0,		# max frequency
	'peakDetec' : True,			# Turn on/off peak detector
	'gkernel'	: (3.,25.),#(5.0,25.0),#(10.0,50.0),	# Kernel and size (5,25),#
	"netFFT"	: False,#True,#False,#True,#False,		# Turn on/off network FFT
	"nrnFFT"	: False,#True,#False,		# Turn on/off neuron FFT
	'netISI'	: 30001,			# max net ISI
	'nrnISI'	: 30001,			# max neuron ISI
	'cliptrn'	: False,#1000,#False alse,#500,#False,	# Clip transience for first n ms or False
	'traceView'	: False,#'n',
	'tV-synmax' : False,
	'traceWidth': 55.0,			#
	'tracetail'	: 'mean total conductance',#'conductance',#'mean total conductance',#'conductance',#'current',#'conductance',#'firing rate',#'total conductance', #'total current' 'total current',
	'patview'	: True,			# Turn on/off Pattern view
	'gui'		: True,
	'git'		: False,		# Turn on/off git core (Never turn-on at the head node!!!!!!!)
	'gif'		: False,		# Generate gif instead pop up on a screan.
	'corefunc'	: (4,8,64),
	'coreindex'	: False,			# Turn on/off Core indexing
	'corelog'	: 'network',
	'noexit'	: False,
	'GPcurve'	: False,
	'IGcurve'	: False,
	'Conn-rec'	: False,
	'Conn-stat'	: False,
	'G-Spikerate'	: False,
	
	'Gtot-dist'	: False,
	'Gtot-rec'	: False,#True,	#record all gtotal in neurons
	'Gtot-stat'	: True, #False, #record gtot statistic
	'sycleprop'	: False,#True,
	'external'	: False,
	'extprop'	: 0.5,				# Calculate probability to fire after external input
	'timestep'	: 0.025,#0.005,
	'sortbysk'	: False, #'ST',#'F',#'GT',#'NC',#'GT','G',#'I', #'F',#'I',#'F',#False,			#Do not use
	'taunorm'	: False,#True,#False,#True,
	'nstart'	: False, #(900.,0.1e-5,1000),<Noise for paper  #False,#(900.,0.2e-5,1000),#False,#(200,0.000002,900),#False, (delay, ampl, dur)
	'cliprst'	: False,#10,#False,#20,
	'T&S'		: False,#True,
	'lastspktrg': True,
	'fullrast'	: True,
	'gtot-dist'	: 'LOGN', #LOGN - lognormal, 'NORM' - normal
	'gsyn-dist' : 'LOGN',#'NORM',#'LOGN', #same
	'cycling'	: False, #4,False
	'popfr'		: False,#True,	#calculate population firing rate
	'cmd-file'	: 'network.start',
	'preview'	: True,
	'2cintercon': False,#True,
	'2clrs-stat': False,#True,
	'tv'		: (0., 500.),
	'tstop'		: 10001,
	'jitter-rec': False,#True,
	'pop-pp-view':False,#True,
	'N2NHI'     :True,		#neuron to network harmonic index
	'N2NHI-netISI' :False,	#the same index but using netowkr ISI to get network harmonics (it is slow)
	'vpop-view' : False,
	'CtrlISI'   : False,#{'bin'   : 5.,'max'   : 120.,},# ISI histogram with controled bin and max
	'nrnFRhist' : False, #Neuron Firing rate histogram
}



class neuron:
	def __init__(self):
		self.soma = h.Section()
		if checkinmethods('/neuron/L'):
			self.soma.L		= methods["neuron"]["L"]
		else:
			self.soma.L		= 100.
		if checkinmethods('/neuron/diam'):
			self.soma.diam	= methods["neuron"]["diam"]
		else:
			self.soma.diam	= 10./np.pi
		if checkinmethods('/neuron/nseg'):
			self.soma.nseg	= int(methods["neuron"]["nseg"])
		else:
			self.soma.nseg	= 1
		if checkinmethods('/neuron/cm'):
			self.soma.cm	= float(methods["neuron"]["cm"])
		self.soma.insert('type21')
		self.soma(0.5).type21.type21 = 1 #default type
		self.type21 = 1
		if checkinmethods('/neuron/set'):
#			print "=================================="
#			print "===      SETTING PARAMETERS    ==="
#			print "  >  Cells in the Cluster A      :",countA
			for p in methods["neuron"]["set"]:
				exec("self.soma(0.5)."+p+" = {}".format(methods["neuron"]["set"][p]))
			
		self.soma(0.5).v = -67.
		self.isyn	= h.Exp2Syn(0.5, sec=self.soma)
		self.isyn.e		= -75.0
		self.isyn.tau1	= 2.0
		self.isyn.tau2	= 10.0
		######## Recorders ##########
		self.spks	= h.Vector()
		self.sptr	= h.APCount(.5, sec=self.soma)
		self.sptr.thresh = 0.#25
		self.sptr.record(self.spks)
		#self.sptr = h.NetCon(self.soma(0.5)._ref_v,None,sec=self.soma)
		#self.sptr.threshold = 25.
		#self.sptr.record(self.spks)
		if checkinmethods('gui'):
			self.volt	= h.Vector()
			self.volt.record(self.soma(0.5)._ref_v)
			if checkinmethods("neuron/record/current"):
				self.isyni	= h.Vector()
				self.isyni.record(self.isyn._ref_i)
			if checkinmethods('neuron/record/conductance') or checkinmethods('pop-pp-view'):
				self.isyng	= h.Vector()
				self.isyng.record(self.isyn._ref_g)
			if checkinmethods('traceView') or checkinmethods('pop-pp-view'):
				self.svar   = h.Vector()
				self.svar.record(self.soma(0.5)._ref_n_type21)
		elif checkinmethods('get-steadystate'):
			self.volt	= h.Vector()
			self.volt.record(self.soma(0.5)._ref_v)
		if checkinmethods('sinmod'):
			self.sin = h.sinIstim(0.5, sec=self.soma)
			if type(methods['sinmod']) is dict:
				for name in methods['sinmod']:
					exec("self.sin."+name+"= {}".format(methods['sinmod'][name]))

		######## Registrations ###### 
		self.gsynscale	= 0.0
		self.concnt		= 0.0
		self.gtotal		= 0.0
		self.tsynscale	= 1.0

	def setparams(self, 
			V=None, N=-1, Type=1,
			Iapp = None, Insd = None, delay = None, duration = None, period = None,
			SynE = None, SynT1 = None, SynT2 = None,
			):
		if not    V is None : self.soma(0.5).v             = V
		if not    N is None : self.soma(0.5).type21.ninit  = N
		if not Type is None : self.soma(0.5).type21.type21 = self.type21 = Type
		########
		if not Iapp is None or not Insd is None:
			self.innp	= h.InNp(0.5, sec=self.soma)
			self.rnd	= h.Random(np.random.randint(0,32562))
			if checkinmethods("/neuron/rnd") and ( methods["neuron"]["rnd"] == "u"  or methods["neuron"]["rnd"] == "U" ):
				self.rnd.uniform(0.,1.)
			self.innp.noiseFromRandom(self.rnd)
			self.innp.dur	= 1e9 if duration == None else duration
			self.innp.delay	= 0   if delay == None else delay
			self.innp.per	= 0.1 if period == None else period
			self.innp.mean	= 0.0 if Iapp == None else Iapp
			self.innp.stdev	= 0.0 if Insd == None else Insd
			if checkinmethods("/neuron/rnd") and ( methods["neuron"]["rnd"] == "u"  or methods["neuron"]["rnd"] == "U" ):
				self.innp.stdev = -self.innp.stdev - self.innp.mean
			if methods['gui']:
				self.inoise	= h.Vector()
				self.inoise.record(self.innp._ref_i)
			elif checkinmethods("rawdata") and type(methods["rawdata"]) is str:
				self.inoise	= h.Vector()
				self.inoise.record(self.innp._ref_i)
		########
		if not SynE  is None: self.isyn.e		= SynE
		if not SynT1 is None: self.isyn.tau1	= SynT1
		if not SynT2 is None: self.isyn.tau2	= SynT2
		########
			
			

	def addnoise(self,Iapp=0.,Insd=0.,delay=0.,dur=0.,per=0.1):
		self.andnoise = h.InNp(0.5, sec=self.soma)
		self.andrnd	= h.Random(np.random.randint(0,32562))
		self.andnoise.noiseFromRandom(self.andrnd)
		self.andnoise.mean  = Iapp
		self.andnoise.stdev = Insd
		self.andnoise.delay	= delay
		self.andnoise.per	= per
		self.andnoise.dur	= dur
		self.iandnoise	    = h.Vector()
		self.iandnoise.record(self.andnoise._ref_i)

#class symulation:
	#def __init___(self,params):
		#if params.get("a",False):

def getType21(nid):
	for i in 'type21,gna,ena,gk,ek,gl,el,n0,sn,t0,st,v12,sl,a,b'.split(","):
		exec("{0} = neurons[{1}].soma(0.5).type21.{0}".format(i,nid))
	s  = neurons[nid].soma.L * neurons[0].soma.diam * np.pi * 1e-8 # cm2
	es = neurons[nid].isyn.e
	return type21,gna,ena,gk,ek,gl,el,n0,sn,t0,st,v12,sl,a,b,s,es


def getnulls(nid,vmin,vmax,gsyn,inoise,ibias):
	type21,gna,ena,gk,ek,gl,el,n0,sn,t0,st,v12,sl,a,b,s,es = getType21(nid)
	#DB>>
	if checkinmethods("local-parameters"):
		print("======== THE SET =======")
		for i in 'type21,gna,ena,gk,ek,gl,el,n0,sn,t0,st,v12,sl,a,b,s,gsyn,es,inoise,ibias'.split(","):
			print(" > % 6s          = "%i,eval(i))
	#<<DB
	
	## N-nullcline
	vx=np.linspace(vmin,vmax,200)
	ninf = lambda v:n0 + sn/(1.+np.exp(-(v-v12)/sl ))
	ntau = lambda v:t0 + st/(1.+np.exp( (v+40.)/20.))
	n0c  = np.dstack( (vx, ninf(vx)) )[0]

	## V - null	
	dvdt = lambda n,v,I:I+gl*(el-v)+gna*(1./(1.+np.exp(-(v+40.)/9.5)))**3*(b*n+a)*(ena-v)+gk*n**4*(ek-v)	
	def vfun(vx,I):
		"""
		Solves 4th order algebraic equation and returns null-cline
		"""
		if type(I) is float or type(I) is int:
			I = np.ones(vx.shape[0])*I 
		res = []
		for vp,ip in zip(vx,I):
			try:
				n=optimize.fsolve(dvdt,0.5,args=(vp,ip),xtol=0.01)[0]	
			except: return np.array([[],[]])
			res.append((vp,n))
		return np.array(res)	

	v0c  = vfun(vx,  -ibias                  *1e-3 /s )
	v0n  = vfun(vx,( -inoise - gsyn*(vx- es))*1e-3 /s )
	if type21 == 2:
		vXn = v0c[ np.where(v0c[:,0] > -40.) ]
		try:
			d2vdt2 = np.polyder(np.polyfit(vXn[:,0], vXn[:,1], 3),m=2)
			vnX = vXn[ np.argmax(d2vdt2) ]
		except:
			vnX = vXn[ np.argmax(vXn[:,1]) ]
		#DB>>
		#print " > % 9s       = "%("Vinit"),vnX
		#<<DB
		dvdt = lambda n,v:-ibias*1e-3/s+gl*(el-v)+gna*(1./(1.+np.exp(-(v+40.)/9.5)))**3*(b*n+a)*(ena-v)+gk*n**4*(ek-v)
		def rhs(t,Y): return [-dvdt(Y[1],Y[0]),-(ninf(Y[0])-Y[1])/ntau(Y[0])]
		slv = integrate.ode(rhs).set_initial_value(vnX, 0)#.set_integrator('zvode', method='bdf')
		thc = [ slv.integrate(slv.t+0.01) ]
		vbl,vbr = vx[0],vnX[0]+0.1
		while slv.successful() and slv.t < 100. and vbl < thc[-1][0] < vbr and 0. < thc[-1][1] < 1:
			thc.append( slv.integrate(slv.t+0.01) )
		thc = np.array(thc)
		#========#
		vXn = v0n[ np.where(v0n[:,0] > -40.) ]
		try:
			d2vdt2 = np.polyder(np.polyfit(vXn[:,0], vXn[:,1], 3),m=2)
			vnX = vXn[ np.argmax(d2vdt2) ]
		except:
			vnX = vXn[ np.argmax(vXn[:,1]) ]
		#DB>>
		#print " > % 9s       = "%("Vinit"),vnX
		#<<DB
		dvdt = lambda n,v:(-inoise-gsyn*(v-es))*1e-3/s+gl*(el-v)+gna*(1./(1.+np.exp(-(v+40.)/9.5)))**3*(b*n+a)*(ena-v)+gk*n**4*(ek-v)
		def rhs(t,Y):return [-dvdt(Y[1],Y[0]),-(ninf(Y[0])-Y[1])/ntau(Y[0])]
		slv = integrate.ode(rhs).set_initial_value(vnX, 0)#.set_integrator('zvode', method='bdf')
		thn = [ slv.integrate(slv.t+0.01) ]
		vbl,vbr = vx[0],vnX[0]+0.1
		while slv.successful() and slv.t < 100. and vbl < thn[-1][0] < vbr and 0. < thn[-1][1] < 1:
			thn.append( slv.integrate(slv.t+0.01) )
		thn = np.array(thn)
		
	else:
		thc = None
		thn = None

	return n0c,v0c,v0n,thc,thn,type21

def onclick1(event):
	if not hasattr(onclick1,"aix"):
		onclick1.aix=zooly.add_subplot(111)
	onclick1.et = event.xdata
	
	### BUG
	onclick1.tl, onclick1.tr = onclick1.et-methods['traceWidth'], onclick1.et+methods['traceWidth']
	onclick1.idx, = np.where( (t > onclick1.tl) * (t < onclick1.tr))
	
	if not hasattr(onclick1,"marks"):
		onclick1.marks = []
		onclick1.marks.append( p.plot([onclick1.tl,onclick1.tl],[-80,30],"r--",lw=2)[0] )
		onclick1.marks.append( p.plot([onclick1.tr,onclick1.tr],[-80,30],"r--",lw=2)[0] )
	else:
		onclick1.marks[0].set_xdata([onclick1.tl,onclick1.tl])
		onclick1.marks[1].set_xdata([onclick1.tr,onclick1.tr])
	
	if not hasattr(onclick1,"lines"):
		onclick1.lines = []
		for n in neurons:
			volt = np.array(n.volt)
			onclick1.lines.append(onclick1.aix.plot(t[onclick1.idx],volt[onclick1.idx])[0])
	else:
		vmin,vmax = 1000,-1000
#		for ind,n in map(None,xrange(methods["ncell"]),neurons):
		for ind,n in enumerate(neurons):
			volt = np.array(n.volt)
			if vmin > volt[onclick1.idx].min():vmin = volt[onclick1.idx].min()
			if vmax < volt[onclick1.idx].max():vmax = volt[onclick1.idx].max()
			onclick1.lines[ind].set_xdata(t[onclick1.idx])
			onclick1.lines[ind].set_ydata(volt[onclick1.idx])
			onclick1.lines[ind].set_linewidth(1)
			onclick1.lines[ind].set_ls("-")

		onclick1.aix.set_xlim(onclick1.tl,onclick1.tr)
		#print vmin,"---",vmax
		onclick1.aix.set_ylim(vmin,vmax)
	if hasattr(moddyupdate,"lines"):
		del moddyupdate.lines
	mainfig.canvas.draw()
	zoolyupdate(vindex)
	moddyupdate(moddyupdate.idx)



def neuronsoverview(event):
	global vindex
	if event.key == "up":
		vindex += 1
		if vindex >= methods["ncell"] : vindex = methods["ncell"] -1
	elif event.key == "down":
		vindex -= 1
		if vindex < 0 : vindex = 0
	elif event.key == "home": vindex = 0
	elif event.key == "end" : vindex = methods["ncell"] -1
	elif event.key == "/"   : vindex = int(methods["ncell"]/2)
	elif event.key == "pageup":
		vindex += 10
		if vindex >= methods["ncell"] : vindex = methods["ncell"] -1
	elif event.key == "pagedown":
		vindex -= 10
		if vindex < 0 : vindex = 0
	nsorted = methods['sortbysk'] == 'I'  or methods['sortbysk'] == 'G' or methods['sortbysk'] == 'NC' or\
	          methods['sortbysk'] == 'GT' or methods['sortbysk'] == 'ST'or methods['sortbysk'] == 'N'  or\
	          methods['sortbysk'] == 'T'  or methods['sortbysk'] == 'FR'
	if nsorted:
		  print(vindex,"->",nindex[vindex][1],"("+methods['sortbysk']+")=",nindex[vindex][0])
		  vtrace.set_ydata( np.array(neurons[nindex[vindex][1]].volt)[tproc:tproc+tprin.size])
	else:
		vtrace.set_ydata( np.array(neurons[vindex].volt)[tproc:tproc+tprin.size])
	
	if methods['tracetail'] == 'conductance':
		if nsorted :
			xvcrv.set_ydata( np.array(neurons[nindex[vindex][1]].isyng)[tproc:tproc+tprin.size]*1e5 )
		else:
			xvcrv.set_ydata( np.array(neurons[vindex].isyng)[tproc:tproc+tprin.size]*1e5 )
	elif methods['tracetail'] == 'current':
		if nsorted :
			xvcrv.set_ydata( np.array(neurons[nindex[vindex][1]].isyni)[tproc:+tprin.size]*1e5 )
		else:
			xvcrv.set_ydata( np.array(neurons[vindex].isyni)[tproc:+tprin.size]*1e5 )
	if event.key == "H":
		if nsorted:
			p.plot(tprin,np.array(neurons[nindex[vindex][1]].volt)[tproc:tprin.size+tproc],"-")
		else:
			p.plot(tprin,np.array(neurons[vindex].volt)[tproc:tprin.size+tproc],"-")
	mainfig.canvas.draw()
	if checkinmethods('traceView'):
		if nsorted:
			zoolyupdate(nindex[vindex][1])
		else: 
			zoolyupdate(vindex)
		moddyupdate(moddyupdate.idx)

def positiveGauss(mean,stdev):
	result = -1
	while result < 0:
		result = mean + np.random.randn()*stdev
	return result

def checkinmethods(item, dirtree = methods):
	def getsubitems(item):
		items = item.split("/")
		if items[ 0] == "" and len(items) !=1 : items = items[1:]
		if items[-1] == "" and len(items) !=1 : items = items[:-1]
		return items[0],"/".join(items[1:])
	item,subitems = getsubitems(item)
	if subitems != "":
		if not item in dirtree : return False
		if not type(dirtree[item]) is dict: return False
		return checkinmethods(subitems,dirtree[item])
	else:
		if not item in dirtree : return False
		if not ( (type(dirtree[item]) is bool or type(dirtree[item]) is int) ):
			if dirtree[item] is None: return False
			else: return True
		return bool(dirtree[item])

def ggap_var(t,t0,t1,r0,r1):
	if t < t0:
		for gj in gapjuctions: gj[0].r, gj[1].r = r0, r0
	elif t > t1:
		for gj in gapjuctions: gj[0].r, gj[1].r = r1, r1
	else :
		r = (r1-r0)*(t-t0)/(t1-t0)+r0
		for gj in gapjuctions: gj[0].r, gj[1].r = r, r
	#DB>>
#	print "ggap_var was called with parameters", t,t0,t1,r0,r1
#	exit(0)
	#<<DB
	
def getNdist(prm):
	if type(prm) is float or type(prm) is int:
		return prm*np.ones(methods["ncell"])
	elif type(prm) is tuple:
		if len(prm) > 1:
			if type(prm[0]) is not str:
				return prm[0]+np.random.randn(methods["ncell"])*prm[1]
			else:
				if   prm[0][0] == "n" or prm[0][0] == "N":
					if   len(prm) == 3:return prm[1]+np.random.randn(methods["ncell"])*prm[2]
					elif len(prm) == 2:return prm[1]
					else:
						print("ERROR: normal distribution should have mean and std parameters ('n',mean,std)!")
						exit(1)
				elif prm[0][0] == "u" or prm[0][0] == "U":
					if   len(prm) == 3:return prm[1]+np.random.rand(methods["ncell"])*(prm[2]-prm[1])
					else:
						print("ERROR: normal distribution should have two parameters left and right borders ('u',min,max)!")
						exit(1)
				#elif prm[0][0] == "l" or prm[0][0] == "L":
				#elif prm[0][0] == "s" or prm[0][0] == "S": #shifted lognormal
				#elif prm[0][0] == "t" or prm[0][0] == "T": #Truncated normal
					#if   len(prm) == 4:
						#return prm[1]+np.random.rand(methods["ncell"])*(prm[2]-prm[1])
				else:
						print("ERROR: unknown distribution type for parameter {}!".format(prm))
						exit(1)
		else:
			return prm[0]*np.ones(methods["ncell"])
	elif type(prm) is list:
		if len(prm) == methods['ncell'] :
			return np.array(prm)
		else: 
			return [ None for i in range(methods["ncell"]) ]
	elif type(prm) is str:
		return np.genfromtxt(prm)
		print("  > Read Vinit from file         :",prm)
	elif prm is None:
		return[ None for i in range(methods["ncell"]) ]

#elif methods["delay-dist"] == "NORM":
						#### Trancated normal
						#dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
						#if len(methods['synapse']['delay']) < 3:
							#while(dx < methods['timestep']*2):
								#dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
						#else:
							#while(dx < methods['synapse']['delay'][2]):
								#dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
							
					#elif methods["delay-dist"] == "LOGN":
						#### Lognormal
						#dlym,dlys=methods['synapse']['delay'][0]-methods['timestep']*2.,methods['synapse']['delay'][1]
						#if len(methods['synapse']['delay']) < 3:
							#dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['timestep']*2
						#else:
							#dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))
							#while dx < methods['synapse']['delay'][2]:
								#dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))
					#elif methods["delay-dist"] == "LOGN-SHIFT":
						#### Lognormal
						#dlym,dlys=methods['synapse']['delay'][0]-methods['timestep']*2.,methods['synapse']['delay'][1]
						#if len(methods['synapse']['delay']) < 3:
							#dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['timestep']*2
						#else:
							#dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['synapse']['delay'][2]
					#elif methods["delay-dist"] == "DIST":
						#dmin = methods['synapse']['delay'][0]
						#dinciment = methods['synapse']['delay'][1] if len(methods['synapse']['delay']) >= 2 else 0.
						#dx = dmin + dinciment*float(abs(pre-x[0]))		
			
	
#===============================================#
#               MAIN PROGRAMM                   #
#===============================================#
if __name__ == "__main__":
	if len(sys.argv) > 1:
		def setmethod(arg):
			global methods
			if not "=" in arg: return
			try:
				name,value = arg.split("=")
			except: 
				print("ERROR! Parameter %s has not = symbol"%arg)
				exit(1)
			if not "/" in name: return
			if name[0] != '/' : return
			names = name.split("/")
			if len(names) > 2:
				name = "methods"
				for nm in names[1:-1]:
					inmethods= eval("\'"+nm+"\' in "+name)
					if inmethods :
						inmethods= eval("type("+name+"[\'"+nm+"\']) is dict")
						if inmethods :
							name += "[\'"+nm+"\']"
							continue
						else:
							inmethods= eval("type("+name+"[\'"+nm+"\']) is bool or type("+name+"[\'"+nm+"\']) is None")
							if inmethods :
								name += "[\'"+nm+"\']"
								exec(name+"={}")
							else:
								sys.stdout.write("method item %s of %s isn't dict\n"%(name,nm))
					else:
						name += "[\'"+nm+"\']"
						exec(name+"={}", globals())
					
			cmd = "methods" + reduce(lambda x,y: x+"[\'"+y+"\']", names[1:],"") + "=" + value
			try:
				exec(cmd, globals())
			except: 
				#cmd = "methods" + reduce(lambda x,y: x+"[\'"+y+"\']", names[1:],"") + "=" + "\'\\\'"+value+"\\\'\'"
				cmd = "methods" + reduce(lambda x,y: x+"[\'"+y+"\']", names[1:],"") + "=" + "\'"+value+"\'"
				exec(cmd, globals())
		def readfromsimdb(simdb,ln):
			rec = None
			with open(simdb) as fd:
				for il,l in enumerate(fd.readlines()):
					if il == ln: rec = l
			if rec == None:
				sys.stderr.write("Cannot find line %d in the file %s\n"%(ln,simdb))
				exit(1)
			for itm in rec.split(":"):
				#DB>>
				print(itm)
				#<<DB
				setmethod(itm)
		simdbrec=None
		for arg in sys.argv:
			if arg[:len('--readsimdb=')] == '--readsimdb=':
				simdbrec=arg[len('--readsimdb='):]
			#if arg == '-h' or arg == '-help' or arg == '--h' or arg == '--help':
				#print __doc__
				#print 
				#print "USAGE: nrngui -nogui -python network.py [parameters]"
				#print "\nPARAMETERS:"
				#print "-n=          number of neurons in population"
				#print "-c=          number of connections per neuron"
				#print "-Iapp=       apply current. Use scaling factor 1e-5 to get nA"
				#print "             Iapp may be a constant or mean,standard deviation across population."
				#print "-Istd=       amplitude of noise. Should be scaled by 1e-5 to get nA"
				#print "-gui=ON|OFF  Turn on/off gui and graphs"
				#print "-F=          Set up neuron dynamics scale factor"
				#print "             F may be a constant or mean,standard deviation across population."
				#print "-gsyn=       conductance of single synapse. Use scaling factor 1e-5 to get nS"
				#print "             gsyn may be a constant or mean,standard deviation for all synapses in model"
				#print "-gsynscale=  total synaptic conductance.  Use scaling factor 1e-5 to get nS"
				#print "             gsynscale may be a constant or mean,standard deviation for all neurons within population"
				#print "-tau1=       rising time constant in ms"
				#print "-tau2=       falling time constant in ms"
				#print "-Esyn=       synaptic reversal potential in mV"
				#print "-taunorm=0|1 On or Off normalization by space under the curve"
				#print "-tsynscaler= scaling coefficient for synaptic time constants"
				#print "-delay=      axonal delay in ms"
				#print "             delay may be a constant or mean,standard deviation for all synapses in model"
				#print "-view        limits simulation and save memory"
				
				#exit(0)
			
		if not simdbrec is None:
			simdbrec = simdbrec.split(":")
			if len(simdbrec) < 2:
				sys.stderr.write("Error format --readsimdb=file:record\n")
			readfromsimdb( simdbrec[0], int(simdbrec[1]) ) 
		for arg in sys.argv: setmethod(arg)
			

	if 'cmd-file' in methods:
		if not type(methods['cmd-file']) is str: methods['cmd-file'] = 'network.start'
	else:
		methods['cmd-file'] = 'network.start'
	with open(methods['cmd-file'],"w") as fd:
		for arg in sys.argv: fd.write("%s "%arg)
	if methods['taunorm']:
		from norm_translation import getscale
		nFactor = getscale(1.0,3.0,methods['synapse']['tau1'],methods['synapse']['tau2'])
		if type(methods["synapse"]["weight"]) == tuple or type(methods["synapse"]["weight"]) == list:
			methods["synapse"]["weight"]		= (methods["synapse"]["weight"][0]*nFactor, methods["synapse"]["weight"][1]*nFactor)
		else:
			methods["synapse"]["weight"] *= nFactor
	if checkinmethods('preview'):
		methods['tstop'] = methods['tv'][1]

###DB>
	print("==================================")
	print("==       ::  METHODS  ::        ==")
	def dicprn(dic, space):
		for nidx,name in enumerate(sorted([ x for x in dic ])):
			if type(dic[name]) is dict:
				rep = "%s%s\\ %s "%(space,"v-" if nidx==0 else "|-", name)
				print(rep)
				dicprn(dic[name], space+"  ")
			else:
				rep = "%s%s> %s "%(space,"`-" if nidx==len(dic)-1 else "|-", name)
				if len(rep) < 31:
					for x in range(31-len(rep)):rep += " "
				if type(dic[name]) is str:
					print(rep," : ","\'%s\'"%dic[name])
				else:
					print(rep," : ",dic[name])
	dicprn(methods,' ')
	print("==================================\n")
###<DB

	
	if methods["gui"]:
		import matplotlib
		import matplotlib.pyplot as plt
		matplotlib.rcParams["savefig.directory"] = ""
		#cmap = matplotlib.cm.get_cmap('jet')
		#cmap = matplotlib.cm.get_cmap('plasma')
		#cmap = matplotlib.cm.get_cmap('autumn')
		#cmap = matplotlib.cm.get_cmap('gist_rainbow')
		cmap = matplotlib.cm.get_cmap('rainbow')
		print("==================================")
		print("===        GUI turned ON       ===")
		print("==================================\n")
	
	h.tstop 	= float(methods['tstop'])
	#h.v_init 	= V
	h.dt		= float(methods["timestep"])
	if checkinmethods('temperature'):
		temp = float(h.celsius)
		h.celsius = methods['temperature']
		print("==================================")
		print("===       SET TEMPERATURE      ===")
		print(" > set temperature")
		print(" \> from {} to {} celsius degree  ".format(temp,methods['temperature']))
		print("==================================\n")
		

	if checkinmethods('simvar'):
		if not type(methods['simvar']) is dict: methods['simvar'] = False
		elif not "type" in methods['simvar']: methods['simvar'] = False
		elif not type(methods['simvar']["type"]) is str: methods['simvar'] = False
		elif not (methods['simvar']["type"] == 'n' or methods['simvar']["type"] == 'c' or methods['simvar']["type"] == 'g'  ): methods['simvar'] = False
		elif not "var" in methods['simvar']: methods['simvar'] = False
		elif not type(methods['simvar']["var"]) is str: methods['simvar'] = False
		elif not "a0" in methods['simvar']: methods['simvar'] = False
		elif not (type(methods['simvar']["a0"]) is float or type(methods['simvar']["a0"]) is int): methods['simvar'] = False
		elif not "a1" in methods['simvar']: methods['simvar'] = False
		elif not (type(methods['simvar']["a1"]) is float or type(methods['simvar']["a1"]) is int): methods['simvar'] = False
		elif not "t0" in methods['simvar']: methods['simvar']['t0'] = methods['tv'][0]
		elif not (type(methods['simvar']["t0"]) is float or type(methods['simvar']["t0"]) is int): methods['simvar'] = False
		elif not "t1" in methods['simvar']: methods['simvar']['t1'] = methods['tv'][1]
		elif not (type(methods['simvar']["t1"]) is float or type(methods['simvar']["t1"]) is int): methods['simvar'] = False
	if methods['tracetail'] == 'R2':
		if not checkinmethods("cont-R2"):
			print("Need /cont-R2= parameter with period of R2 sliding window")
			exit(1)	

	#### Save mamory, record only what is needed
	if checkinmethods('gui'):
		print("==================================")
		print("===          RECORDER          ===")

		methods['neuron']['record'] = {}
		if methods['tracetail'] == 'total conductance' or methods['tracetail'] == 'mean total conductance' or\
		   methods['tracetail'] == 'TG'                or methods['tracetail'] == 'MTG'                    or\
		   methods['tracetail'] == 'conductance'       or\
		   checkinmethods('traceView'):
			   methods['neuron']['record']['conductance'] = True
			   print(" > RECORD                       : cunductance")
		
		if methods['tracetail'] == 'total current'               or methods['tracetail'] == 'TI'   or\
		   methods['tracetail'] == 'mean total current'          or methods['tracetail'] == 'MTI'  or\
		   methods['tracetail'] == 'total synaptic current'      or methods['tracetail'] == 'TSI'  or\
		   methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'MTSI' or\
		   methods['tracetail'] == 'current'                     or\
		   checkinmethods('spectrogram'):
			   methods['neuron']['record']['current'] = True
			   print(" > RECORD                       : current")
		if methods['tracetail'] == 'LFP':
			methods[ "peakDetec" ] = True
		print("==================================\n")

	#### Create Neurons and setup noise and Iapp
	print("==================================")
	print("===        Create Neurons      ===")
	neurons = [ neuron() for x in range(methods["ncell"]) ]
	
	if   type(methods["neuron"]["Type"]) is int:
		xT = [ methods["neuron"]["Type"] for i in range(methods["ncell"]) ]
	elif type(methods["neuron"]["Type"]) is float:
		xT = [ 2 if i < methods["neuron"]["Type"] else 1 for i in np.random.random(methods["ncell"]) ]
	else:
		print("ERROR: unknown type of /neuron/Type parameter!")
		exit(1)
	

	xV      = getNdist( methods["neuron"]["Vinit"] )
	xEsyn   = getNdist( methods['synapse']['Esyn'] )
	xIapp   = getNdist( methods["neuron"]["Iapp"]  )
	xIstdev = getNdist( methods["neuron"]["Istdev"])

	
	for n,i in zip(neurons,range(methods["ncell"])):
		if not methods['synapse']['synscaler'] is None:
			if type(methods['synapse']['synscaler']) is float or type(methods['synapse']['synscaler']) is int:
				n.tsynscale = float(methods['synapse']['synscaler'])
				xTau1,xTau2 = methods['synapse']['tau1'] * n.tsynscale, methods['synapse']['tau2'] * n.tsynscale
			elif type(methods['synapse']['synscaler']) is list or type(methods['synapse']['synscaler']) is tuple:
				if len(methods['synapse']['synscaler']) >= 2:
					n.tsynscale = -1.0
					while( n.tsynscale < 0.0 ):
						n.tsynscale = methods['synapse']['synscaler'][0] + np.random.randn()*methods['synapse']['synscaler'][1]
					xTau1,xTau2 = methods['synapse']['tau1'] * n.tsynscale, methods['synapse']['tau2'] * n.tsynscale
				else :
					n.tsynscale = float(methods['synapse']['synscaler'][0])
					xTau1,xTau2 = methods['synapse']['tau1'] * n.tsynscale, methods['synapse']['tau2'] * n.tsynscale
			else:
				xTau1,xTau2 = methods['synapse']['tau1'],methods['synapse']['tau2']
		else:
			xTau1,xTau2 = methods['synapse']['tau1'],methods['synapse']['tau2']

		n.setparams(
			V=xV[i], Type = xT[i],
			SynT1=xTau1, SynT2=xTau2, SynE=xEsyn[i], 
			Iapp = xIapp[i] if xIapp[i] is None else -1.*xIapp[i], Insd=xIstdev[i]
			)
	print("==================================\n")
	
	if checkinmethods("neuron/distribution"):
		if type(methods["neuron"]["distribution"]) is not dict:
			print("==================================")
			print("===           ERROR            ===")
			print("=== neuron/distribution should ===")
			print("=== be a set of parameters and ===")
			print("=== parameters def.            ===")
			print("==================================\n")
			exit(1)
		#>> Init neurons
		h.finitialize()
		#>> do not init as a specific type
		for n in neurons: n.soma(0.5).type21.type21 = 0
		#>> set new distribution parameters
		for k,v in list(methods["neuron"]["distribution"].items()):
			pl = getNdist( v )
			for n,p in zip(neurons,pl):
				try:
					exec("n.soma(0.5).type21."+k+" = p")
				except: 
					print("Cannot set parameter",k)

	if checkinmethods('nstart'):
		if type(methods['nstart']) is list or type(methods['nstart']) is tuple:
			methods['nstart'] = {
				'delay'    : methods['nstart'][0],
				'Istdev'   : methods['nstart'][1],
				'duration' : methods['nstart'][2],
			}
		if not checkinmethods('nstart/Iapp'     ): methods['nstart']['Iapp'    ] = 0.
		if not checkinmethods('nstart/period'   ): methods['nstart']['period'  ] = 0.1
		if not checkinmethods('nstart/delay'    ): methods['nstart']['delay'   ] = 0.
		if not checkinmethods('nstart/duration' ): methods['nstart']['duration'] = 1e9
			
		if not checkinmethods('nstart/Istdev'):
			raise RuntimeError("/nstart/Istdev isn't set up")
		for n in neurons:
			n.addnoise(\
				Iapp  = methods['nstart']['Iapp'],\
				Insd  = methods['nstart']['Istdev'],\
				delay = methods['nstart']['delay'],\
				dur   = methods['nstart']['duration'],\
				per   = methods['nstart']['period'] )


	#DB>>
	if checkinmethods("DB-nrn"):
		h.finitialize()
		h.fcurrent()
		h.frecord_init()
		for n in neurons:
			print(n.soma(0.5).type21.type21)
			for x in "gl,el,n0,v12,sl,t0,st,v0,sg".split(","):
				print("    ","% 3s"%x,"=",eval("n.soma(0.5).type21."+x))
			print("-----")
		exit(0)
	#<<DB

	t = h.Vector()
	t.record(h._ref_t)


	#### Create Connection List:
	if checkinmethods("ncon"):
		def CreateConnectionList():
			def CreateFixNumberOrRange(n0,n1=None):
				OUTList = [ [] for x in range(methods["ncell"])]
				for toid in range(methods["ncell"]):
					from_excaption = [ 0 for x in range(methods["ncell"]) ]
					from_excaption[toid] = 1
					upcnt = 0
					total = 0
					if not n1 is None:
						neurons[toid].concnt = int(np.random.random()*(n1-n0) + n0)
					else:
						neurons[toid].concnt = n0
					while  upcnt < 10000*methods["ncell"]:
						upcnt += 1
						fromid = rnd.randint(0, methods["ncell"]-1)
						if from_excaption[fromid] == 1 : continue
						upcnt  = 0
						total += 1
						from_excaption[fromid] = 1
						OUTList[toid].append(fromid)
						if total >= neurons[toid].concnt :break
					else:
						sys.stderr.write("Couldn't obey connections conditions\nNeuron:%d TOTLA:%d CURRENT:%d\n"%(toid,n0,total))
						for x in OUTList:
							sys.stderr.write("ID:%d #%d\n"%(x[0],x[1]))
						sys.exit(1)
				return OUTList
			def CreateNormalDistribution(mean,stdev=0.):
				OUTList = [ [] for x in range(methods["ncell"])]
				for toid in range(methods["ncell"]):
					from_excaption = [ 0 for x in range(methods["ncell"]) ]
					from_excaption[toid] = 1
					upcnt = 0
					total = 0
					neurons[toid].concnt = int( positiveGauss(mean,stdev) )
					while  upcnt < 10000*methods["ncell"]:
						upcnt += 1
						fromid = rnd.randint(0, methods["ncell"]-1)
						if from_excaption[fromid] == 1 : continue
						upcnt  = 0
						total += 1
						from_excaption[fromid] = 1
						OUTList[toid].append(fromid)
						if total >= neurons[toid].concnt :break
					else:
						sys.stderr.write("Couldn't obey connections conditions\nNeuron:%d TOTLA:%d CURRENT:%d\n"%(toid,n0,total))
						for x in OUTList:
							sys.stderr.write("ID:%d #%d\n"%(x[0],x[1]))
						sys.exit(1)
				return OUTList
			def CreateBinomialDistribution(prob):
				OUTList = [ [] for x in range(methods["ncell"])]
				for toid in range(methods["ncell"]):
					for fromid in range(methods["ncell"]):
						if fromid == toid: continue
						if np.random.random() > prob : continue
						OUTList[toid].append(fromid)
						neurons[toid].concnt += 1
					#DB>>
					#print OUTList[toid]
					#<<DB
				return OUTList
#>>			
			#DB>>
			#print type(methods["ncon"]),methods["ncon"]
			#<<DB
			if type(methods["ncon"]) is int:
				#DB>>
				#print "ncon - int"
				#<<DB
				return CreateFixNumberOrRange(methods["ncon"])
			elif type(methods["ncon"]) is tuple or type(methods["ncon"]) is list:
				#DB>>
				#print "Ncon tuple or list"
				#<<DB
				if type(methods["ncon"][0]) is int:
					if len(methods["ncon"]) > 2:
						methods["normalize-weight-by-ncon"] = methods["ncon"][2]
						return CreateFixNumberOrRange(methods["ncon"][0],methods["ncon"][1])
					elif len(methods["ncon"]) > 1:
						return CreateFixNumberOrRange(methods["ncon"][0],methods["ncon"][1])
					else:
						return CreateFixNumberOrRange(methods["ncon"][0])
				elif type(methods["ncon"][0]) is str:
					if methods["ncon"][0] == "u":
						#print "  > Uniform Distribution "
						if len(methods["ncon"]) > 3:
							methods["normalize-weight-by-ncon"] = methods["ncon"][3]
							return CreateFixNumberOrRange(methods["ncon"][1],methods["ncon"][2])
						elif len(methods["ncon"]) > 2:
							return CreateFixNumberOrRange(methods["ncon"][1],methods["ncon"][2])
						elif len(methods["ncon"]) > 1:
							return CreateFixNumberOrRange(methods["ncon"][1])
						else:
							print("ERROR in ncon parameter:\nUSAGE of uniform distribution: /ncom=('u',n-from, {n-to}, {{norm-by}})")
							sys.exit(1)
				
					if methods["ncon"][0] == "n":
						if len(methods["ncon"]) > 3:
							methods["normalize-weight-by-ncon"] = methods["ncon"][3]
							return CreateNormalDistribution(methods["ncon"][1],methods["ncon"][2])
						elif len(methods["ncon"]) > 2:
							return CreateNormalDistribution(methods["ncon"][1],methods["ncon"][2])
						elif len(methods["ncon"]) > 1:
							return CreateFixNumberOrRange(methods["ncon"][1])
						else:
							print("ERROR in ncon parameter:\nUSAGE for noormal distribution: /ncom=('n',mean, {stdev}, {{norm-by}})")
							sys.exit(1)

					if methods["ncon"][0] == "b":
						if len(methods["ncon"]) > 2:
							methods["normalize-weight-by-ncon"] = methods["ncon"][1]
							return CreateBinomialDistribution(methods["ncon"][1])
						elif len(methods["ncon"]) > 1:
							return CreateBinomialDistribution(methods["ncon"][1])
						else:
							print("ERROR in ncon parameter:\nUSAGE for binomial distribution: /ncom=('b',prob,{norm-by})")
							sys.exit(1)

		print("==================================")
		print("===        Map Connections     ===")
		if checkinmethods('connectom'):
			print("  > Try to Read Connectom file :", methods['connectom'])
			if os.access(methods['connectom'],os.R_OK):
				print("  > File is accessible         : ")
				with open(methods['connectom'],"r") as fd:
					xncell = pickle.load(fd)
					xnconn = pickle.load(fd)
					if xncell != methods['ncell'] or xnconn != methods['ncon']:
						print("  > File has different numbers : ")
						print("  > n cell                     : ", xncell,"|",methods['ncell'])
						print("  > n connection               : ", xnconn,"|",methods['ncon'])
						OUTList = CreateConnectionList()
					else:
						print("  > Read Connection Map        : ", end=' ')
						OUTList = pickle.load(fd)
						for n,cpn in zip(neurons,pickle.load(fd)):
							n.concnt = cpn
						if not checkinmethods("normalize-weight-by-ncon"):
							methods["normalize-weight-by-ncon"] = pickle.load(fd)
						print("Successfully")
					
			elif not os.access(methods['connectom'],os.F_OK):
				print("  > File dos not exist         : try to create")
				OUTList = CreateConnectionList()
				with open(methods['connectom'],"w") as fd:
					pickle.dump(methods['ncell'],fd)
					pickle.dump(methods['ncon'],fd)
					pickle.dump(OUTList,fd) 
					pickle.dump([ n.concnt for n in neurons ],fd)
					if checkinmethods("normalize-weight-by-ncon"):
						pickle.dump(methods["normalize-weight-by-ncon"],fd)
					else:
						pickle.dump(False,fd)
			else:
				print()
				print("============= ERROR =============")
				print(" > Cannot create file \'{}\' ".format(methods['connectom']))
				print()
				exit(0)
		else:
			print(" > Generate connections         :", end=' ')
			OUTList = CreateConnectionList()
			print("Successfully")
		print("==================================\n")

	#DB>
		#for i in OUTList:
			#print len(i)
			#for j in i:	print "%03d"%(j),
			#print
		#sys.exit(0)
	#<DB
	if checkinmethods('cycling'):
		print("==================================")
		print("===      Cycles counting       ===")
		mat=np.matrix( np.zeros((methods["ncell"],methods["ncell"])) )
#		for i,vec in map(None,xrange(methods["ncell"]),OUTList):
		for i,vec in enumerate(OUTList):
			mat[i,vec]=1
		kx = []
		for cnt in range(methods['cycling']):
			kx.append(np.trace(mat)/methods["ncell"])
			mat = mat.dot(mat)
		print(" > Cyclopedic numbers           : ",kx)
		print("==================================\n")
		methods['cycling-result'] = kx
		del mat
		
		
	

	#### Create Conneactions:
	if checkinmethods("ncon"):
		print("==================================")
		print("===    Make the Connections    ===")
		print("==================================\n")
		connections = []
		if not checkinmethods("gtot-dist"):  methods["gtot-dist"]  = "NORM"
		if not checkinmethods("gsyn-dist"):  methods["gsyn-dist"]  = "NORM"
		if not checkinmethods("delay-dist"): methods["delay-dist"] = "NORM"
#		for x in map(None,xrange(methods["ncell"]),OUTList):
		for x in enumerate(OUTList):
			if type(methods['synapse']['gsynscale']) is int or type(methods['synapse']['gsynscale']) is float:
				gx = float(methods['synapse']['gsynscale'])
			elif type(methods['synapse']['gsynscale']) is tuple :
				#DB>>
				#print "DB: TUPLE size=",len(methods['synapse']['gsynscale'])
				#print "DB: DIST",methods["gtot-dist"]
				#exit(0)
				#<<DB
				if methods['synapse']['gsynscale'][1] <= 0.: 
					gx  = methods['synapse']['gsynscale'][0] 
					gx *= 1. if len(methods['synapse']['gsynscale']) < 3 else methods['synapse']['gsynscale'][2]
				elif methods["gtot-dist"] == "NORM":
					### Trancated normal
					gx = positiveGauss(methods['synapse']['gsynscale'][0],methods['synapse']['gsynscale'][1])
				elif methods["gtot-dist"] == "LOGN":
					### Lognormal
					if len(methods['synapse']['gsynscale']) != 2 and len(methods['synapse']['gsynscale']) != 3:
						print("ERROR: wrong scaler size!\n/synapse/gsynscale should have 2 or more parameters")
						exit(1)
					elif len(methods['synapse']['gsynscale']) == 2:
						gsymtotm,gsymtots,gsyntotsc=methods['synapse']['gsynscale'],1.
						##DB>>
						#print gsymtotm,gsymtots,gsyntotsc
						#exit(0)
						##<<DB
					elif len(methods['synapse']['gsynscale']) == 3:
						gsymtotm,gsymtots,gsyntotsc=methods['synapse']['gsynscale']
						##DB>>
						#print gsymtotm,gsymtots,gsyntotsc
						#exit(0)
						##<<DB
					gx = gsyntotsc*np.random.lognormal(mean=np.log(gsymtotm/np.sqrt(1.+gsymtots**2/gsymtotm**2)),sigma=np.sqrt(np.log(1.+gsymtots**2/gsymtotm**2)))
					#DB>>
					#print "DB: gx=",gx
					#<<DB
			else:
				print("ERROR: wrong type of/synapse/gsynscale")
				exit(1)
				
			neurons[x[0]].gsynscale = gx
			for pre in x[1]:
				if type(methods['synapse']['delay']) is tuple :
					if methods['synapse']['delay'][1] <= 0: dx = float(methods['synapse']['delay'][0])
					elif methods["delay-dist"] == "NORM":
						### Trancated normal
						dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
						if len(methods['synapse']['delay']) < 3:
							while(dx < methods['timestep']*2):
								dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
						else:
							while(dx < methods['synapse']['delay'][2]):
								dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
							
					elif methods["delay-dist"] == "LOGN":
						### Lognormal
						dlym,dlys=methods['synapse']['delay'][0]-methods['timestep']*2.,methods['synapse']['delay'][1]
						if len(methods['synapse']['delay']) < 3:
							dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['timestep']*2
						else:
							dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))
							while dx < methods['synapse']['delay'][2]:
								dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))
					elif methods["delay-dist"] == "LOGN-SHIFT":
						### Lognormal
						dlym,dlys=methods['synapse']['delay'][0]-methods['timestep']*2.,methods['synapse']['delay'][1]
						if len(methods['synapse']['delay']) < 3:
							dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['timestep']*2
						else:
							dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['synapse']['delay'][2]
					elif methods["delay-dist"] == "DIST":
						dmin = methods['synapse']['delay'][0]
						dinciment = methods['synapse']['delay'][1] if len(methods['synapse']['delay']) >= 2 else 0.
						dx = dmin + dinciment*float(abs(pre-x[0]))
					elif methods["delay-dist"] == "RING":
						dmin = methods['synapse']['delay'][0]
						dinciment = methods['synapse']['delay'][1] if len(methods['synapse']['delay']) >= 2 else 0.
						dist = min([float(abs(pre-x[0])),float(abs(pre-x[0]-methods['ncell'])),float(abs(pre-x[0]+methods['ncell']))])
						dx = dmin + dinciment*dist
					elif methods["delay-dist"] == "UNIFORM":
						dmin = methods['synapse']['delay'][0]
						dmax = methods['synapse']['delay'][1]
						dx = dmin + np.random.rand()*(dmax-dmin)
						
				else:
					dx = float(methods['synapse']['delay'])
				if checkinmethods("gtot-set"):
					wx=1.
				else:
					if type(methods['synapse']['weight']) is tuple :
						if methods['synapse']['weight'][1] <= 0: wx = methods['synapse']['weight'][0]
						elif methods["gsyn-dist"] == "NORM":
							#### Trancated normal
							wx = methods['synapse']['weight'][1]*np.random.randn()+methods['synapse']['weight'][0]
							while wx < 0.0 : wx = methods['synapse']['weight'][1]*np.random.randn()+methods['synapse']['weight'][0]
						elif methods["gsyn-dist"] == "LOGN":
							### Lognormal
							wm,ws=methods['synapse']['weight']
							wx = np.random.lognormal(mean=np.log(wm/np.sqrt(1.+ws**2/wm**2)),sigma=np.sqrt(np.log(1.+ws**2/wm**2)))
							#wx = np.random.lognormal(mean=np.log(wm**2/np.sqrt(wm**2+ws**2)),sigma=np.sqrt(np.log(1.+ws**2/wm**2)))
					else:
						wx = float(methods['synapse']['weight'])
					#if type(methods["ncon"]) is tuple or type(methods["ncon"]) is list:
						#if len(methods["ncon"]) > 2:
							#wx *= float(methods["ncon"][2])/float(neurons[x[0]].concnt)
					if checkinmethods("normalize-weight-by-ncon"):
						#DB>>
						#print "Norm by Factor",float(methods["normalize-weight-by-ncon"]),float(neurons[x[0]].concnt),float(methods["normalize-weight-by-ncon"])/float(neurons[x[0]].concnt)
						#<<DB

						wx *= float(methods["normalize-weight-by-ncon"])/float(neurons[x[0]].concnt)
					if methods['taunorm'] and not methods['synapse']['synscaler'] is None:
						#DB print "norm by factor",1./neurons[x[0]].tsynscale
						wx /= neurons[x[0]].tsynscale
				#####DB>>
				#print "DB:gx=",gx,"dx=",dx,"wx=",wx
				#####<<DB
				#connections.append( (h.NetCon(neurons[pre].soma(0.5)._ref_v,neurons[x[0]].isyn,
						#0., dx, gx*wx,
						#sec=neurons[pre].soma),pre,x[0]) )
				#connections.append( (h.NetCon(neurons[pre].soma(0.5)._ref_v,neurons[x[0]].isyn,
						#25., dx, gx*wx,
						#sec=neurons[pre].soma),pre,x[0]) )
				connections.append( (h.NetCon(neurons[pre].soma(0.5)._ref_v,neurons[x[0]].isyn,
						0., dx, gx*wx,
						sec=neurons[pre].soma),pre,x[0]) )
				neurons[x[0]].gtotal += gx*wx
		if checkinmethods('Conn-alter'):
			print("================================================")
			print("===        ALTER CONNECTIONS SETTINGS        ===")
			n_alter = methods['Conn-alter']['n']      if checkinmethods('Conn-alter/n')      else 1
			d_alter = methods['Conn-alter']['delay']  if checkinmethods('Conn-alter/delay')  else 0.8
			w_alter = methods['Conn-alter']['weight'] if checkinmethods('Conn-alter/weight') else 0.1e-2
			alter = list(range(len(connections)))
			for i in range(n_alter):
				Xalter = alter[np.random.randint(len(alter))]
				connections[Xalter][0].weight[0] = w_alter
				connections[Xalter][0].delay     = d_alter
				print(" > %03d -> %03d                                 : were altered"%(connections[Xalter][1],connections[Xalter][2]))
				alter.remove(Xalter)
			print("Total number of connections                   :",len(connections))
			print("Number of altered connections                 :",n_alter)
			print("Procentage                                    :",n_alter*100/len(connections))
			print("================================================\n")
				
		if checkinmethods('Conn-rec'):
			methods['Conn-rec-results'] = [ (n[1],n[2],n[0].weight[0],n[0].delay) for n in connections ]
		if checkinmethods('Conn-stat'):
			print("================================================")
			print("===           Connections Statistics         ===")
			statn = np.array( [ float(len(o)) for o in OUTList ] )
			meann,stdrn = np.mean(statn),np.std(statn)
			minin,maxin = np.min(statn), np.max(statn)
			statw = np.array( [ n[0].weight[0] for n in connections ] )
			meanw,stdrw = np.mean(statw),np.std(statw)
			miniw,maxiw = np.min(statw), np.max(statw)
			statd = np.array( [ n[0].delay for n in connections ] )
			meand,stdrd = np.mean(statd),np.std(statd)
			minid,maxid = np.min(statd), np.max(statd)
			methods['Conn-stat-results'] = {
				'ncon': {
					'mean':meann, 'stdr':stdrn, 'min':minin, 'max':maxin
				},
				'weight':{
					'mean':meanw, 'stdr':stdrw, 'min':miniw, 'max':maxiw
				},
				'delay':{
					'mean':meand, 'stdr':stdrd, 'min':minid, 'max':maxid
				}
			}
			print(" > Number min / max / mean / stdev / CV       :",minin,"/",maxin,"/",meann,"/",stdrn,"/",stdrn/meann)
			print(" > Weight min / max / mean / stdev / CV       :",miniw,"/",maxiw,"/",meanw,"/",stdrw,"/",stdrw/meanw)
			print(" > Delay  min / max / mean / stdev / CV       :",minid,"/",maxid,"/",meand,"/",stdrd,"/",stdrd/meand)
			print("================================================\n")
		#DB>>
		#plt.figure(0)
		#w=np.array([c[0].weight[0] for c in connections])
		#print np.mean(w), np.std(w)
		#plt.hist(w,bins=50,range=(0,1e-6))
		#plt.show()
		#exit(0)
		#<<DB
			
	
	#### Create gapjunctions:
	if checkinmethods('gapjunction'):
		if   not 'ncon' in methods['gapjunction']            : GJList = OUTList
		elif methods['gapjunction']['ncon'] is None          : GJList = OUTList
		elif not type(methods['gapjunction']['ncon']) is int : GJList = OUTList
		elif not methods['gapjunction']['ncon'] > 0          : GJList = OUTList
		else:
			GJList = [ [] for x in range(methods['ncell'])]
			gjncon = methods['gapjunction']['ncon']
			print("==================================")
			print("===       Map Gap-junctions     ===")
			print("==================================\n")
			for toid in range(methods['ncell']):
				from_excaption = [ 0 for x in range(methods['ncell']) ]
				from_excaption[toid] = 1
				upcnt = 0
				total = 0
				if type(gjncon) is tuple or type(gjncon) is list:
					if len(gjncon) > 1:
						neurons[toid].g_jcnt = int(np.random.random()*(gjncon[1]-gjncon[0]) + gjncon[0])
					else:
						neurons[toid].g_jcnt = gjncon[0]
				else:
					neurons[toid].g_jcnt = gjncon
				while  upcnt < 10000*methods['ncell']:
					upcnt += 1
					fromid = rnd.randint(0, methods['ncell']-1)
					if from_excaption[fromid] == 1 : continue
					upcnt  = 0
					total += 1
					from_excaption[fromid] = 1
					GJList[toid].append(fromid)
					if total >= neurons[toid].g_jcnt :break
				else:
					sys.stderr.write("Couldn't obey connections conditions\nNeuron:%d TOTLA:%d CURRENT:%d\n"%(toid,gjncon,total))
					for x in GJList:
						sys.stderr.write("ID:%d #%d\n"%(x[0],x[1]))
					sys.exit(1)
		
		print("==================================")
		print("===    Make the Gap-Junctions   ===")
		print("==================================\n")
		gapjuctions = []
		for cellid,gjlst in enumerate(GJList):
			for preidx in gjlst:
				gj0,gj1 = h.gap(0.5, sec=neurons[cellid].soma), h.gap(0.5, sec=neurons[preidx].soma)
				h.setpointer(neurons[preidx].soma(.5)._ref_v, 'vgap', gj0)
				h.setpointer(neurons[cellid].soma(.5)._ref_v, 'vgap', gj1)
				gj0.r, gj1.r = methods['gapjunction']['r'],methods['gapjunction']['r']
				gapjuctions.append( (gj0,gj1,neurons[cellid].soma,neurons[preidx].soma) )

	if checkinmethods('simvar'):
		print("==================================")
		print("===      SIMVAR was found!     ===")
		print("==================================\n")
		simvars = []
		if methods['simvar']['type'] == 'n':
			for n in neurons:
				sv = h.variator(0.5, sec=n.soma)
				exec("h.setpointer(n.soma(0.5)."+methods['simvar']["var"]+", \'var\',sv)")
				simvars.append(sv)
			simvarrec = h.Vector()
			exec("simvarrec.record(neurons[0].soma(0.5)."+methods['simvar']["var"]+")")
		#elif methods['simvar']['type'] == 'c':
			#for n in neurons:
				#sv = h.variator()
				#exec "h.setpointer(n."+methods['simvar']["var"]+", \'var\',sv)"
				#simvars.append(sv)
			#simvarrec = h.Vector()
			#exec "simvarrec.record(neurons[0]."+methods['simvar']["var"]+")"
		elif methods['simvar']['type'] == 'g':
			for g0,g1,s0,s1 in gapjuctions:
				sv = h.variator(0.5, sec=s0)
				#DB>>
				#print "h.setpointer(g0."+methods['simvar']["var"]+", \'var\',sv)"
				#<<DB
				exec("h.setpointer(g0."+methods['simvar']["var"]+", \'var\',sv)")
				simvars.append(sv)
				sv = h.variator(0.5, sec=s1)
				exec("h.setpointer(g1."+methods['simvar']["var"]+", \'var\',sv)")
				simvars.append(sv)
			simvarrec = h.Vector()
			exec("simvarrec.record(gapjuctions[0][0]."+methods['simvar']["var"]+")")
		for sv in simvars:
			sv.a0 = methods['simvar']['a0']
			sv.a1 = methods['simvar']['a1']
			sv.t0 = methods['simvar']['t0']
			sv.t1 = methods['simvar']['t1']
	
	if checkinmethods('external'):
		ex_netstim	= h.NetStim(.5, sec = neurons[0].soma)
		if type(methods['external']) is list:
               #              0      1          2         3     4    5    6    7                8
               #/external=\(Start,interval,spike-count,weight,Esyn,Tau1,Tau2,delay,probability of connections\)
			if len(methods['external']) < 8:
			   methods['external'] = {
					'start'       :methods['external'][0],
					'interval'    :methods['external'][1],
					'count'       :methods['external'][2],
					'weight'      :methods['external'][3],
					'E'           :methods['external'][4],
					'tau1'        :methods['external'][5],
					'tau2'        :methods['external'][6]
			   }
			elif len(methods['external']) < 9:
			   methods['external'] = {
					'start'       :methods['external'][0],
					'interval'    :methods['external'][1],
					'count'       :methods['external'][2],
					'weight'      :methods['external'][3],
					'E'           :methods['external'][4],
					'tau1'        :methods['external'][5],
					'tau2'        :methods['external'][6],
					'delay'       :methods['external'][8]
			   }
			elif len(methods['external']) >= 9:
			   methods['external'] = {
					'start'       :methods['external'][0],
					'interval'    :methods['external'][1],
					'count'       :methods['external'][2],
					'weight'      :methods['external'][3],
					'E'           :methods['external'][4],
					'tau1'        :methods['external'][5],
					'tau2'        :methods['external'][6],
					'delay'       :methods['external'][8],
					'p'           :methods['external'][9]
			   }
			if type(methods['external']['delay']) is list or type(methods['external']['delay']) is tuple:
				if len(methods['external']['delay']) < 2:
					methods['external']['delay'] = methods['external']['delay'][0]
				else:
					methods['external']['delay'] = {
						'mean'  : methods['external']['delay'][0],
						'stdev' : methods['external']['delay'][1]
					}
		if not checkinmethods('external/start'   ): methods['external']['start'   ] = methods["tstop"]/3.
		if not checkinmethods('external/interval'): methods['external']['interval'] = methods["tstop"]/6.
		if not checkinmethods('external/count'   ): methods['external']['count'   ] = 1
		if not checkinmethods('external/E'       ): methods['external']['E'       ] = 0
		if not checkinmethods('external/tau1'    ): methods['external']['tau1'    ] = 0.8
		if not checkinmethods('external/tau2'    ): methods['external']['tau2'    ] = 1.2
		if not checkinmethods('external/weight'  ): methods['external']['weight'  ] = 0.
		if not checkinmethods('external/delay'   ): methods['external']['delay'   ] = 1.
		print("================================================")
		print("===              External Input              ===")
		print(" > Start                                      :", methods['external']['start'])
		print(" > Interval                                   :", methods['external']['interval'])
		print(" > Count                                      :", methods['external']['count'])
		if  checkinmethods('external/p'):
			print(" > P                                          :", methods['external']['p'])
		print(" > Reversal potential                         :", methods['external']['E'])
		print(" > Tau 1                                      :", methods['external']['tau1'])
		print(" > Tau 2                                      :", methods['external']['tau2'])
		print(" > Weight                                     :", methods['external']['weight'])
		if checkinmethods('external/delay/mean') or checkinmethods('external/delay/stdev'):
			print(" > Delay                                    ")
			if checkinmethods('external/delay/mean'):
				print("   > mean                                     :", methods['external']['delay']['mean'])
			if checkinmethods('external/delay/stdev'):
				print("   > stdev                                    :", methods['external']['delay']['stdev'])
		else:
			print(" > Delay                                      :", methods['external']['delay'])
		print("================================================\n")
		ex_netstim.start	= methods['external']['start'   ]
		ex_netstim.noise	= 0
		ex_netstim.interval	= methods['external']['interval'] 
		ex_netstim.number	= methods['external']['count']
		ex_syn,ex_netcon = [],[]
		for n in neurons:
			if  checkinmethods('external/p'):
				if rnd.random() > methods['external']['p']: continue
			ex_syn_new = h.Exp2Syn(0.5, sec=n.soma)
			ex_syn_new.e	= methods['external']['E'   ]
			ex_syn_new.tau1	= methods['external']['tau1'] if checkinmethods('external/tau1') else 0.8
			ex_syn_new.tau2	= methods['external']['tau2'] if checkinmethods('external/tau2') else 1.2
			ex_syn.append(ex_syn_new)
			if checkinmethods('external/delay/mean') and checkinmethods('external/delay/stdev'):
				exdelay = -1.0
				while exdelay <= 0.0 : exdelay = methods['external']['delay']['mean']+np.random.randn()*methods['external']['delay']['stdev']
			elif checkinmethods('external/delay/mean'):
				exdelay = methods['external']['delay']['mean']
			else :
				exdelay = methods['external']['delay']
			ex_netcon_new	= h.NetCon(ex_netstim, ex_syn_new, 1,exdelay , methods['external']['weight'], sec = n.soma)
			ex_netcon.append(ex_netcon_new)

	if checkinmethods("wmod"):
		print("==================================")
		print("===      Weight Modulator      ===")
		if not checkinmethods("wmod/scale"          ):methods["wmod"]["scale"        ] = 2.
		if not checkinmethods("wmod/time-points"    ):methods["wmod"]["time-points"  ] = [0., methods['tstop']]
		if not checkinmethods("wmod/weight-points"  ):methods["wmod"]["weight-points"] = [ methods["synapse"]["weight"], methods["wmod"]["scale"]*methods['synapse']["weight"] ]
		print(" > Scale                        : ",methods["wmod"]["scale"        ])
		print(" > Time points                  : ",methods["wmod"]["time-points"  ])
		print(" > Weight points                : ",methods["wmod"]["weight-points"])
		wmodT, wmodW = h.Vector(), h.Vector()
		wmodT.from_python(methods["wmod"]["time-points"  ])
		wmodW.from_python(methods["wmod"]["weight-points"  ])
		for c,pre,post in connections:
			wmodW.play(c._ref_weight[0],wmodT,1)
		print("==================================\n")
			

	if checkinmethods("imod") and checkinmethods("neuron/Iapp"):
		print("==================================")
		print("===     Current Modulator      ===")
		if not checkinmethods("imod/scale"          ):methods["imod"]["scale"         ] = 2.
		if not checkinmethods("imod/time-points"    ):methods["imod"]["time-points"   ] = [0.                           , methods['tstop']]
		if not checkinmethods("imod/current-points" ):methods["imod"]["current-points"] = [-1.*methods["neuron"]["Iapp"], -1.*methods["imod"]["scale"]*methods["neuron"]["Iapp"] ]
		print(" > Time Points                  : ",methods["imod"]["time-points"   ])
		print(" > Current Points               : ",methods["imod"]["current-points"])
		imodAll = []
		
		for idx, n in enumerate(neurons):
			imodt, imodw = h.Vector(), h.Vector()
			imodt.from_python(methods["imod"]["time-points"     ])
			imodw.from_python(methods["imod"]["current-points"  ])
			imodw.play(n.innp._ref_mean,imodt,1)
			imodAll.append( (imodt,imodw) )
			#++++
		print("==================================\n")


	##DB>>
	#for n in neurons:
		#print n.isyn.e
	#exit(0)
	##<<DB

	print("==================================")
	print("===           RUN              ===")
	npc = h.ParallelContext()
	if checkinmethods("ncon"):
		mindel = np.array([ x[0].delay for x in connections ] )
		mindel = np.min(mindel)
		if mindel > h.dt*2:
			if type(methods['corefunc']) is int:
				npc.nthread(methods['corefunc'])
				sys.stderr.write( " > Setup                            : %g core\n"%(methods['corefunc']) )
				print(" > Setup                        : %g core"%(methods['corefunc'])) 
			else:
				#### Setup parallel context if there are delays
				if not os.path.exists("/etc/beowulf") and os.path.exists("/sysini/bin/busybox"):
					#I'm not on head node. I can use all cores (^-^)
					methods['corefunc'] = methods['corefunc'][2]
					npc.nthread(methods['corefunc'])
					sys.stderr.write( " > Setup                        : %g core\n"%(methods['corefunc']) )
					print(" > Setup                        : %g core"%(methods['corefunc'])) 
				elif os.path.exists("/etc/beowulf"):
					#I'm on head node. I grub only half (*_*)
					methods['corefunc'] = methods['corefunc'][1]
					npc.nthread(methods['corefunc'])
					sys.stderr.write( " > Setup                        : %g core\n"%(methods['corefunc']) )
					print(" > Setup                        : %g core"%(methods['corefunc'])) 
				else:
					#I'm on Desktop (-.-)
					methods['corefunc'] = methods['corefunc'][0]
					npc.nthread(methods['corefunc'])
					sys.stderr.write( " > Setup                        : %g cores\n"%(methods['corefunc']) )
					print(" > Setup                        : %g core"%(methods['corefunc'])) 
		else:
			#I'm in (_!_)
			methods['corefunc'] = 0
			#npc.nthread(methods['corefunc'])
			sys.stderr.write( " > Setup                        : %g cores\n"%(methods['corefunc']) )
			print(" > Setup                        : %g core"%(methods['corefunc'])) 

	if checkinmethods("cvode"):
		cvode = h.CVode()
		cvode.active(1)
		print(" > CVODE                    : ON")
		
	h.finitialize()
	h.fcurrent()
	h.frecord_init()
		
	while h.t < methods['tstop']:h.fadvance()

	print("==================================\n")
		
	

	print("==================================")
	print("===          Analysis          ===")
	print("==================================\n")
	
	t = np.array(t)
	if checkinmethods('gui'):
		plt.rc('text', usetex = True )
		plt.rc('font', family = 'serif')
		plt.rc('svg', fonttype = 'none')
		mainfig = plt.figure(1)
		if checkinmethods("MainFigTitle"):
			mainfig.suptitle(methods["MainFigTitle"])
		if checkinmethods('traceView'):
			cid = mainfig.canvas.mpl_connect('button_press_event', onclick1)
		nplot = 411
		if checkinmethods('simvar')      : nplot += 100
		if checkinmethods('spectrogram') : nplot += 100
		p = plt.subplot(nplot)
			
		tprin=np.array(t)
		tprin = tprin[ np.where( tprin < methods['tv'][1] ) ]
		if methods['tv'][0] <= 0.:
			tproc = 0
		else:
			tproc = tprin[ np.where( tprin < methods['tv'][0] ) ]
			tproc = tproc.shape[0]
		tprin = tprin[tproc:]
		vindex = int((methods["ncell"]-1)/2)
		##PLOT VOLTAGE>>
		vtrace, = plt.plot(tprin,np.array(neurons[vindex].volt)[tproc:tprin.size+tproc],"k")
		plt.ylim(ymax=40.)
		mainfig.canvas.mpl_connect('key_press_event',neuronsoverview)
		plt.ylabel("Voltage (mV)", fontsize=16)
		if methods["external"]:
			ex0 = methods["external"]['start']
			ex1 = methods["external"]['interval']
			for ex2 in range(methods["external"]['count']):
				plt.plot([ex0+ex1*ex2,ex0+ex1*ex2],[0,30],"r--")
		plt.subplot(nplot+1,sharex=p)
		nurch = np.arange(1,methods["ncell"]+1,1)
		if checkinmethods('sortbysk'):
			if methods['sortbysk'] == 'I':
				nindex = [ (-neurons[i].innp.mean,i) for i in range(methods["ncell"])]
				nindex.sort()
				#DB>>
				#print nindex
				#<<DB
				for i in range(methods["ncell"]):
					nurch[nindex[i][1]]=i
				#DB>>
				#for i in xrange(methods["ncell"]):
				#	print i,'->',nurch[i],'=',-neurons[nurch[i]].innp.mean,"|",-neurons[i].innp.mean
				#exit(0)
				#<<DB
			if methods['sortbysk'] == 'N':
				nindex = [ (-neurons[i].innp.stdev,i) for i in range(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in range(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'T':
				nindex = [ (neurons[i].type21,i) for i in range(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in range(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'G':
				nindex = [ (neurons[i].gsynscale,i) for i in range(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in range(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'NC':
				nindex = [ (-neurons[i].concnt,i) for i in range(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in range(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'GT':
				nindex = [ (neurons[i].gtotal,i) for i in range(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in range(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'ST':
				nindex = [ (neurons[i].tsynscale,i) for i in range(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in range(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'FR':
				nindex = [ (neurons[i].spks.size(),i) for i in range(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in range(methods["ncell"]):
					nurch[nindex[i][1]]=i
				
			#print nurch
	pmean, fmean = 0., 0
	pcnt,  fcnt  = 0 , 0
	nrnfr        = []  


	meancur = np.zeros(t.size)
	if checkinmethods('spectrogram'):
		populationcurrent = np.zeros(t.size)
	spbins  = np.zeros( int(np.floor(methods['tstop']))+1 )
	specX	= np.arange(spbins.size, dtype=float)
	specX	*= 1000.0/methods['tstop']
	#pnum	= specX.size/2
	pnum 	= int(200.*methods['tstop']/1000.0)
	specX	= specX[:pnum]
	if checkinmethods("nrnFFT"):
		specN	= np.zeros(pnum)
#	specV	= np.zeros(t.size())

	#NEED ISI for indices 
	if checkinmethods('N2NHI-netISI') or checkinmethods('N2NHI'):
		cg_nrnisi = []

	if 10 < methods["nrnISI"] <= 3000:
		isi		= np.zeros(methods["nrnISI"])
	if checkinmethods('coreindex'):
		coreindex = [0.0, 0.0]

	if checkinmethods('gui'):
		rast = []
		#xrast = np.array([])
		#yrast = np.array([])
	if checkinmethods('jitter-rec'):
		jallspikes = np.zeros(int((methods['tstop']-methods['cliptrn'])/methods['timestep'])\
		                       if checkinmethods('cliptrn') else \
		                      int(methods['tstop']/methods['timestep']) ) 
	
	analdur = (methods["tstop"]-methods['cliptrn']) if checkinmethods('cliptrn') else methods["tstop"]
 
	#for (idx,n) in map(None,xrange(methods["ncell"]),neurons):
	for idx,n in enumerate(neurons):
		n.spks = np.array(n.spks)
		if checkinmethods('gui'):
			if not methods['cliprst']:
				rast += [ (st,nurch[idx]) for st in n.spks if methods['tv'][0] < st < methods['tv'][1] ]
			elif nurch[idx]%methods['cliprst'] == 0:
				rast += [ (st,nurch[idx]) for st in n.spks if methods['tv'][0] < st < methods['tv'][1] ]
			#spk = n.spks[ np.where (n.spks < methods['tv'][1]) ]
			#if methods['tv'][0] > 0:
				#spk = spk[ np.where (spk > methods['tv'][0]) ]
			
				#xrast = np.append(xrast,spk)
				#yrast = np.append(yrast,np.repeat(,spk.size))
			##elif idx%methods['cliprst'] == 0:
			#elif nurch[idx]%methods['cliprst'] == 0:
				#xrast = np.append(xrast,spk)
				#yrast = np.append(yrast,np.repeat(nurch[idx],spk.size))
		
		if checkinmethods('cliptrn'):
			fstidx = np.where(n.spks > methods['cliptrn'] )[0]
			if len(fstidx) < 2:
				aisi = None
			else:
				fstidx = fstidx[0]
				aisi = n.spks[fstidx+1:] - n.spks[fstidx:-1]
		else:
			aisi = n.spks[1:] - n.spks[:-1]
		if checkinmethods('coreindex'):
			coreindex[0] += np.sum((aisi[1:] - aisi[:-1])/aisi[:-1])
			coreindex[1] += aisi.size - 1
		if 10 < methods["nrnISI"] <= 3000:
			for i in aisi[ np.where(aisi < methods["nrnISI"]) ]:
				isi[ int(np.floor(i)) ] += 1.0
		if not aisi is None  and( checkinmethods('N2NHI-netISI') or checkinmethods('N2NHI') ):
			cg_nrnisi += aisi.tolist()
		if not aisi is None:
			pmean += np.sum(aisi)
			pcnt  += aisi.shape[0]
			fr    = ( float(n.spks.shape[0] - fstidx)*1000./analdur ) if checkinmethods('cliptrn') else float(n.spks.shape[0])*1000./analdur
			fmean += fr
			fcnt  += 1
			nrnfr.append(fr)
		if checkinmethods('gui'):
			if methods['tracetail'] == 'total current' or methods['tracetail'] == 'mean total current' or methods['tracetail'] == 'TI' or methods['tracetail'] == 'MTI':
				meancur += np.array(n.isyni.x) + np.array(n.inoise.x)
			elif methods['tracetail'] == 'total synaptic current' or methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'TSI' or methods['tracetail'] == 'MTSI':
				meancur += np.array(n.isyni.x)
			elif methods['tracetail'] == 'total conductance' or methods['tracetail'] == 'mean total conductance' or methods['tracetail'] == 'TG' or methods['tracetail'] == 'MTG':
				meancur += np.array(n.isyng.x)
			
		if checkinmethods('spectrogram'):
			populationcurrent += np.array(n.isyni.x) + np.array(n.inoise.x)
			
		spn	= np.zeros(spbins.size)
		for sp in n.spks:
			spbins[ int( np.floor(sp) ) ] +=1
			spn[ int( np.floor(sp) ) ] +=1
		
			if checkinmethods('jitter-rec'):
				jps = int( (sp - methods['cliptrn'])/methods["timestep"] ) if checkinmethods('cliptrn') else int( sp/methods["timestep"] )
				jallspikes[jps] += 1
				
				
		if checkinmethods('cliptrn'):
			spn = spn[methods['cliptrn']:]
		if checkinmethods("nrnFFT"):
			fft = np.abs( np.fft.fft(spn)*1.0/methods['tstop'] )**2
			specN += fft[:pnum]
	
	methods["nrnPmean"] = None if pcnt < 1 else float(pmean)/float(pcnt)
	methods["nrnFmean"] = None if fcnt < 1 else float(fmean)/float(fcnt)
	methods["nrnfr"]    = nrnfr[:]
	print("==================================")
	print("===           Neurons          ===")
	print(" > mean Period      (ms)        :",methods["nrnPmean"])
	print(" > mean Firing Rate (Hz)        :",methods["nrnFmean"])
	print("==================================\n")

	
	if checkinmethods('jitter-rec') and checkinmethods('cliptrn'):
		jallspikes = jallspikes[ np.where( jallspikes > methods['cliptrn'] ) ]
		
	if checkinmethods('gui'):
		if not checkinmethods('rstmark'    ):methods['rstmark'    ]="."
		if not checkinmethods('rstmarksize'):methods['rstmarksize']=5
		#PLOT RASTER>>
		#plt.plot(xrast,yrast,"k"+methods['rstmark'],mew=0.75,ms=methods['rstmarksize'])#,ms=10)
		rast = np.array(rast)
		plt.plot(rast[:,0],rast[:,1],"k"+methods['rstmark'],mew=0.75,ms=methods['rstmarksize'])#,ms=10)
		
		if methods['fullrast']	: plt.ylim(ymin=0,ymax=methods["ncell"])
		else			: plt.ylim(ymin=0)
	if checkinmethods("nrnFFT"):
		specN /= float(methods["ncell"])#	specV /= float(methods["ncell"])
		methods["nrnFFT-results"] = { 'sectrum':specN[:pnum], 'freq':specX }
	
	if checkinmethods('cliptrn'):
		spbins = spbins[methods['cliptrn']:]
	
	if checkinmethods('popfr'):
		popfr = np.mean(spbins)
		print("==================================")
		print("===       MEAN FIRING RATE     ===")
		print("  > MFR =           ",popfr)
		print("==================================\n")
		methods['popfr-results'] = popfr

	if checkinmethods("netFFT") or checkinmethods("nrnFFT"):
		print("==================================")
		print("===            FFT             ===")
		print("==================================\n")
		fft = np.abs( np.fft.fft(spbins)*1.0/methods['tstop'] )**2
		methods["netFFT-results"] = { 'sectrum':fft[:pnum], 'freq':specX }

	##EN
	#probscale = np.zeros(methods["ncell"] + 1)
	#probscale[0] = 1./float(methods["ncell"] + 1)
	#for x in range(1,methods["ncell"] + 1):
		#probscale[x] = probscale[0]*probscale[x-1]
	#pspbin = np.array([ probscale[int(x)] for x in  spbins] )
	#en = np.sum( (-1)*pspbin*np.log(pspbin) )
	
	if checkinmethods('coreindex'):
		coreindex = coreindex[0]/coreindex[1]
		print(coreindex, 1./(1.+ abs(coreindex)))
		sys.exit(0)
		#with open("coreindex.csv","w") as fd:
			#for i in coreindex: fd.write("%g\n"%i)
		#coreindex = np.corrcoef(coreindex[:-1],y=coreindex[1:])[0][1]
	
	#external stimulation index
	if checkinmethods('external') and checkinmethods('extprop'):
		print("==================================")
		print("===      Spike Probability     ===")
		spprop = 0
		for etx in range(methods['external']['count']):
			lidx = int( np.floor(methods['external']['start']+methods['external']['interval']*etx) )
			ridx = int( np.floor(lidx + methods['external']['interval']*methods['extprop']) )
			spprop += float( np.sum(spbins[lidx:ridx]) )
		spprop /= methods['external']['count']*methods["ncell"]
		methods['extprop-results'] = spprop
		print(" > Spike group probability      :",spprop)
		print("==================================\n")
			

	if checkinmethods("peakDetec") or checkinmethods("R2") or checkinmethods("SAC") or checkinmethods('N2NHI') or checkinmethods('N2NHI-netISI') or checkinmethods('spike2net-dist'):
		print("==================================")
		print("===         Peak Detector      ===")
		print("==================================\n")
		kernel = np.arange(-methods["gkernel"][1],methods["gkernel"][1],1.)
		kernel = np.exp(kernel**2/methods["gkernel"][0]/(-methods["gkernel"][0]))
		module = np.convolve(spbins,kernel)
		module = module[int(kernel.size/2):1-int(kernel.size/2)]
		#spbinmax = (np.diff(np.sign(np.diff(module))) < 0).nonzero()[0] + 1
		#spbinmin = (np.diff(np.sign(np.diff(module))) > 0).nonzero()[0] + 1
		spbinmark = []
		for idx in (np.diff(np.sign(np.diff(module))) < 0).nonzero()[0] + 1:
			spbinmark.append([idx,1])
		for idx in (np.diff(np.sign(np.diff(module))) > 0).nonzero()[0] + 1:
			spbinmark.append([idx,-1])
		peakmark  = []
		spc,ccnt = 0.,0.
		if checkinmethods("SPC-std"):
			spccun = []
		if len(spbinmark) > 2:
			spbinmark.sort()
			spbinmark = np.array(spbinmark)
			for mx in np.where( spbinmark[:,1] > 0 )[0]:
				if mx <= 2 or mx >= (spbinmark.shape[0] -2):continue
				if spbinmark[mx-1][1] > 0 or spbinmark[mx+1][1] > 0 or spbinmark[mx][1] < 0:continue
				peakmark.append(spbinmark[mx])
				ccnt    += 1
				curespc = np.sum(spbins[spbinmark[mx-1][0]:spbinmark[mx+1][0]])
				if checkinmethods("SPC-std"):spccun.append(curespc)
				spc += curespc
		else:
			spbinmark = None
		if ccnt > 0:
			spc /= ccnt
		
	
	if checkinmethods("SAC") and not spbinmark is None:
		print("==================================")
		print("===    Spikes Autocorrelation  ===")
		#sac_v_max = len( peakmark )- 2
		#sac_vector = np.zeros( (methods['ncell'],sac_v_max) )
		#sac_peaks  = [ (lp[0],mp[0],rp[0]) for lp,mp,rp in zip(peakmark[:-2],peakmark[1:-1],peakmark[2:]) ]
		#for pi,(lp,mp,rp) in enumerate(sac_peaks):
			#Pl = (lp - mp)/2 + mp
			#Pr = (rp - mp)/2 + mp
			#for ni, n in enumerate(neurons):
				#sac_vector[ni,pi] = 0 if len( np.where( (n.spks>Pl)*(n.spks<Pr) )[0] ) == 0 else 1
				##sac_vector[ni,pi] = -1 if len( np.where( (n.spks>Pl)*(n.spks<Pr) )[0] ) == 0 else 1
		#sac_c_max = methods["SAC"] if type(methods["SAC"]) is int else sac_v_max/2-1
		#sac_events = []
		#for sac_ac_idx in range(1,sac_c_max):
			#sac_ac = np.sum(sac_vector[:,:-sac_ac_idx]*sac_vector[:,sac_ac_idx:],axis=1)
			#sac_events.append( (float(np.amax(sac_ac))/float(sac_vector.shape[1]-sac_ac_idx-1),sac_ac_idx,np.where(sac_ac==int(np.amax(sac_ac)) )[0] ) )
		#sac_events.sort()
		#for sac_ev in  sac_events:
			#print sac_ev
			
			
		##DB>>
		#np.savetxt("sac_ac.txt",sac_vector, fmt='%-1d')
		#with open("sac_ev.txt","w") as fd:
			#for sac_ev in  sac_events:
				#fd.write("{}\n".format(sac_ev) )
			
		##exit(0)
		##<<DB 
					
		print("==================================\n")
	if methods['tracetail'] == 'R2':
		costR2p = float( methods['cont-R2'] )
		contR2t = np.arange(0,costR2p,1.)
		sinkern = np.sin(np.pi*2.*contR2t/costR2p)
		coskern = np.cos(np.pi*2.*contR2t/costR2p)
		cntkern = np.ones(contR2t.shape)
		sinkern = np.convolve(spbins,sinkern)
		coskern = np.convolve(spbins,coskern)
		cntkern = np.convolve(spbins,cntkern)
		contR2  = (coskern/cntkern)**2+(sinkern/cntkern)**2
		contR2  = contR2[contR2t.shape[0]/2:1-contR2t.shape[0]/2]
		contSPC = cntkern[contR2t.shape[0]/2:1-contR2t.shape[0]/2]/methods["ncell"]
		if checkinmethods("cont-R2-smooth"):
			if type(methods["cont-R2-smooth"]) is float:
				costR2p = methods["cont-R2-smooth"]
			else:
				costR2p *= 3.
			contR2t = np.arange(0,costR2p*2,1.)
			contR2t = np.exp(-((contR2t-costR2p)/costR2p)**2)
			contR2  = np.convolve(contR2, contR2t)/np.sum(contR2t)
			contR2  = contR2[contR2t.shape[0]/2:1-contR2t.shape[0]/2]
			contSPC = np.convolve(contSPC, contR2t)/np.sum(contR2t)
			contSPC = contSPC[contR2t.shape[0]/2:1-contR2t.shape[0]/2]
		print("==================================")
		print("===        Continues R2        ===")
		print(" > Preiod (Frequency)            :",methods['cont-R2'],"(ms) /",1000./methods['cont-R2'],"(Hz)")
		if checkinmethods("cont-R2-smooth"):
			print(" > Smooth kernel                 :",costR2p,"(ms)")
		print("==================================\n")
		
		
	if checkinmethods('jitter-rec'):
		print("==================================")
		print("===       Jitter Detector      ===")
		print("==================================\n")
		jkernel = np.arange(-methods["gkernel"][1],methods["gkernel"][1],methods['timestep'])
		jkernel = np.exp(jkernel**2/methods["gkernel"][0]/(-methods["gkernel"][0]))
		jmodule = np.convolve(jallspikes,jkernel)
		jmodule = jmodule[jkernel.size/2:1-jkernel.size/2]
		jpeaks = []
		#for idx in (np.diff(np.sign(np.diff(jmodule))) < 0).nonzero()[0] + 1:
			#jpeaks.append(float(idx))
		#for il,ic,ir in zip(jpeaks[:-2],jpeaks[1:-1],jpeaks[2:]):
			#for ik in jmodule[(il+ic)/2:ic]:
			#???*methods['timestep']

	##R2
	##Per
	if checkinmethods("R2"):
		methods["R2-results"] = {}
		if ccnt > 0:
			methods["R2-results"]['spc'] = spc
		else:
			methods["R2-results"]['spc'] = None
		print("==================================")
		print("===             R2             ===")
		X,Y,Rcnt,netpermean,netpercnt,netfrqmean=0.,0.,0.,0.0,0.0,0.0
		phydist  = []
		if not spbinmark is None:
			for mx in np.where( spbinmark[:,1] > 0 )[0]:
				#if mx >= (spbinmark.shape[0]/2 - 3):continue
				if mx >= (spbinmark.shape[0] - 3):continue
				if spbinmark[mx+1][1] > 0 or spbinmark[mx+2][1] < 0 or spbinmark[mx][1] < 0:continue
				Pnet = float(spbinmark[mx+2][0] - spbinmark[mx][0])
				netpermean += Pnet
				if Pnet > 0.:
					netfrqmean += 1000./Pnet
				netpercnt  += 1.
#				for n,i in map(None,spbins[spbinmark[mx][0]:spbinmark[mx+2][0]],xrange(spbinmark[mx+2][0] - spbinmark[mx][0])):
				for i,n in enumerate(spbins[spbinmark[mx][0]:spbinmark[mx+2][0]]):
					phyX = np.cos(np.pi*2.*float(i)/Pnet)
					phyY = np.sin(np.pi*2.*float(i)/Pnet)
					X += n*phyX
					Y += n*phyY
					Rcnt += n
					if methods['sycleprop']:
						#phydist.append( (360.*np.arctan2(phyY,phyX)/2/np.pi,n) )
						#phydist.append( (np.arctan2(phyY,phyX),n) )
						phydist.append( (np.pi*2.*float(i)/Pnet,n) )
		if Rcnt > 0.:
			R2 = (X/Rcnt)**2+(Y/Rcnt)**2
			methods["R2-results"]["R2"] = R2
		else:
			methods["R2-results"]["R2"] = None
		if netpercnt > 1.:
			netpermean /= ( netpercnt - 1)
			methods["R2-results"]["netPmean"] = netpermean
			netfrqmean /= ( netpercnt - 1)
			methods["R2-results"]["netFmean"] = netfrqmean
		else:
			methods["R2-results"]["netPmean"] = None
			methods["R2-results"]["netFmean"] = None
		if not(methods["R2-results"]["netFmean"] is None or methods["nrnFmean"] is None):
			nrnfr = np.array(nrnfr)
			methods["R2-results"]["mean_Fr/Fnet"] = np.mean(nrnfr/methods["R2-results"]["netFmean"])
			methods["R2-results"]["stdr_Fr/Fnet"] = np.std(nrnfr/methods["R2-results"]["netFmean"])
		else:
			methods["R2-results"]["mean_Fr/Fnet"] = methods["R2-results"]["stdr_Fr/Fnet"] = None
		print("  > R2       =           ",methods["R2-results"]["R2"])
		print("  > SPC      =           ",methods["R2-results"]['spc'])
		print("  > netPmean =           ",methods["R2-results"]["netPmean"])
		print("  > netFmean =           ",methods["R2-results"]["netFmean"])
		if not(methods["R2-results"]["netFmean"] is None or methods["nrnFmean"] is None):
			print("  > Fsr/Fnet =           ",methods["R2-results"]["mean_Fr/Fnet"], "+-",methods["R2-results"]["stdr_Fr/Fnet"])
		print("==================================\n")
		if checkinmethods('sycleprop'):
			phydist = np.array(phydist)
			phydist[:,1] /= np.sum(phydist[:,1])
			phyhist,phyhistbins = np.histogram(phydist[:,0], bins=37, weights=phydist[:,1],range=(-np.pi/36,2.*np.pi+np.pi/36))
			methods['sycleprop-results'] = { 'histogram':phyhist, 'bins-bounders':phyhistbins }


	if 10 < methods["netISI"] < 3000 or checkinmethods('N2NHI-netISI'):
		print("==================================")
		print("===          NET ISI           ===")
		print("==================================\n")
		#netisi	= np.zeros(methods["netISI"])
		#lock = threading.RLock()
		#def calcnetisi(ns):
			#global netisi, lock
			#scans	= np.zeros(methods["ncell"],dtype=int)
			#localnetisi = np.zeros(methods["netISI"])
			#for n in ns:
				#for sp in n.spks:
					#for (idx,m) in map(None,xrange(methods["ncell"]),neurons):
						#if m.spks.size < 2 : continue
						#while m.spks[scans[idx]] <= sp and scans[idx] < m.spks.size - 1 : scans[idx] += 1
						#if m.spks[scans[idx]] <= sp : continue
						#if m == n and m.spks[scans[idx]] - sp < 1e-6 : continue
						#aisi = m.spks[scans[idx]] - sp
						#if int(round(aisi)) >= methods["netISI"] : continue
						#localnetisi[ int(round(aisi)) ] += 1
			#with lock:
				#netisi += localnetisi
		#pids = [ threading.Thread(target=calcnetisi, args=(neurons[x::methods['corefunc']],)) for x in xrange(methods['corefunc']) ]
		#for pidx in pids:
			#pidx.start()
			##print pidx, "starts"
		#for pidx in pids:
			#pidx.join()
			##print pidx,"finishs"
		#methods['netISI-results'] = netisi
		netisi   = []
		netsp    = [ np.array(n.spks) for n in neurons ]
		#lock = threading.RLock()
		#def calcnetisi(ns):
			#global netisi, lock
			#localnetisi = [ np.amin(x-netsp[n][np.where(netsp[n] <= x)],initial=100000) for n in xrange(methods['ncell']) for nisi in ns for x in nisi  ]
			#with lock:
				#netisi += localnetisi
		#pids = [ threading.Thread(target=calcnetisi, args=(netsp[x::methods['corefunc']],)) for x in xrange(methods['corefunc']) ]
		#for pidx in pids: pidx.start()
		#for pidx in pids: pidx.join()
		netisi   = np.array([ np.amin(x-netsp[n][np.where(netsp[n] <= x)],initial=100000) for n in range(methods['ncell']) for nisi in netsp for x in nisi  ])
		if checkinmethods('N2NHI-netISI'):
			cg_netisi,_ = np.histogram(np.array(netisi), range=(0,300), bins=100)
			cg_netisi   = cg_netisi.astype(float)
			##DB>>
			#print "CG NetISI:",cg_netisi
			#print "CG NetBIN:",_
			##<<DB
		netisi,_ = np.histogram(np.array(netisi), range=(0,methods["netISI"]), bins=int(round(methods["netISI"]/1.) ))
		netisi   = netisi.astype(float)
		methods['netISI-results'] = netisi

	if checkinmethods('N2NHI') or checkinmethods('N2NHI-netISI'):
		def peakdetector(isi,bins):
			#kernel = np.arange(-25,25,1.)
			#kernel = np.exp(kernel**2/(-9))
			#module = np.convolve(isi,kernel)
			#module = module[kernel.size/2:1-kernel.size/2]
			#return [
				#idx for idx in (np.diff(np.sign(np.diff(module))) < 0).nonzero()[0] + 1
			#]
			return [
				idx for idx in (np.diff(np.sign(np.diff(isi))) < 0).nonzero()[0] + 1
			]
		
		cg_nrnisi,x = np.histogram(np.array(cg_nrnisi), range=(0,300), bins=100)
		cg_nrnisi   = cg_nrnisi.astype(float)
		cg_nrnbin   = (x[1:]+x[:-1])/2.
		##DB>>
		#print "CG NrnISI:",cg_nrnisi
		#print "CG NrnBIN:",_
		##<<DB
		if checkinmethods('N2NHI'):
			print("==================================")
			print("===  Carmen's Cluster Indices  ===")
			Pnet = methods["R2-results"]["netPmean"]
			if Pnet is None:
				ccc_clsidx = [ None for clsidx in range(10) ]
			else:
				ccc_clsidx = [ int(round(Pnet*(clsidx+1))/3) for clsidx in range(10) if int(round(Pnet*(clsidx+1)))/3+1 < cg_nrnisi.shape[0] ]
				ccc_clsidx = [ sum(cg_nrnisi[clsidx-1:clsidx+1])/sum(cg_nrnisi) for clsidx in ccc_clsidx ]
			methods['N2NHI']=ccc_clsidx
			for clsidx,clsize in enumerate(ccc_clsidx):
				print(" > Harmonic %02d                    :"%clsidx,clsize)
			print("==================================\n")
		if checkinmethods('N2NHI-netISI'):
			clidx = peakdetector(cg_netisi,None)
			print("==================================")
			print("===    RTH's Cluster Indices   ===")
			rth_clsidx = [ cg_nrnisi[xidx]/sum(cg_nrnisi) for xidx in clidx if xidx < cg_nrnisi.shape[0] ]
			methods['N2NHI-netISI'] = rth_clsidx
			for clsidx,clsize in enumerate(rth_clsidx):
				print(" > Harmonic %02d                    :"%clsidx,clsize)
			print("==================================\n")

		#methods["netISI"] = 300
		#methods["nrnISI"] = 300
	#if 
		#methods["nrnISI"] = 300
		
	if checkinmethods('spike2net-dist'):
		if not spbinmark is None:
			piskd = np.zeros(200)
			for mx in np.where( spbinmark[:,1] > 0 )[0]:
				if mx >= (spbinmark.shape[0] - 3):continue
				#if spbinmark[mx+1][1] > 0 or spbinmark[mx+2][1] < 0 or spbinmark[mx][1] < 0:continue
				LPnet = float(spbinmark[mx  ][0] - spbinmark[mx-2][0])
				RPnet = float(spbinmark[mx+2][0] - spbinmark[mx  ][0])
				for i,n in enumerate( spbins[spbinmark[mx-2][0]:spbinmark[mx][0]] ):
					binID = int( round( float(i)*100./LPnet ) )
					piskd[binID] +=  n
				for i,n in enumerate( spbins[spbinmark[mx][0]:spbinmark[mx+2][0]] ):
					binID = int( round( float(i)*100./RPnet+100 ) )
					if binID>=200: continue
					piskd[binID] +=  n
			methods['spike2net-dist-result'] = piskd.tolist()
		else:
			methods['spike2net-dist-result'] = None
		
	if checkinmethods("T&S") or checkinmethods('lastspktrg'):
		print("==================================")
		print("===           T & S            ===")
		print("==================================\n")
		allspikes,activeneurons = [],0.
		for n in neurons:
			allspikes += list(n.spks)
			if n.spks.size !=0 :activeneurons += 1.
		allspikes.sort()
		allspikes = np.array(allspikes)
		TaSisi = allspikes[1:]-allspikes[:-1]
		if checkinmethods('lastspktrg'):
			lastspktrg = int( np.mean(allspikes) > methods['tstop']/4. )
			methods['lastspktrg-results'] = lastspktrg
		
		del allspikes
		if checkinmethods("T&S"):
			if bool(lastspktrg):
				mean1TaSisi = np.mean(TaSisi)
				TaSindex	= (np.sqrt(np.mean(TaSisi**2) - mean1TaSisi**2)/mean1TaSisi - 1.)/np.sqrt(activeneurons) 
				methods['T&S-results'] = TaSindex
			else:
				methods['T&S-results'] = None
		
	if checkinmethods("Delay-stat"):
		print("==================================")
		print("===     Delays distribution    ===")
		delays = np.array([ x[0].delay for x in connections])
		mdly, sdly,mxdly,Mxdly = np.mean(delays), np.std(delays), np.min(delays), np.max(delays)
		if not type(methods["Delay-stat"]) is tuple:
			methods["Delay-stat"] = (0., 15., 500)
		dlyhist,dlybins = np.histogram(delays, bins=methods["Delay-stat"][2], normed=True, range=methods["Delay-stat"][0:2] )
		dlyhist /= np.sum(dlyhist)
		print("  > Delays mean  =           ",mdly)
		print("  > Delays stdev =           ",sdly)
		print("  > Delays CV    =           ",sdly/mdly)
		print("  > Delays min   =           ",mxdly)
		print("  > Delays max   =           ",Mxdly)
		methods['Delay-stat-results'] = {
			'mean': mdly, 'stdev': sdly, 'min': mxdly, 'max': Mxdly, 'histogram':dlyhist, 'bins-bounders':dlybins
		}
		print("==================================\n")
		
	if checkinmethods('Gtot-dist') :
		print("==================================")
		print("===   G-total  distribution    ===")
		#gsk = [ n.gsynscale for n in neurons ]
		gsk = [ n.gtotal for n in neurons ]
		mgto, sgto,mxgto,Mxgto = np.mean(gsk), np.std(gsk), np.min(gsk), np.max(gsk)
		if not type(methods["Gtot-dist"]) is tuple:
			methods["Gtot-dist"] = (0,0.06,200)
		#gskhist,gskbins = np.histogram(gsk, bins=methods["ncell"]/25, normed=True, range=[0,Mxgto])#/10)#, normed=True)
		gskhist,gskbins = np.histogram(gsk, bins=methods["Gtot-dist"][2], normed=True, range=methods["Gtot-dist"][0:2] )#/10)#, normed=True)
		gskhist /= np.sum(gskhist)
		print("  > Total Syn. Cond mean  =  ",mgto)
		print("  > Total Syn. Cond stdev =  ",sgto)
		print("  > Total Syn. Cond CV    =  ",sgto/mgto)
		print("  > Total Syn. Cond min   =  ",mxgto)
		print("  > Total Syn. Cond max   =  ",Mxgto)
		methods['Gtot-dist-results'] = { 
			'mean': mgto, 'stdev': sgto, 'min': mxgto, 'max': Mxgto,
			'histogram':gskhist, 'bins-bounders':gskbins 
		}
		if checkinmethods('Gtot-rec') :
			methods['Gtot-rec'] = gsk
		else:
			del gsk
		print("==================================\n")
		
	if checkinmethods('Gtot-stat'):
		print("==================================")
		print("===     G-total Statistics     ===")
		agtot = np.array([ n.gtotal/n.concnt for n in neurons ])
		#DB>>
		#print [n.gtotal for n in neurons ]
		#print [n.concnt for n in neurons ]
		#exit(0)
		#<<DB
		mgtot = np.mean(agtot)
		sgtot = np.std(agtot)
		print("  > mean   gtotal norm    =  ",mgtot)
		print("  > stderr gtotal norm    =  ",sgtot)
		print("  > CV     gtotal norm    =  ",sgtot/mgtot)
		print("==================================\n")
		methods['Gtot-stat-results'] = { 'mean':mgtot, 'stdev':sgtot, 'CV':sgtot/mgtot }

	
	if checkinmethods('2cintercon'):
		print("==================================")
		print("===  2 clusters connectivity   ===")
		tims = methods['tstop']*3./4.
		if pcnt != 0:
			halfpnet = pmean/pcnt/2.
			clslst = []
			print("  >  Searching for clusters       ")
			rarr = np.array(neurons[0].spks)
			tims = rarr[ np.where( rarr > tims ) ]
			del rarr
			tims = tims[0] + halfpnet/2.
			print("  >  Time to search              :",tims)
			for idx, n in enumerate(neurons):
				getlest = np.array(n.spks)
				getlest = getlest[ np.where( getlest>tims )]
				if getlest.shape[0] < 1: continue
				if getlest[0] > tims+halfpnet:
					clslst.append(True)
				else:
					clslst.append(False)
			print("  >  Searching for connectivity index")
			WithinA, WithinB, CrossAB, CrossBA = 0, 0, 0, 0
			countA, countB = 0, 0
			fullstat = checkinmethods('2clrs-stat')
			if fullstat:
				within,cross = np.zeros(methods['ncell']),np.zeros(methods['ncell'])
			for idx, cnt in enumerate(OUTList):
				if clslst[idx]:
					#Cluster A
					countA += 1
					for c in cnt:
						if clslst[c]:
							WithinA += 1
							if fullstat : within[idx] += 1
						else:
							CrossAB += 1
							if fullstat : cross[idx] += 1
				else:
					#cluster B
					countB += 1
					for c in cnt:
						if clslst[c]:
							CrossBA += 1
							if fullstat : cross[idx] += 1
						else:
							WithinB += 1
							if fullstat : within[idx] += 1
			print("  >  Cells in the Cluster A      :",countA)
			print("  >  Cells in the Cluster B      :",countB)
			print("  >  Within Cluster A            :",WithinA)
			print("  >  Within Cluster B            :",WithinB)
			print("  >  From A to B                 :",CrossAB)
			print("  >  From A to B                 :",CrossBA)
			print("  >  Total Within Both Clusters  :",WithinA + WithinB)
			print("  >  Total Between Both Clusters :",CrossAB + CrossBA)
			print("  >  Ratio Between to Within     :",float(CrossAB + CrossBA)/float(WithinA + WithinB))
			if fullstat:
				print("  >  FUUL STATISTICS ")
				#print "  >  ",
				#for idx,(win,btwn) in enumerate(zip(within,cross)):
					#print "{}:{}".format(win,btwn),
					#if not bool((idx+1)%6): print "\n  >  ",
				#print "  >  RATIOS "
				#print "  >  ",
				#for idx,(win,btwn) in enumerate(zip(within,cross)):
					#print "{}".format(btwn/win),
					#if not bool((idx+1)%6): print "\n  >  ",
				#print "  >  "
				withinA = within[ np.where( np.array(clslst) ) ]
				crossA = cross[ np.where( np.array(clslst) ) ]
				print("  >  Cluster A within mean,stdev :",np.mean(withinA), np.std(withinA))
				print("  >  Cluster A to B mean,stdev   :",np.mean(crossA), np.std(crossA))
				withinB = within[ np.where(  (1-1*np.array(clslst).astype(int)).astype(bool) )]
				crossB = cross[ np.where(  (1-1*np.array(clslst).astype(int)).astype(bool) )]
				print("  >  Cluster B within mean,stdev :",np.mean(withinB), np.std(withinB))
				print("  >  Cluster B to A mean,stdev   :",np.mean(crossB), np.std(crossB))
			methods['2cintercon-results'] = {
				"cells-in-A":countA, "cells-in-B":countB,
				'connections-in-A':WithinA, 'connections-in-B':WithinB,
				"connections-A2B":CrossAB, "connections-B2A":CrossBA,
				'total in A&B':WithinA + WithinB,
				'total between A&B':CrossAB + CrossBA,
				'total ratio between/in':float(CrossAB + CrossBA)/float(WithinA + WithinB)
			}
		else:
			print("  >  Pnet isn't defined...,       ")
			methods['2cintercon-results'] = None
		print("==================================\n")
	
	if checkinmethods('CtrlISI'):
		print("==================================")
		print("=== Controlled ISI calculation ===")
		if type(methods['CtrlISI']) is not dict:
			methods['CtrlISI'] = {'bin'   : 5.,'max'   : 120.,}
		if not checkinmethods('CtrlISI/bin'): methods['CtrlISI']['bin'] =   5.
		if not checkinmethods('CtrlISI/max'): methods['CtrlISI']['max'] = 120.
		xbin,xmax = methods['CtrlISI']['bin'], methods['CtrlISI']['max']
		CtrINT = np.linspace(0,xmax,xmax/xbin)+xbin/2
		CtrISI = np.zeros(int(round(xmax/xbin)))
		for n in neurons:
			for isi in n.spks[1:]-n.spks[:-1]:
				isiid = int( round(isi/xbin) )
				if isiid < CtrISI.shape[0]:
					CtrISI[isiid] += 1
		CtrISI = np.column_stack((CtrINT,CtrISI))
		methods['CtrlISI']['result'] = CtrISI
	print("==================================\n")
	
	if checkinmethods('T1vsT2/spikerate'):
		T1cnt,T2cnt = 0, 0
		T1spk,T2spk = 0, 0
		for n in neurons:
			if n.type21 == 1:
				T1cnt += 1
				T1spk += n.spks.shape[0]
			elif n.type21 == 2:
				T2cnt += 1
				T2spk += n.spks.shape[0]
		methods['T1vsT2']['spikerate'] = {
			'T1' : float(T1spk)/float(T1cnt)/methods['tstop']*1000. if T1cnt != 0 else 0.,
			'T2' : float(T2spk)/float(T2cnt)/methods['tstop']*1000. if T2cnt != 0 else 0.
		}
		print("==================================")
		print("===   Type I vs. Type2 Rate    ===")
		print(" > Type I  rate                 :",methods['T1vsT2']['spikerate']['T1'],"[Hz]")
		print(" > Type II rate                 :",methods['T1vsT2']['spikerate']['T2'],"[Hz]")
		print("==================================\n")

	if checkinmethods('get-steadystate'):
		if type(methods['get-steadystate']) is str:
			ssthr=+30.
			ssfilename = methods['get-steadystate']
		elif type(methods['get-steadystate']) is float or type(methods['get-steadystate']) is int:
			ssthr = float(methods['get-steadystate'])
			if type(methods["neuron"]["Vinit"]) is str:
				ssfilename =  methods["neuron"]["Vinit"]+"-ss.dat"
			else:
				ssfilename = 'get-steadystate.dat'
		else:
			ssthr=+30.
			ssfilename = 'get-steadystate.dat'
		print("==================================")
		print("===     Write Steady State     ===")
		print("  >  Threshold                   :",ssthr)
		print("  >  Output File                 :",ssfilename)
		ssvec = np.array(neurons[0].volt)
		ssmsk = np.where( (t>methods['tstop']*4./5.)*(ssvec >= ssthr) )[0]
		if ssmsk.shape[0] == 0:
			print("Error Cannot Get Voltage above Threshold!")
			with open(ssfilename,"w") as fd:
				fd.write("None")
				for n in neurons[1:]:
					fd.write(" None")
				fd.write("\n")
		else:
			ssmsk = int(ssmsk[0])
			with open(ssfilename,"w") as fd:
				fd.write("%g"%ssvec[ssmsk])
				for n in neurons[1:]:
					fd.write(" %g"%(np.array(n.volt)[ssmsk]))
				fd.write("\n")
		print("==================================\n")
	#EN
	#p.set_title("Mean individual Period = %g, Sychrony(Entropy) = %g(%g)"%(pmean/pcnt,1./(1.+en),en))
	
	##R2
	if checkinmethods('gui'):
		title = methods["MainFigTitle"] if checkinmethods("MainFigTitle") else ""
		title += "Mean individual Period = %s"%("NONE" if pcnt == 0 else "%g"%(pmean/pcnt))
		if checkinmethods('popfr'):
			title += 'Mean FR =%g'%popfr
		if checkinmethods("R2"):
			if Rcnt > 0 :
				title += r", $R^2$ = %g, Mean network Period = %g, Spike per cycle = %g"%(R2,netpermean,spc)
			else:
				title += ", *Fail to estimate network period*"
		elif checkinmethods("peakDetec"):
			title += ", Spike per cycle = %g. "%(spc)
		if checkinmethods('T&S'):
			title += ", TaS = %g"%TaSindex
		if checkinmethods('lastspktrg'):
			title += ", LST = %g"%lastspktrg
		p.set_title(title)

		
		plt.subplot(nplot+2,sharex=p)
		if checkinmethods('cliptrn'):
			nppoints = np.arange(methods['tv'][0]+methods['cliptrn'],methods['tv'][1],1.0)
			plt.bar(nppoints,spbins[:methods['tv'][1]-methods['cliptrn']],0.5,color="k")
			hight = spbins[:methods['tv'][1]-methods['cliptrn']].max()
			if (checkinmethods("peakDetec") or checkinmethods("R2")) and not spbinmark is None :
				for mark in spbinmark:
					if mark[0]+methods['cliptrn'] < methods['tv'][0] or mark[0]+methods['cliptrn'] > methods['tv'][1]: continue
					if mark[1] > 0:
						plt.plot([mark[0]+methods['cliptrn'],mark[0]+methods['cliptrn']],[0,hight],"r--")
					else:
						plt.plot([mark[0]+methods['cliptrn'],mark[0]+methods['cliptrn']],[0,hight],"b--")
		else:
			nppoints = np.arange(methods['tv'][0],methods['tv'][1],1.0)
			plt.bar(nppoints,spbins[int(methods['tv'][0]):int(methods['tv'][1])],0.5,color="k")
			hight = spbins[int(methods['tv'][0]):int(methods['tv'][1])].max()
			if (checkinmethods("peakDetec") or checkinmethods("R2")) and not spbinmark is None :
				for mark in spbinmark:
					if mark[0] < methods['tv'][0] or mark[0] > methods['tv'][1]: continue
					if mark[1] > 0:
						plt.plot([mark[0],mark[0]],[0,hight],"r--")
					else:
						plt.plot([mark[0],mark[0]],[0,hight],"b--")
#			plt.plot(nppoints,module[methods['tv'][0]:methods['tv'][1]]/np.sum(kernel),"k--")
#			plt.plot(nppoints,module[methods['tv'][0]:methods['tv'][1]],"k--")
			##DB>>
			#print peakmark
			#plt.plot(np.array(peakmark)[:,0],np.array(peakmark)[:,1],"k^",ms=20)
			##<<DB
		plt.ylabel("Rate (ms$^{-1}$)", fontsize=16)

	if checkinmethods('gui'):
		if methods['tracetail'] == 'mean total current' or methods['tracetail'] == 'MTI':
			meancur = meancur / float(-methods["ncell"])
		elif methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'MTSI':
			meancur = meancur / float(-methods["ncell"])
		elif methods['tracetail'] == 'mean total conductance' or methods['tracetail'] == 'MTG':
			meancur = meancur / float(methods["ncell"])
	
	if checkinmethods('gui'):
		
		plt.subplot(nplot+3,sharex=p)
			
		if methods['tracetail'] == "R2":
			plt.ylabel(r"$R^2$/SPC", fontsize=16)
			xvcrv, = plt.plot(np.arange(0,float(contR2.shape[0])),contR2,'k-',lw=2)
			xvcrv, = plt.plot(np.arange(0,float(contSPC.shape[0])),contSPC,"r--",lw=2)
			plt.ylim(0.,1.)
			#DB>>
			#print contR2
			#print contSPC
			#<<DB
		elif methods['tracetail'] == 'total current' or methods['tracetail'] == 'TI' or methods['tracetail'] == 'mean total current' or methods['tracetail'] == 'MTI'\
		  or methods['tracetail'] == 'total synaptic current' or methods['tracetail'] == 'TSI' or methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'MTSI':
			plt.ylabel("Current (nA)", fontsize=16)
			plt.plot(tprin,-meancur[tproc:tprin.size+tproc])
			plt.plot([tprin[0],tprin[-1]],[0.,0.],"k--")
			#plt.plot([tprin[0],tprin[-1]],[-methods["neuron"]["Iapp"],-methods["neuron"]["Iapp"]],"r--")
			#print np.amax(meancur[tproc+tprin.size/2:tprin.size+tproc]),methods["neuron"]["Iapp"]
		elif methods['tracetail'] == 'total conductance' or methods['tracetail'] == 'mean total conductance' or methods['tracetail'] == 'TG' or methods['tracetail'] == 'MTG':
			plt.ylabel("Total Conductance (nS)", fontsize=16)
			plt.plot(tprin,meancur[tproc:tprin.size+tproc]*1e5)
		elif methods['tracetail'] == 'firing rate' and ( methods["peakDetec"] or methods["R2"] ):
			plt.ylabel("Firing Rate (ms$^{-1}$)", fontsize=16)
			tvl,tvr = int( round(methods['tv'][0]) ), int( round(methods['tv'][1]) )
			plt.plot(nppoints,module[tvl:tvr]/np.sum(kernel),"k--")
			hight = np.max(module[tvl:tvr]/np.sum(kernel))
			if not spbinmark is None :
				for mark in spbinmark:
					if mark[0] < methods['tv'][0] or mark[0] > methods['tv'][1]: continue
					if mark[1] > 0:
						plt.plot([mark[0],mark[0]],[0,hight],"k--")
		elif methods['tracetail'] == 'conductance':
			plt.ylabel("Conductance (nS)", fontsize=16)
			xvcrv, = plt.plot(tprin,np.array(neurons[vindex].isyng)[tproc:tprin.size+tproc]*1e5,'k-',lw=2)
		elif methods['tracetail'] == 'current':
			plt.ylabel(r"Current ($\mu$A)", fontsize=16)
			xvcrv, = plt.plot(tprin,np.array(neurons[vindex].isyni)[tproc:tproc+tprin.size]*1e5,'k-',lw=2)
		elif methods['tracetail'] == 'LFP':
			plt.ylabel(r"LFP", fontsize=16)
			xvcrv, = plt.plot(np.arange(0,float(module.shape[0])),module,'k-',lw=2)
		elif methods['tracetail'] == 'p2eLFP':
			lfp=np.zeros(module.shape[0])
			x1,x2=0.,0,
			for i,s in enumerate(module):
				x1 = s+x1-x1/2.
				x2 = s+x2-x2/5.
				lfp[i]=x2-x1
			if checkinmethods("p2eLFP/LPF"):
				from scipy.signal import butter, lfilter, freqz
				nyq = 0.5 * 1000.				#SAMPLING EVERY 1 ms
				normal_cutoff = methods["p2eLFP"]["LPF"] / nyq		#lowpass 100Hz
				b, a = butter(5, normal_cutoff, btype='low', analog=False)
				lfp = lfilter(b, a, lfp)
			lfp = lfp[int(round(methods["tv"][0])):int(round(methods["tv"][1]))]
			#plt.plot(np.arange(lfp.shape[0])+methods["tv"][0],lfp-np.mean(lfp),'k-',lw=2)
			plt.plot(np.arange(lfp.shape[0])+methods["tv"][0],lfp,'k-',lw=2)
			plt.ylim(ymin=0)
			if checkinmethods("p2eLFP_max"):
				plt.ylim(ymax=methods["p2eLFP_max"])
		
		if checkinmethods('simvar'):
			simvarrec = np.array(simvarrec)
			plt.subplot(nplot+4,sharex=p)
			#plt.ylabel(methods['simvar']['var'], fontsize=16)
			plt.ylim(min([methods['simvar']["a0"],methods['simvar']["a1"]]),max([methods['simvar']["a0"],methods['simvar']["a1"]]))
			plt.plot(tprin,simvarrec[tproc:tprin.size+tproc])
		if checkinmethods('spectrogram'):
			plt.subplot(nplot+(5 if checkinmethods('simvar') else 4),sharex=p)
			populationcurrent
			#NFFT = 131072       # the length of the windowing segments
			NFFT = 65535       # the length of the windowing segments
			Fs = int(1000.0/methods['timestep'])  # the sampling frequency
			from scipy.signal import spectrogram
			f, tf, Sxx = spectrogram(populationcurrent, fs=Fs, nperseg=NFFT,noverlap=NFFT*1020/1024,window='hanning')
			Sxx = Sxx[np.where(f<100),:][0]
			f   = f[np.where(f<100)]
			Sxx = Sxx[np.where(f>20),:][0]
			f   = f[np.where(f>20)]
			print("T SHAPE",tf.shape)
			print("T      ",tf)
			print("F SHAPE",f.shape)
			print("F      ",f)
			print("S SHAPE",Sxx.shape)
			print("s      ",Sxx)
			plt.pcolormesh(tf*1e3, f, Sxx)

		plt.xlabel("time (ms)", fontsize=16)


	
	if (checkinmethods("netFFT") or checkinmethods("nrnFFT")) and checkinmethods('gui'):
		plt.figure(2)
		if checkinmethods("netFFT") and checkinmethods("nrnFFT"):
			pl=plt.subplot(211)
		elif checkinmethods("netFFT"):
			pl=plt.subplot(111)
		if checkinmethods("netFFT"):
			plt.title( (methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+ "Network spectrum")
			plt.bar(specX[1:],fft[1:pnum],0.75,color="k",edgecolor="k")
		if checkinmethods("netFFT") and checkinmethods("nrnFFT"):
			plt.subplot(212,sharex=pl)
		elif checkinmethods("nrnFFT"):
			plt.subplot(111)
		if checkinmethods("nrnFFT"):
			plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Neuronal spectrum")
			plt.bar(specX[1:],specN[1:],0.75,color="k",edgecolor="k")

	#plt.subplot(313,sharex=p)
	#specX =np.arange(0.0,methods['tstop']+h.dt,h.dt)
	#specX *= 1000.0/methods['tstop']/h.dt
	#pnum = specX.size/2
	#plt.title("Voltage spectrum")
	#plt.plot(specX[1:pnum],specV[1:pnum])
	#plt.xlim(0,200)
	
	if 10 < methods["netISI"] <= 3000 and sum(netisi) > 0: netisi /= sum(netisi)
	if 10 < methods["nrnISI"] <= 3000 and sum(isi) > 0: isi /= sum(isi)
	if (10 < methods["netISI"] <= 3000 or 10 < methods["nrnISI"] <= 3000) and methods['gui']:
		plt.figure(3)
		if 10 < methods["netISI"] <= 3000 and 10 < methods["nrnISI"] <= 3000:
			pl=plt.subplot(211)
		elif 10 < methods["netISI"] <= 3000 :
			plt.subplot(111)
		if 10 < methods["netISI"] <= 3000: 
			plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Network ISI")
			plt.ylabel("Normalized number of interspike intervals", fontsize=16)
			plt.bar(np.arange(methods["netISI"]),netisi,0.75,color='k')
			plt.xlim(0,methods["netISI"])
		if 10 < methods["netISI"] <= 3000 and 10 < methods["nrnISI"] <= 3000:
			plt.subplot(212)#,sharex=pl)
		elif 10 < methods["nrnISI"] <= 3000:
			plt.subplot(111)
		if 10 < methods["nrnISI"] <= 3000:
			plt.ylabel("Normalized number of interspike intervals", fontsize=16)
			plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Neuronal ISI")
			plt.bar(np.arange(methods["nrnISI"]),isi,0.75,color='k')
			plt.xlim(0,methods["nrnISI"])
			plt.xlabel("Interspike intervals (ms)", fontsize=16)
	
	if ( checkinmethods('traceView') or checkinmethods('pop-pp-view') )and checkinmethods('gui'):
		if not checkinmethods('PhaseLims'):
			methods['PhaseLims'] = [ (-76.,40.),(0.,1.) ]
		elif not (type(methods['PhaseLims']) is list or type(methods['PhaseLims']) is tuple ) or len(methods['PhaseLims']) != 2:
			print("/PhaseLims should be a list of two tuples for x and y coordinats.\n Not list given\n")
			exit(1) 
		elif not (type(methods['PhaseLims'][0]) is list or type(methods['PhaseLims'][0]) is tuple ) or len(methods['PhaseLims'][0]) != 2:
			print("/PhaseLims should be a list of two tuples for x and y coordinats\n First coordinat isn'a a tuple")
			exit(1) 
		elif not (type(methods['PhaseLims'][1]) is list or type(methods['PhaseLims'][1]) is tuple ) or len(methods['PhaseLims'][1]) != 2:
			print("/PhaseLims should be a list of two tuples for x and y coordinats\n Second coordinat isn'a a tuple")
			exit(1) 
	if checkinmethods('traceView') and checkinmethods('gui'):
#>>>>>>>			
		def numsptk(postidx,idxrange):
			prespikes = np.array([])
			trange=t[idxrange]
			sptk = np.zeros(trange.size)
			for nidx in OUTList[postidx]:
				sptime = np.array(neurons[nidx].spks)
				sptime = sptime[ np.where( (sptime > trange[0]) * (sptime < trange[-1]) ) ]
				prespikes = np.append(prespikes,sptime)
			
			prespikes = np.sort(prespikes)
			#print prespikes
			accumulator = 0
			for tm in trange:
				mp = np.where(prespikes < tm)[0]
				sptk[np.where( trange == tm )] = mp.size
			return sptk
			
		def getprespikes(postidx,tl,tr):
			postspk = []
			for nidx in OUTList[postidx]:
				for nspk in neurons[nidx].spks[ np.where( (neurons[nidx].spks >= tl)*(neurons[nidx].spks < tr) ) ]:
					postspk.append([nspk,nidx] )
				
			return np.array( postspk )

		def zoolyupdate(imax):
			zoolyclickevent.spikesymbol = "."
			zoolyclickevent.imax = imax
			onclick1.lines[imax].set_linewidth(4)
			onclick1.lines[imax].set_ls("--")
			zooly.canvas.draw()
			
			zoolyclickevent.v = np.array(neurons[imax].volt)
			zoolyclickevent.u = np.array(neurons[imax].svar)
			zoolyclickevent.g = np.array(neurons[imax].isyng)
			zoolyclickevent.i = np.array(neurons[imax].inoise)
			if hasattr(neurons[imax], "iandnoise"):
				zoolyclickevent.i += np.array(neurons[imax].iandnoise)
			zoolyclickevent.rst = getprespikes(imax,onclick1.tl, onclick1.tr)
			moddyupdate(idx)
			
		def zoolyclickevent(event):
			if not hasattr(onclick1,"lines"): return
			et = event.xdata
			ev = event.ydata
			idx = np.where( np.abs(t-et)<h.dt)[0][0]
			#DB>>
			#print idx, et,ev
			#<<DB
			vmax = abs(neurons[0].volt.x[idx] - ev)
			zoolyclickevent.imax = 0
#			for ind,n in map(None,xrange(methods["ncell"]),neurons):
			for ind,n in enumerate(neurons):
				onclick1.lines[ind].set_linewidth(1)
				onclick1.lines[ind].set_ls("-")
				if vmax > abs(n.volt.x[idx] - ev) :
					vmax = abs(n.volt.x[idx] - ev)
					zoolyclickevent.imax = ind
				#print vmax,n.volt.x[idx],ev

		def moddyclickevent(event):
			et = event.xdata
			idx = np.where( np.abs(t-et)<h.dt)[0][0]
			moddyupdate(idx)


		def moddyupdate(idx):
			if not hasattr(moddyupdate,"tail"):
				moddyupdate.tail = False
			if moddyupdate.tail:
				ridx = np.where( onclick1.idx == idx)[0][0]+1
				moddyupdate.tailsize = 300
				lidx = ridx-moddyupdate.tailsize
				if lidx < 0 :
					lidx = 0
			moddyupdate.idx = idx
			vmin,vmax = methods["PhaseLims"][0]
			nmin,nmax = methods["PhaseLims"][1]
			n0c,v0c,v0n,thc,thn,tpy = getnulls(vindex,vmin,vmax,zoolyclickevent.g[idx], zoolyclickevent.i[idx],neurons[zoolyclickevent.imax].innp.mean)
			#DB>>
			#print "\n\nDB>> THC=",thc,"THN=",thn,tpy != 2 and not thc is None,"\n\n"
			#<<DB
			moddyupdate.rst  = getprespikes(zoolyclickevent.imax,onclick1.tl, onclick1.tr)
			###!!!!zoolyclickevent.lines

			if not hasattr(moddyupdate,"lines"):
				
				dsynmax = np.max(zoolyclickevent.g[onclick1.idx]) if not checkinmethods("tV-synmax") else methods["tV-synmax"]
				#DB>>
				print("\n\n----\nDB! dsynmax",dsynmax,"\n----\n\n")
				#<<DB
				moddyupdate.lines = [
					faxi.plot([],[], "k"+zoolyclickevent.spikesymbol,ms=9,lw=5)[0]\
						if zoolyclickevent.rst.shape[0] == 0 else \
						faxi.plot(zoolyclickevent.rst[:,0],zoolyclickevent.rst[:,1],"k"+zoolyclickevent.spikesymbol,ms=9,lw=5)[0],
					vaxi.plot(tprin[onclick1.idx],zoolyclickevent.v[onclick1.idx],"k-")[0],
					uaxi.plot(tprin[onclick1.idx],zoolyclickevent.u[onclick1.idx],"k-")[0],
					gaxi.plot(tprin[onclick1.idx],zoolyclickevent.g[onclick1.idx],"k-")[0],
					iaxi.plot(tprin[onclick1.idx],zoolyclickevent.i[onclick1.idx],"k-")[0],
					naxi.scatter(zoolyclickevent.v[onclick1.idx],zoolyclickevent.u[onclick1.idx],\
						c=zoolyclickevent.g[onclick1.idx]/dsynmax,cmap=cmap,vmin=0., vmax=1.,linewidths=0)\
						if checkinmethods("color-phase") else\
					naxi.plot(zoolyclickevent.v[onclick1.idx],zoolyclickevent.u[onclick1.idx],"k-")[0],
					naxi.plot(n0c[:,0],n0c[:,1],"r:",label="n\'=0")[0],
					naxi.plot(v0c[:,0],v0c[:,1],"b-.",label="v\'=0",lw=2)[0],
					#None if checkinmethods("non-isnt-vnul") else naxi.plot(v0n[:,0],v0n[:,1],"b.",mfc="b",mec="b",ms=9)[0],
					None if checkinmethods("non-isnt-vnul") else naxi.plot(v0n[:,0],v0n[:,1],"b--",mfc="b",mec="b",ms=9)[0],
					naxi.plot([zoolyclickevent.v[idx]],[zoolyclickevent.u[idx]],"r.",mfc="r",mec="r",ms=24)[0],
					naxi.plot([],[],"k--",lw=3)[0] if tpy != 2 or thc is None else naxi.plot(thc[:,0],thc[:,1],"k--",lw=3)[0],
					naxi.plot([],[],"k-.",lw=3)[0] if tpy != 2 or thn is None else naxi.plot(thn[:,0],thn[:,1],"k-.",lw=3)[0],
				]
				#try:
					#with open("nulls/threshould-JR.pkl",'rb') as fd:
						#thx = pickle.load(fd)
						#moddyupdate.lines.append(naxi.plot(thx[:,0],thx[:,1],"g--",label="dv/dn=1"),)
				#except:
					#pass
				naxi.legend(loc=0)
			else:
				if zoolyclickevent.rst.shape[0] == 0 :
					moddyupdate.lines[0].set_xdata([])
					moddyupdate.lines[0].set_ydata([])
				else:
					moddyupdate.lines[0].set_xdata(zoolyclickevent.rst[:,0])
					moddyupdate.lines[0].set_ydata(zoolyclickevent.rst[:,1])
				moddyupdate.lines[1].set_xdata(tprin[onclick1.idx])
				moddyupdate.lines[1].set_ydata(zoolyclickevent.v[onclick1.idx])
				moddyupdate.lines[2].set_xdata(tprin[onclick1.idx])
				moddyupdate.lines[2].set_ydata(zoolyclickevent.u[onclick1.idx])
				moddyupdate.lines[3].set_xdata(tprin[onclick1.idx])
				moddyupdate.lines[3].set_ydata(zoolyclickevent.g[onclick1.idx])
				moddyupdate.lines[4].set_xdata(tprin[onclick1.idx])
				moddyupdate.lines[4].set_ydata(zoolyclickevent.i[onclick1.idx])
				##
				if checkinmethods("color-phase"):pass
				elif moddyupdate.tail:
					moddyupdate.lines[5].set_xdata(zoolyclickevent.v[onclick1.idx[lidx:ridx]])
					moddyupdate.lines[5].set_ydata(zoolyclickevent.u[onclick1.idx[lidx:ridx]])
				else:
					moddyupdate.lines[5].set_xdata(zoolyclickevent.v[onclick1.idx])
					moddyupdate.lines[5].set_ydata(zoolyclickevent.u[onclick1.idx])
				moddyupdate.lines[6].set_xdata(n0c[:,0])
				moddyupdate.lines[6].set_ydata(n0c[:,1])
				moddyupdate.lines[7].set_xdata(v0c[:,0])
				moddyupdate.lines[7].set_ydata(v0c[:,1])
				if not checkinmethods("non-isnt-vnul"):
					moddyupdate.lines[8].set_xdata(v0n[:,0])
					moddyupdate.lines[8].set_ydata(v0n[:,1])
				moddyupdate.lines[9].set_xdata([zoolyclickevent.v[idx]])
				moddyupdate.lines[9].set_ydata([zoolyclickevent.u[idx]])
				if tpy == 2 and not thc is None:
					moddyupdate.lines[10].set_xdata(thc[:,0])
					moddyupdate.lines[10].set_ydata(thc[:,1])
				else:
					moddyupdate.lines[10].set_xdata([])
					moddyupdate.lines[10].set_ydata([])
				if tpy == 2 and not thn is None:
					moddyupdate.lines[11].set_xdata(thn[:,0])
					moddyupdate.lines[11].set_ydata(thn[:,1])
				else:
					moddyupdate.lines[11].set_xdata([])
					moddyupdate.lines[11].set_ydata([])
			faxi.set_ylim(0,methods["ncell"])
			vaxi.set_ylim(-85.,40.)
			uaxi.set_ylim(0.,1.)
			if not checkinmethods("tV-synmax"):
				gaxi.set_ylim(0.,zoolyclickevent.g[onclick1.idx].max())
			else:
				gaxi.set_ylim(0.,methods["tV-synmax"])
			#iaxi.set_ylim(zoolyclickevent.i[onclick1.idx].min(),zoolyclickevent.i[onclick1.idx].max())
			faxi.set_xlim(onclick1.tl, onclick1.tr)
			naxi.set_xlim(vmin,vmax)
			naxi.set_ylim(nmin,nmax)
			if not hasattr(moddyupdate,"markers"):
				moddyupdate.markers= [
					faxi.plot([t[idx],t[idx]],[0,methods['ncell']],"r--")[0],
					vaxi.plot([t[idx],t[idx]],[vmin,vmax],"r--")[0],
					uaxi.plot([t[idx],t[idx]],[0,1],"r--")[0],
					gaxi.plot([t[idx],t[idx]],[0,zoolyclickevent.g[onclick1.idx].max()],"r--")[0],
					iaxi.plot([t[idx],t[idx]],iaxi.get_ylim(),"r--")[0]
				]

			else:
				for line in moddyupdate.markers:
					line.set_xdata([t[idx],t[idx]])
			moddy.canvas.draw()

			
		def zoolykeyevent(event):
			if not hasattr(zoolyclickevent,"lines"): return
			if event.key == "K":
				v,u,g,i = (
					np.array(neurons[zoolyclickevent.imax].volt),
					np.array(neurons[zoolyclickevent.imax].svar),
					np.array(neurons[zoolyclickevent.imax].isyng),
					np.array(neurons[zoolyclickevent.imax].inoise)
					)
				sptk = numsptk(zoolyclickevent.imax,onclick1.idx)
				rst = getprespikes(zoolyclickevent.imax,onclick1.tl, onclick1.tr)
				moddyupdate.lines.append(faxi.plot(zoolyclickevent.rst[:,0],zoolyclickevent.rst[:,1],zoolyclickevent.spikesymbol,ms=7,lw=5)[0])
				moddyupdate.lines.append(vaxi.plot(tprin[onclick1.idx],v[onclick1.idx])[0])
				moddyupdate.lines.append(uaxi.plot(tprin[onclick1.idx],u[onclick1.idx])[0])
				moddyupdate.lines.append(gaxi.plot(tprin[onclick1.idx],g[onclick1.idx])[0])
				moddyupdate.lines.append(naxi.plot(v[onclick1.idx],u[onclick1.idx])[0])
				moddyupdate.lines.append(iaxi.plot(tprin[onclick1.idx],i[onclick1.idx])[0])
				#zoolyclickevent.lines.append(saxi.plot(tprin[onclick1.idx],sptk)[0])
			elif event.key == "X":
				for lin in moddyupdate.lines:
					lin.remove()
				del zoolyclickevent.lines
			moddy.canvas.draw()	

		def moddykeyevent(event):
			if event.key == "K" or event.key == "X":
				zoolykeyevent(event)
			elif event.key == "left":
				moddyupdate.idx -= 1
				if moddyupdate.idx < onclick1.idx[0] :
					moddyupdate.idx = onclick1.idx[0]
				moddyupdate(moddyupdate.idx)
			elif event.key == "right":
				moddyupdate.idx += 1
				if moddyupdate.idx > onclick1.idx[-1] :
					moddyupdate.idx = onclick1.idx[-1]
				moddyupdate(moddyupdate.idx)
			elif event.key == "pageup":
				moddyupdate.idx -= 10
				if moddyupdate.idx < onclick1.idx[0] :
					moddyupdate.idx = onclick1.idx[0]
				moddyupdate(moddyupdate.idx)
			elif event.key == "pagedown":
				moddyupdate.idx += 10
				if moddyupdate.idx > onclick1.idx[-1] :
					moddyupdate.idx = onclick1.idx[-1]
				moddyupdate(moddyupdate.idx)
			elif event.key == "home":
				moddyupdate.idx = onclick1.idx[0]
				moddyupdate(moddyupdate.idx)
			elif event.key == "end":
				moddyupdate.idx = onclick1.idx[-1]
				moddyupdate(moddyupdate.idx)
			elif event.key == "T":
				moddyupdate.tail = not moddyupdate.tail
				moddyupdate(moddyupdate.idx)
			elif event.key == "M":
				ridx = np.where( onclick1.idx == moddyupdate.idx)[0][0]+1
				nmax = len(onclick1.idx[ridx::5])
				moddy.set_size_inches(18.5, 10.5, forward=True)
				timestamp = time.strftime("%Y%m%d%H%M%S")
				moviedir = methods["movie-dir"] if checkinmethods("movie-dir") else "/home/rth/tmp/"
				print("==================================")
				print("===        Making MOVIE        ===")
				print("  > Time Stamp                 : ",timestamp)
				print("  > Fraim step (mc)            : ",5. * methods['timestep'])
				print("  > Tail length (ms)           : ",float(moddyupdate.tailsize) * methods['timestep'])
				print("  > Movie Dir                  : ",moviedir)
				for ndx,idx in enumerate(onclick1.idx[ridx::5]):
					moddyupdate(idx)
					#moddy.savefig("/home/rth/media/rth-storage-old/rth/tmp/%s-%04d.jpg"%(timestamp,ndx))
					moddy.savefig("%s/%s-%04d.jpg"%(moviedir,timestamp,ndx))
					sys.stderr.write("  > Frame:%04d of %04d         : Saved\r"%(ndx,nmax))
				print("\n==================================\n")
			elif event.key == "S":
				if hasattr(moddykeyevent,"spx"):
					spx.remove()
					del spx
				else:
					with open("separatrix/separatrix.pkl",'rb') as fd:
						spx = pickle.load(fd)
					#spx = np.genfromtxt("quasithresh.dat")[:,1:]
					for fx in np.linspace(0.0,0.27,6):
						naxi.fill_between(spx[:,0],spx[:,1]+fx, spx[:,1]-fx, facecolor='grey', alpha=0.3-fx*0.3/0.27)
					
					spx, = naxi.plot(spx[:,0],spx[:,1],"k--")
				moddy.canvas.draw()	
			#elif event.key == "J":
				#if hasattr(moddykeyevent,"thx"):
					#thx.remove()
					#del thx
				#else:
					##with open("nulls/threshould.pkl",'rb') as fd:
					#with open("nulls/threshould-JR.pkl",'rb') as fd:
						#thx = pickle.load(fd)
					#print thx
					#spx, = naxi.plot(thx[:,0],thx[:,1],"g--")
				#moddy.canvas.draw()	
			elif event.key == "D":
				if hasattr(moddykeyevent,"d10p"):
					for x in moddykeyevent.d10p: x.remove()
					del moddykeyevent.d10p
				else:
					vdata = np.array(neurons[zoolyclickevent.imax].volt)
					udata = np.array(neurons[zoolyclickevent.imax].svar)
					gdata = np.array(neurons[zoolyclickevent.imax].isyng)
					moddykeyevent.d10p = []
					maxg = methods['maxg'] if checkinmethods('maxg') else np.max(gdata)*0.1 
					print("MAXG:",maxg)
					d10idx = np.where(gdata<maxg)[0]
					d10idx = [ idx0 for idx0, idx1 in zip(d10idx[:-1],d10idx[1:]) if idx1 != idx0+1 and idx0>=onclick1.idx[0] and idx0<=onclick1.idx[-1]]+\
							 [ idx1 for idx0, idx1 in zip(d10idx[:-1],d10idx[1:]) if idx0 != idx1-1 and idx1>=onclick1.idx[0] and idx1<=onclick1.idx[-1]]
					d10idx.sort()
					moddykeyevent.d10p.append( gaxi.plot(tprin[d10idx], gdata[d10idx], "kX",ms=12)[0] )
					moddykeyevent.d10p.append( naxi.plot(vdata[d10idx], udata[d10idx], "kX",ms=12)[0] )
					
					d10idx = (np.diff(np.sign(np.diff(gdata))) < 0).nonzero()[0] + 1
					d10idx = [ idx for idx in d10idx if idx>=onclick1.idx[0] and idx<=onclick1.idx[-1]]
					d10idx.sort()
					moddykeyevent.d10p.append( gaxi.plot(tprin[d10idx], gdata[d10idx], "rX",ms=12)[0] )
					moddykeyevent.d10p.append( naxi.plot(vdata[d10idx], udata[d10idx], "rX",ms=12)[0] )
				moddy.canvas.draw()
			elif  event.key == "G":
				print(np.max(np.array(neurons[zoolyclickevent.imax].isyng) ))
			else:
				print(event.key)
				

#<<<<<<<		
		zooly = plt.figure(4 )
		zooly.canvas.mpl_connect('button_press_event', zoolyclickevent)
		zooly.canvas.mpl_connect('key_press_event', zoolykeyevent)
		moddy = plt.figure(5,figsize=(16,7) )
		faxi = plt.subplot2grid((6,10),(0,0),colspan=4,rowspan=2)
		faxi.set_ylabel("Presynaptic spikes",fontsize=12)
		vaxi = plt.subplot2grid((6,10),(2,0),colspan=4,sharex=faxi)
		vaxi.set_ylabel("V[mV]",fontsize=12)
		uaxi = plt.subplot2grid((6,10),(3,0),colspan=4,sharex=faxi)
		uaxi.set_ylabel('n',fontsize=12)
		gaxi = plt.subplot2grid((6,10),(4,0),colspan=4,sharex=faxi)
		gaxi.set_ylabel(r"$g_{syn} [uS]$",fontsize=12)
		iaxi = plt.subplot2grid((6,10),(5,0),colspan=4,sharex=faxi)
		iaxi.set_ylabel(r"$I_{noise} [nA]$",fontsize=12)
		#saxi = plt.subplot2grid((6,10),(5,0),colspan=4,sharex=faxi)
		naxi = plt.subplot2grid((6,10),(0,5),colspan=6,rowspan=6)
		naxi.set_ylabel('n',fontsize=12)
		naxi.set_xlabel("V[mV]",fontsize=12)
		moddy.canvas.mpl_connect('key_press_event', moddykeyevent)# zoolykeyevent)
		moddy.canvas.mpl_connect('button_press_event', moddyclickevent)


	if checkinmethods('GPcurve') and checkinmethods('gui'):
		plt.figure(7)
		f  = np.array([ [n.gsynscale,n.spks.size]       for n in neurons])
		#f = np.sort(f, axis=0)
		plt.plot(f[:,0] ,f[:,1],"k+")
			
	if checkinmethods('sycleprop') and checkinmethods('gui'):
		plt.figure(8)
		polarax = plt.subplot(111, polar=True)
		#bars = polarax.bar(phydist[:,1], phydist[:,0], width=0.25, bottom=0.0)
		#np.histogram(phydist[:,0], bins=180, weights=phydist[:,1])
		#polarax.hist(phydist[:,0], bins=36, weights=phydist[:,1])
		polarax.bar(phyhistbins[:-1],phyhist,width=phyhistbins[1]-phyhistbins[0],bottom=0)
		#DB>>
		plt.figure(9)
		plt.bar(phyhistbins[:-1],phyhist,width=phyhistbins[1]-phyhistbins[0],bottom=0)
		#<<DB
	if checkinmethods('Gtot-dist') and checkinmethods('gui'):
		plt.figure(10)
		plt.bar(gskbins[:-1]+(gskbins[1]+gskbins[0])/2.,gskhist,width=(gskbins[1]-gskbins[0]),color="k")
		#plt.hist(gsk,bins=methods["ncell"]/50)
		plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"mean total synaptic conductance={}(uS), stdr total synaptic conductance={}(uS)".format(mgto,sgto) )
		plt.ylabel("Probability")
		plt.xlabel("Toaol conductance (uS)")

	if checkinmethods("Delay-stat") and checkinmethods('gui'):
		plt.figure(11)
		plt.bar((dlybins[1:]+dlybins[:-1])/2.,dlyhist,width=dlybins[1]-dlybins[0],color="k")
		plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"mean delay={}(ms), stdr delay={}(ms)".format(mdly,sdly))
		plt.ylabel("Probability")
		plt.xlabel("delay(ms)")
	
	if checkinmethods('T1vsT2/spikerate') and checkinmethods('gui'):
		plt.figure(12)
		plt.bar([1,2],[methods['T1vsT2']['spikerate']['T1'],methods['T1vsT2']['spikerate']['T2']],width=0.1,color="k")
		plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Mean firing rate")
		plt.ylabel("Firing rate spike/sec")
		plt.xlabel("Type")

	if checkinmethods('pop-pp-view') and checkinmethods('gui'):
		def PopPPview_update(idx):
			ppp_ppp = np.array(
				[ (n.volt.x[idx],n.svar.x[idx]) for n in neurons]
			)
			if checkinmethods('pop-pp-view-color'):
				ppp_pfpp.set_offsets(ppp_ppp)
			else:
				ppp_pfpp.set_xdata(ppp_ppp[:,0])
				ppp_pfpp.set_ydata(ppp_ppp[:,1])
			
			vmin,vmax = methods["PhaseLims"][0]
			n0c,v0c,v0n,thc,thn,type21 = getnulls(0,vmin,vmax,float(ppp_av_gsyn[idx]),float(ppp_av_mean+ppp_simmod[idx-tproc]),float(ppp_av_mean+ppp_simmod[idx-tproc]) )
			ppp_sax_m.set_xdata(t[idx])
			ppp_vax_m.set_xdata(t[idx])
			ppp_nax_m.set_xdata(t[idx])
			ppp_gax_m.set_xdata(t[idx])
			ppp_pfn0.set_xdata(n0c[:,0])
			ppp_pfn0.set_ydata(n0c[:,1])
			ppp_pfv0.set_xdata(v0c[:,0])
			ppp_pfv0.set_ydata(v0c[:,1])
			ppp_pfvN.set_xdata(v0n[:,0])
			ppp_pfvN.set_ydata(v0n[:,1])
			if type21 == 2:
				ppp_pfth0.set_xdata(thc[:,0])
				ppp_pfth0.set_ydata(thc[:,1])
				ppp_pfthi.set_xdata(thn[:,0])
				ppp_pfthi.set_ydata(thn[:,1])
			else:
				ppp_pfth0.set_xdata([])
				ppp_pfth0.set_ydata([])
				ppp_pfthi.set_xdata([])
				ppp_pfthi.set_ydata([])
			PopPPview.canvas.draw()
		
		def PopPPview_keyevent(event):
			global ppp_idx, ppp_lines
			if event.key == "K":
				n0c,v0c,v0n,thc,thn,type21 = getnulls(0,vmin,vmax,float(ppp_av_gsyn[ppp_idx]),float(ppp_av_mean+ppp_simmod[ppp_idx-tproc]),float(ppp_av_mean+ppp_simmod[ppp_idx-tproc]) )
				ppp_lines.append( ppp_ppax.plot(v0n[:,0],v0n[:,1],"-",ms=3)[0] )
				if type21 == 2:
					ppp_lines.append(ppp_ppax.plot(thn[:,0],thn[:,1],"-",lw=1)[0])
			if event.key == "X":
				for lin in ppp_lines.append:
					lin.remove()
				ppp_lines.append = []
			elif event.key == "left":      ppp_idx -= 1
			elif event.key == "right":     ppp_idx += 1
			elif event.key == "pageup":    ppp_idx -= 10
			elif event.key == "pagedown":  ppp_idx += 10
			elif event.key == "home":      ppp_idx  = tproc
			elif event.key == "end":       ppp_idx  = tproc+tprin.size
			elif event.key == "M":
				#PopPPview.set_size_inches(18.5, 10.5, forward=True)
				nmax = tprin.size/5
				timestamp = time.strftime("%Y%m%d%H%M%S")
				moviedir = methods["movie-dir"] if checkinmethods("movie-dir") else "/home/rth/tmp/"
				print("==================================")
				print("===        Making MOVIE        ===")
				print("  > Time Stamp                 : ",timestamp)
				print("  > Fraim step (mc)            : ",5. * methods['timestep'])
				#print "  > Tail length (ms)           : ",float(moddyupdate.tailsize) * methods['timestep']
				print("  > Movie Dir                  : ",moviedir)
				

				for ndx,idx in enumerate(range(tproc,tproc+tprin.size,5)):
					PopPPview_update(idx)
					#PopPPview.savefig("/home/rth/tmp/movies/%s-%04d.jpg"%(timestamp,ndx))
					#PopPPview.savefig("/home/rth/media/rth-media/rth-media/tmp/movie/%s-%04d.jpg"%(timestamp,ndx))
					PopPPview.savefig("%s/%s-%04d.jpg"%(moviedir,timestamp,ndx))
					sys.stderr.write("  > Frame:%04d of %04d         : Saved\r"%(ndx,nmax))
				os.system("ffmpeg -r 10 -f image2  -i %s/%s-%%04d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p %s/%s.mp4"%(moviedir,timestamp,moviedir,timestamp) )
				os.system("rm %s/%s-*.jpg"%(moviedir,timestamp) )
				print("\n==================================\n")
				
			if ppp_idx < tproc:            ppp_idx  = tproc
			if ppp_idx > tproc+tprin.size: ppp_idx  = tproc+tprin.size
			PopPPview_update(ppp_idx)
			
		def PopPPview_clickevent(event):
			global ppp_idx
			et = event.xdata
			ppp_idx = np.where( np.abs(t-et)<h.dt)[0][0]
			PopPPview_update(ppp_idx)
		
		PopPPview=plt.figure(13,figsize=(16,7) )
		ppp_lines = []
		ppp_idx = tprin.size/2+tproc
		ppp_sax = plt.subplot2grid((4,3),(0,0) )#,colspan=4,rowspan=2)
		ppp_sax.set_ylabel("neuron",fontsize=12)
		if checkinmethods('pop-pp-view-color'):
			ppp_s_color = np.array([ -neurons[nindex[int(n)][1]].innp.mean for n in rast[:,1] ])
			ppp_s_color /= np.amax(ppp_s_color)
			if not checkinmethods('pop-pp-view-spsize'): methods['pop-pp-view-spsize'] = 3
			ppp_sax.scatter(rast[:,0],rast[:,1], s=methods['pop-pp-view-spsize'],c=ppp_s_color, cmap=matplotlib.cm.get_cmap('rainbow'))#,ms=9)
		else:
			ppp_sax.plot(rast[:,0],rast[:,1],"k"+methods['rstmark'],mew=0.75,ms=methods['rstmarksize'])
		ppp_sax_m, = ppp_sax.plot([t[ppp_idx],t[ppp_idx]],[0,methods['ncell']],"r--")
		ppp_vax = plt.subplot2grid((4,3),(1,0),sharex=ppp_sax) 
		ppp_vax.set_ylabel("V[mV]",fontsize=12)
		xmin,xmax = methods["PhaseLims"][0]
		for n in neurons:
			ppp_vax.plot(tprin,np.array(n.volt)[tproc:tprin.size+tproc],'-',c="#000000",lw=0.1)
			xm, xM = np.min(np.array(n.volt)[tproc:tprin.size+tproc]),np.max(np.array(n.volt)[tproc:tprin.size+tproc])
			if xmin > xm : xmin = xm
			if xmax < xM : xmax = xM
		ppp_vax_m, = ppp_vax.plot([t[ppp_idx],t[ppp_idx]],[xmin,xmax],"r--")

		ppp_nax = plt.subplot2grid((4,3),(2,0),sharex=ppp_sax)
		ppp_nax.set_ylabel('n',fontsize=12)
		xmin,xmax = methods["PhaseLims"][1]
		for n in neurons:
			ppp_nax.plot(tprin,np.array(n.svar)[tproc:tprin.size+tproc],'-',c="#000000",lw=0.1)
			xm, xM = np.min(np.array(n.svar)[tproc:tprin.size+tproc]),np.max(np.array(n.svar)[tproc:tprin.size+tproc])
			if xmin > xm : xmin = xm
			if xmax < xM : xmax = xM
		ppp_nax_m, = ppp_nax.plot([t[ppp_idx],t[ppp_idx]],[xmin,xmax],"r--")

		ppp_gax = plt.subplot2grid((4,3),(3,0),sharex=ppp_sax)
		ppp_gax.set_ylabel(r"mean $g_{syn} [uS]$",fontsize=12)
		ppp_av_gsyn = np.array(neurons[0].isyng)
		for n in neurons[1:]:
			ppp_av_gsyn += np.array(n.isyng)
		ppp_av_gsyn /= methods['ncell']
		ppp_gax.plot(tprin,np.array(ppp_av_gsyn)[tproc:tprin.size+tproc],"k-")
		xmin,xgmax = np.min(ppp_av_gsyn),np.max(ppp_av_gsyn)
		ppp_gax_m, = ppp_gax.plot([t[ppp_idx],t[ppp_idx]],[xmin,xgmax],"r--")
		
		ppp_ppax = plt.subplot2grid((4,3),(0,1),colspan=2,rowspan=5)
		ppp_ppax.set_ylabel('n',fontsize=12)
		ppp_ppax.set_xlabel("V[mV]",fontsize=12)
		ppp_av_mean = np.mean( [ n.innp.mean for n in neurons ] )
		ppp_ppp = np.array(
			[ (n.volt.x[ppp_idx],n.svar.x[ppp_idx]) for n in neurons]
		)
		if checkinmethods('pop-pp-view-color'):
			ppp_color = np.array([ -n.innp.mean for n in neurons ])
			ppp_color /= np.amax(ppp_color)
			if not checkinmethods('pop-pp-view-nrnsize'): methods['pop-pp-view-nrnsize'] = 52
			ppp_pfpp = ppp_ppax.scatter(ppp_ppp[:,0],ppp_ppp[:,1], s=methods['pop-pp-view-nrnsize'],c=ppp_color, cmap=matplotlib.cm.get_cmap('rainbow'))#,ms=9)
		else:
			ppp_pfpp, = ppp_ppax.plot(ppp_ppp[:,0],ppp_ppp[:,1],"ro",ms=9)
		vmin,vmax = methods["PhaseLims"][0]
		nmin,nmax = methods["PhaseLims"][1]
		if checkinmethods('sinmod'):
			tstart = methods["sinmod"]["tstart"] if checkinmethods('sinmod/tstart') else 200.
			tstop  = methods["sinmod"]["tstop" ] if checkinmethods('sinmod/tstop' ) else 2200.
			per    = methods["sinmod"]["per"   ] if checkinmethods('sinmod/per'   ) else 2000.
			amp    = methods["sinmod"]["amp"   ] if checkinmethods('sinmod/amp'   ) else 7e-7
			bias   = methods["sinmod"]["bias"  ] if checkinmethods('sinmod/bias'  ) else 0.
			ppp_simmod =  bias - amp/2*(1.-np.cos(np.pi*2./per*(tprin-tstart))) 
			ppp_simmod[np.where(tprin<tstart)] = 0.
			ppp_simmod[np.where(tprin>tstop )] = 0.
			ppp_gax.plot(tprin,(bias-ppp_simmod/amp)*xgmax,"b--")
		else:
			ppp_simmod = np.zeros(tprin.shape)
			
		n0c,v0c,v0n,thc,thn,type21 = getnulls(0,vmin,vmax,float(ppp_av_gsyn[ppp_idx]),float(ppp_av_mean+ppp_simmod[ppp_idx-tproc]),float(ppp_av_mean+ppp_simmod[ppp_idx-tproc]) )
		ppp_pfn0, = ppp_ppax.plot(n0c[:,0],n0c[:,1],"r--",lw=1)
		ppp_pfv0, = ppp_ppax.plot(v0c[:,0],v0c[:,1],"b--",lw=1)
		ppp_pfvN, = ppp_ppax.plot(v0n[:,0],v0n[:,1],"b-",ms=3)
		if type21 == 2:
			ppp_pfth0, = ppp_ppax.plot(thc[:,0],thc[:,1],"k--",lw=1)
			ppp_pfthi, = ppp_ppax.plot(thn[:,0],thn[:,1],"k-.",lw=1)
		else:
			ppp_pfth0, = ppp_ppax.plot([],[],"k--",lw=1)
			ppp_pfthi, = ppp_ppax.plot([],[],"k-.",lw=1)
		ppp_ppax.set_ylim(nmin,nmax)
		ppp_ppax.set_xlim(vmin,vmax)
		PopPPview.canvas.mpl_connect('key_press_event',    PopPPview_keyevent)# zoolykeyevent)
		PopPPview.canvas.mpl_connect('button_press_event', PopPPview_clickevent)

	if checkinmethods("ttFFT"):
		if methods['tracetail'] == 'total current' or methods['tracetail'] == 'TI' or methods['tracetail'] == 'mean total current' or methods['tracetail'] == 'MTI'\
		  or methods['tracetail'] == 'total synaptic current' or methods['tracetail'] == 'TSI' or methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'MTSI':
			ttfft_dt = np.mean(tprin[1:]-tprin[:-1])
			ttfft_sr = meancur[tproc:tprin.size+tproc]
			ttfft_dr = (tprin[-1] - tprin[0])
			ttspecX	= np.arange(ttfft_sr.shape[0], dtype=float)
			ttspecX	*= 1000.0/ttfft_dr
			ttpnum 	= int(200.*ttfft_dr/1000.0)
			ttspecX	= ttspecX[:ttpnum]
			ttfft = np.abs( np.fft.fft(ttfft_sr)*1.0/ttfft_dr )**2
			if checkinmethods('gui'):
				ttftpf = plt.figure(12)
				plt.bar(ttspecX[1:],ttfft[1:ttpnum],0.75,color="k",edgecolor="k")
				plt.ylabel("Power ($nA^2$)", fontsize=16)
				plt.xlabel("Frequency (Hz)", fontsize=16)
				if methods['tracetail'] == 'total current' or methods['tracetail'] == 'TI':
					plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Power Spectrum of Total Current", fontsize=16)
				elif methods['tracetail'] == 'mean total current' or methods['tracetail'] == 'MTI':
					plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Power Spectrum of Mean Total Current", fontsize=16)
				elif methods['tracetail'] == 'total synaptic current' or methods['tracetail'] == 'TSI':
					plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Power Spectrum of Total Synaptric Current", fontsize=16)
				elif methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'MTSI':
					plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Power Spectrum of Mean Total Synaptric Current", fontsize=16)
		elif methods['tracetail'] == 'total conductance' or methods['tracetail'] == 'mean total conductance' or methods['tracetail'] == 'TG' or methods['tracetail'] == 'MTG':
			ttfft_dt = np.mean(tprin[1:]-tprin[:-1])
			ttfft_sr = meancur[tproc:tprin.size+tproc]
			ttfft_dr = (tprin[-1] - tprin[0])
			ttspecX	= np.arange(ttfft_sr.shape[0], dtype=float)
			ttspecX	*= 1000.0/ttfft_dr
			ttpnum 	= int(200.*ttfft_dr/1000.0)
			ttspecX	= ttspecX[:pnum]
			ttfft = np.abs( np.fft.fft(ttfft_sr)*1.0/ttfft_dr )**2			
			if checkinmethods('gui'):
				ttftpf = plt.figure(12)
				plt.bar(ttspecX[1:],ttfft[1:pnum],0.75,color="k",edgecolor="k")
				plt.ylabel("Power ($\mu S^2$)", fontsize=16)
				plt.xlabel("Frequency (Hz)", fontsize=16)
				if   methods['tracetail'] == 'total conductance' or methods['tracetail'] == 'TG':
					plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Power Spectrum of Total Conductance", fontsize=16)
				elif methods['tracetail'] == 'mean total conductance' or methods['tracetail'] == 'MTG':
					plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Power Spectrum of Mean Total Conductance", fontsize=16)
		elif methods['tracetail'] == 'p2eLFP':			
			ttfft_lfp=np.zeros(module.shape[0])
			x1,x2=0.,0,
			for i,s in enumerate(module):
				x1 = s+x1-x1/2.
				x2 = s+x2-x2/5.
				ttfft_lfp[i]=x2-x1
			if checkinmethods("p2eLFP/LPF"):
				from scipy.signal import butter, lfilter, freqz
				nyq = 0.5 * 1000.				#SAMPLING EVERY 1 ms
				normal_cutoff = methods["p2eLFP"]["LPF"] / nyq		#lowpass 100Hz
				b, a = butter(5, normal_cutoff, btype='low', analog=False)
				ttfft_lfp = lfilter(b, a, ttfft_lfp)
			ttfft_lfp = ttfft_lfp[int(round(methods['cliptrn'])):]
			ttfft_dt = 1.
			ttfft_dr = float(ttfft_lfp.shape[0])
			ttspecX	= np.arange(ttfft_lfp.shape[0], dtype=float)
			ttspecX	*= 1000.0/ttfft_dr
			ttpnum 	= int(200.*ttfft_dr/1000.0)
			ttspecX	= ttspecX[:pnum]
			ttfft = np.abs( np.fft.fft(ttfft_lfp)*1.0/ttfft_dr )**2	
			if checkinmethods('gui'):
				ttftpf = plt.figure(12)
				plt.bar(ttspecX[1:],ttfft[1:pnum],0.75,color="k",edgecolor="k")
				plt.ylabel("Power ($\mu S^2$)", fontsize=16)
				plt.xlabel("Frequency (Hz)", fontsize=16)
				plt.title((methods["MainFigTitle"] if checkinmethods("MainFigTitle") else "")+"Power Spectrum of Generated LFP", fontsize=16)
		methods["ttFFT-Retults"]={
			'Freq'  : ttspecX[1:],
			'Power' : ttfft[1:pnum]
			}
	if ( checkinmethods('N2NHI') or checkinmethods('N2NHI-netISI') ) and checkinmethods('gui'): 
		#plt.figure(19)
		#if checkinmethods('N2NHI') and checkinmethods('N2NHI-netISI'):
			#clsAx = plt.subplot(121)
			#plt.bar(np.arange(len(ccc_clsidx))+1,np.array(ccc_clsidx),0.75,color="k",edgecolor="k")
			#plt.subplot(122, sharex=clsAx,sharey=clsAx)
			#plt.bar(np.arange(len(rth_clsidx))+1,np.array(rth_clsidx),0.75,color="k",edgecolor="k")
			#plt.xlim(1,10)
		#elif checkinmethods('N2NHI'):
			#plt.bar(np.arange(len(ccc_clsidx))+1,np.array(ccc_clsidx),0.75,color="k",edgecolor="k")
			#plt.xlim(1,10)
		#elif checkinmethods('N2NHI-netISI'):
			#plt.bar(np.arange(len(rth_clsidx))+1,np.array(rth_clsidx),0.75,color="k",edgecolor="k")
			#plt.xlim(1,10)
		plt.figure(20)
		plt.bar(cg_nrnbin,cg_nrnisi,0.75,color="k",edgecolor="k")

	if checkinmethods('vpop-view') and checkinmethods('gui'):
		import matplotlib as mpl
		from mpl_toolkits.mplot3d import Axes3D
		vpop_view=plt.figure(21)
		ax = vpop_view.add_subplot(111, projection='3d')
		for nidx in range(methods['ncell']-1,-1,-1):
			ax.plot(tprin,float(nidx)*np.ones(tprin.shape[0]),np.array(neurons[nindex[nidx][1]].volt)[tproc:tproc+tprin.size],"-")
	if checkinmethods('CtrlISI') and checkinmethods('gui'):
		plt.figure(22)
		plt.bar(CtrISI[:,0],CtrISI[:,1],methods["CtrlISI"]["bin"],color="k",edgecolor="k")
		plt.xlim(0,methods["CtrlISI"]["bin"]+methods["CtrlISI"]["max"])
		plt.xlabel("ISI (ms)")
		plt.ylabel("Number of spikes")
		if Pnet is not None:
			pnets = np.arange(methods["CtrlISI"]["max"]/Pnet)*Pnet+methods["CtrlISI"]["bin"]/2
			pnets = pnets[1:]
			xmax = np.amax(CtrISI[:,1])
			plt.plot(pnets,np.ones(pnets.shape[0])*xmax*1.1,"kv")
			plt.text(methods["CtrlISI"]["max"]/2,xmax/2,r"$F_{{nrn}}/F_{{net}}$"+"\n{}".format(Pnet/methods["nrnPmean"]),fontsize=24)
	if checkinmethods('spike2net-dist') and checkinmethods('gui') and checkinmethods('spike2net-dist-result'):
		plt.figure(23)
		peakd = np.array(methods['spike2net-dist-result'])
		settd = np.arange(-100.,100)
		plt.bar(settd,peakd,0.75,color="k",edgecolor="k")
		plt.ylabel("Number of spikes")
		plt.xlabel(r"Phase of network period ( \% )")
	if checkinmethods('nrnFRhist') and checkinmethods('gui'):
		plt.figure(24)
		if type(methods['nrnFRhist']) is dict:
			#filter for names:
			hparam = {}
			for n in methods['nrnFRhist']:
				if n in ["bins","range","normed","weights","density"]:
					hparam[n]=methods['nrnFRhist'][n]
			nrnfrhist,nrnfrages=np.histogram(nrnfr,**hparam)
			if "xnorm" in methods['nrnFRhist'] and methods['nrnFRhist']["xnorm"]:
				nrnfrhist = nrnfrhist.astype(float)/float(np.sum(nrnfrhist))
			nrnfrages = (nrnfrages[1:]+nrnfrages[:-1])/2.
			plt.bar(nrnfrages,nrnfrhist,width=(nrnfrages[1]-nrnfrages[0])*0.9,edgecolor='k',facecolor='k')
			if "ymax" in methods['nrnFRhist']:
				plt.ylim(ymax=methods['nrnFRhist']["ymax"])
			if "ymin" in methods['nrnFRhist']:
				plt.ylim(ymin=methods['nrnFRhist']["ymin"])
		else:
			nrnfrhist,nrnfrages=np.histogram(nrnfr)
			nrnfrages = (nrnfrages[1:]+nrnfrages[:-1])/2.
			plt.bar(nrnfrages,nrnfrhist,width=(nrnfrages[1]-nrnfrages[0])*0.9,edgecolor='k',facecolor='k')
		
			
	if checkinmethods('git'):
		os.system("git commit -a &")
	if checkinmethods('beep'):
		os.system("beep &")
	if checkinmethods('corelog'):
		def writetree(tree,fd,prefix):
			for name in tree:
				if type(tree[name]) is dict:
					writetree(tree[name],fd,prefix+name+'/')
				elif isinstance(tree[name],np.ndarray):
					fd.write(":{}={}".format(prefix+name,tree[name].tolist()))
				else:
					#DB>>
					#print prefix,name,tree[name]
					#<<DB
					fd.write(":{}={}".format(prefix+name,tree[name]))
		with open(methods['corelog']+".simdb","a") as fd:
			now = datetime.now()
			fd.write("SIMDB/TIMESTAMP=(%04d,%02d,%02d,%02d,%02d,%02d)"%(now.year, now.month, now.day, now.hour, now.minute, now.second) )
			writetree(methods,fd,"/")
			fd.write("\n")
	if methods['gui']:
		if methods['gif']:
			plt.savefig(methods['gif'])
		else:
			plt.show()
	if not methods['noexit']:
		sys.exit(0)

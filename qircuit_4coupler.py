import numpy as np
import copy

def check_circuit_doublewell(circuit):
	"""
		Check whether the circuit spectrum is None, zero-length, flat, or has no double-well feature.
		If one or more of those is the case, the output is False. Otherwise, it is True. Spectrum is
		in GHz.
		Input
		circuit: circuit dict
		Output
		check: bool
	"""

	# Initialization
	check    = True
	spectrum = np.array(circuit['measurements']['eigen_spectrum']).T

	# Check validity of spectrum
	if circuit['measurements']['eigen_spectrum'] is None:
		check = False
	elif len(spectrum) == 0:
		check = False
	elif np.max(spectrum[0]) - np.min(spectrum[0]) < 0.001: #almost constant spectrum
		check = False

	# Check outermost level population
	elif 'max_pop' in circuit['measurements'] and circuit['measurements']['max_pop'] > 0.01:
		print('MAX POP TOO HIGH!')
		check = False

	# Check for double well
	else:
		# Extract ground and excited state spectra
		spectrumGS = spectrum[0]
		spectrumES = spectrum[1]

		### Check for double-well feature ###
		# Edges higher than inner region?
		c1 = (spectrumGS[0] > np.min(spectrumGS)) and (spectrumGS[-1] > np.min(spectrumGS))
		# Center higher than minima?
		mididx = len(spectrumGS) // 2
		c2 = spectrumGS[mididx] > np.min(spectrumGS)
		# Peak has significant height
		c3 = (spectrumGS[mididx] - np.min(spectrumGS)) > 0.05
		# Energy levels do not cross
		c4 = np.min(np.array(spectrumES) - np.array(spectrumGS)) > 0.1
		check = c1 and c2 and c3 and c4

	return check


def merit_DoubleWell(circuit, merit_options, **kwargs):

	print('# Calculating double-well merit ...')

	# Loss function settings
	max_peak       = merit_options['max_peak']
	max_split      = merit_options['max_split']
	norm_p         = merit_options['norm_p']
	flux_sens_bool = merit_options['flux_sens']
	max_merit      = merit_options['max_merit']

	# Check if circuit is valid and has double well
	if check_circuit_doublewell(circuit):

		# Circuit spectrum
		spectrum = np.array(circuit['measurements']['eigen_spectrum']).T

		# Extract ground and excited state spectra
		spectrumGS = spectrum[0]
		spectrumES = spectrum[1]

		### Calculate loss ###

		# (1) Check if flux sensitivity should be calculated and submit extra task if necessary
		hsens = None
		# Context circuits have been calculated
		if flux_sens_bool and ('context_circuits' in kwargs):
			hsens = 0
			context_circuits = kwargs['context_circuits']
			for context_circuit in context_circuits:
				# Context circuit valid
				if check_circuit_doublewell(context_circuit):
					spectrum_context = np.array(context_circuit['measurements']['eigen_spectrum']).T
					spectrumGS_context = spectrum_context[0]
					mididx = len(spectrumGS_context) // 2
					hsens += abs( np.min(spectrumGS_context[:mididx]) - np.min(spectrumGS_context[mididx:]) )
				# Context circuit invalid
				else:
					loss = max_merit
					print('# INVALID CONTEXT CIRCUIT -> LOSS: {} ...'.format(loss))
					merit_dict = {'loss': loss, 'extra_tasks': []}
					return merit_dict
					
		# Context circuits have *not* been calculated
		elif flux_sens_bool:
			print('# PREPARING CONTEXT CIRCUITS ...')
			# Determine number of flux biases
			if 'num_biases' in circuit['measurements']:
				num_biases = circuit['measurements']['num_biases']
				# Case A: no additional biases beyond the main loop
				if num_biases<=1:
					print('# No additional biases, moving to merit calculation ...')
					pass
				# Case B: perturb existing additional biases
				else:
					print('# Preparing {} perturbed circuit(s)'.format(num_biases-1))
					perturbed_circuits = []
					for i in range(1,num_biases):
						perturbed_circuit = copy.deepcopy(circuit)
						perturbed_circuit['measurements'].clear()
						perturbed_circuit['circuit']['circuit_values']['phiOffs'][i] += 0.0001
						perturbed_circuits.append(perturbed_circuit)
					merit_dict = {'extra_tasks': perturbed_circuits}
					return merit_dict
			# Number of biases can not be determined (was not saved)
			else:
				loss = max_merit
				print('# NUMBER OF BIASES COULD NOT BE DETERMINED -> LOSS: {} ...'.format(loss))
				merit_dict = {'loss': loss, 'extra_tasks': []}
				return merit_dict

		# (2) Center peak height
		idx_mid = int(len(spectrumGS)/2)
		hpeak = spectrumGS[idx_mid] - np.min(spectrumGS)
		hpeak = np.min((hpeak, max_peak))

		# (3) Level separation
		hsplit = np.min(spectrumES-spectrumGS)
		hsplit = np.min((hsplit, max_split))

		# Combined loss
		if hsens == None:
			loss = max_merit - ( (hpeak/max_peak)**norm_p + (hsplit/max_split)**norm_p )**(1/norm_p)
		else:
			# print('A:', (hpeak/max_peak))
			# print('B:', (hsplit/max_split))
			# print('C:', noise_factor*(1 - np.min((hsens/hpeak, 1))))
			loss = max_merit - ( abs(hpeak/max_peak)**norm_p + abs(hsplit/max_split)**norm_p \
								 + (1 - np.min((hsens/hpeak, 1)))**norm_p )**(1/norm_p)

	else:
		loss = max_merit

	print('# LOSS: {} ...'.format(loss))
	
	merit_dict = {'loss': loss, 'extra_tasks': []}

	return merit_dict

def solver_JJcircuitSimV3(Carr, Larr, JJarr, nLin=6, nNol=8, nJos=11, nIsl=1, timeout=600, fluxSweep=True, phiOffs=[0.5,0.5,0.5,0.5]):
	"""
		Returns:
		eigenspec - 10-dim numpy array of the circuit eigenspectrum at fixed flux. If fluxSweep=True,
		returns a 10x41-dim array with flux sweep of each eigenmode (first flux bias chosen if multiple
		biases available).
		timeout_bool - flag for expired timeout
		Only returned if it can be determined:
			num_biases - number of flux biases available in circuit
		Parameters:
		Carr - flattened upper triangular of capacitance matrix [fF]
		Larr - flattened upper triangular of inductance matrix [pH]
		JJarr - flattened upper triangular of Josephson junction matrix [GHz]
		nLin, nNol, nJos, nIsl - truncation of linear, non-linear, Josephson and island modes
		timeout - timeout for Mathematica simulation in seconds
		fluxBool - perform flux sweep if set to True
		Note: The components matrices are entered as 1-dim numpy arrays stepping through the
		upper triangular matrix row-wise, e.g. np.array([c11, c12, c13, c22, c23, c33])
	"""

	print('Running circuit simulation...')

	if not os.path.exists('../Mathematica_scriptV2-JJsimV3.wl'):
		raise NotImplementedError('JJcircuitSim module not available')

	tA = time.time()

	timeout_bool = False

	tstart = time.time()

	# Write parameters to text file, for Mathematica to read
	Carr.astype('float32').tofile('Carr.dat')
	Larr.astype('float32').tofile('Larr.dat')
	JJarr.astype('float32').tofile('JJarr.dat')
	phiOffs = np.array(phiOffs)
	phiOffs.astype('float32').tofile('phiOffs.dat') #flux biases for loops

	# Subprocess to run Mathematica script
	try:
		run(["wolframscript", "-file", "../Mathematica_scriptV2-JJsimV3.wl",
			str(nLin), str(nNol), str(nJos), str(nIsl), str(int(fluxSweep))],
			timeout=timeout, stdout=open(os.devnull, 'wb'))
	except TimeoutExpired:
		print('ERROR: timeout expired')
		timeout_bool = True
		return None, timeout_bool

	# Extract eigenspectrum from file
	eigenspec = []
	if not os.path.isfile('log_eigenspectrum.csv'):
		return None, timeout_bool

	try: #try opening and reading spectrum file
		with open('log_eigenspectrum.csv', 'r') as datafile:
			if fluxSweep:
				reader = csv.reader(datafile, delimiter=',')
				for row in reader:
					eigenspec.append([float(e) for e in row])
			else:
				for n, line in enumerate(datafile):
					eigenspec.append(float(line.strip()))
	except AttributeError:
		return None, timeout_bool

	# Determine number of flux biases and outermost level population
	try:
		## Number of flux biases ##
		with open('log_biasinfo.txt', 'r') as datafile:
			content = datafile.read()
		start_indices = [m.start() for m in re.finditer('Fb', content)]
		matches = [content[i:i+3] for i in start_indices] #assumes no more than 9 biases
		matches = list(set(matches))
		num_biases = len(matches)
		print('# Found {} flux biases'.format(num_biases))

		## Maximum outermost level population of eigenstates ##
		with open('log_diagonalization.txt', 'r') as datafile:
			content = datafile.read()
		# Find indices of level pop information for each mode
		start_indices = np.array([m.start() for m in re.finditer('Max level probs', content)])
		stop_temp     = np.array([m.start() for m in re.finditer('}', content)])
		stop_indices  = np.array([np.min([t for t in stop_temp-s if t>0])+s for s in start_indices])
		# Extract level pops and convert to float
		levelpops = []
		for s,t in zip(start_indices,stop_indices):
			l = content[s+19:t]
			pattern = re.compile('\*\^')
			l = pattern.sub('e', l)
			l = l.split(',')
			l = [float(e) for e in l]
			levelpops.append(l)
		# Determine maximum outermost level pop overall
		max_pop = np.max([np.max(l) for l in levelpops])
		print('# Maximum outermost level pop: {}'.format(max_pop))

		print('Simulation time: {} s'.format(time.time()-tA))
		return eigenspec, timeout_bool, num_biases, max_pop

	except:
		print('ERROR: could not determine number of flux biases or outermost level population')
		print('Simulation time: {} s'.format(time.time()-tA))
		return eigenspec, timeout_bool



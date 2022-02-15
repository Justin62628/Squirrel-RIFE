import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io

def mean_cov(X):
	n,p = X.shape
	m = X.mean(axis=0)
	cx = X - m
	S = dgemm(1./(n-1), cx.T, cx.T, trans_a=0, trans_b=1)
	return cx,m,S.T


def rolling_window_lastaxis(a, window):
	"""Directly taken from Erik Rigtorp's post to numpy-discussion.
	<http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
	if window < 1:
		raise(ValueError, "`window` must be at least 1.")
	if window > a.shape[-1]:
		raise(ValueError, "`window` is too long.")
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window(a, window):
	if not hasattr(window, '__iter__'):
		return rolling_window_lastaxis(a, window)
	for i, win in enumerate(window):
		if win > 1:
			a = a.swapaxes(i, -1)
			a = rolling_window_lastaxis(a, win)
			a = a.swapaxes(-2, i)
	return a

class Steerable:
	def __init__(self, height = 4, order = 4, twidth = 1):
		"""
		height is the total height, including highpass and lowpass
		"""
		self.nbands = np.round(order)
		self.nbands = np.double(self.nbands)

		self.height = height
		self.twidth = twidth

	#this should just return the levels at angle
	#a lvl x images array
	def steerAngle(self, im, angle):
		#anglemask = self.pointOp(angle, Ycosn, Xcosn + (np.pi*b)/self.nbands).astype(np.complex)
		#banddft = (np.complex(0,-1)**order) * lodft
		#banddft *= anglemask
		#banddft *= himask
		pass
		
	def buildSFpyr(self, im):

		M, N = im.shape[:2]
		log_rad, angle = self.base(M, N)

		Xrcos, Yrcos = self.rcosFn(1, -0.5)
		Yrcos = np.sqrt(Yrcos)
		YIrcos = np.sqrt(1 - Yrcos*Yrcos)

		lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
		hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)

		imdft = np.fft.fftshift(np.fft.fft2(im))
		lo0dft = imdft * lo0mask

		coeff = self.buildSFpyrlevs(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height - 1)

		hi0dft = imdft * hi0mask
		hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))
		coeff.insert(0, hi0.real)
		return coeff


	def buildSFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
		if (ht <=1):
			lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
			coeff = [lo0.real]
		
		else:
			#shift by 1 octave
			Xrcos = Xrcos - np.log2(2)

			# ==================== Orientation bandpass =======================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = np.pi * np.array(range(-(2*lutsize+1),(lutsize+2)))/lutsize
			order = self.nbands - 1
			const = (2**(2*order) * sc.factorial(order)**2) / (self.nbands * sc.factorial(2*order))
			Ycosn = np.sqrt(const) * (np.cos(Xcosn)**order)

			M, N = np.shape(lodft)
			orients = np.zeros((int(self.nbands), M, N))
			for b in range(int(self.nbands)):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + (np.pi*b)/self.nbands).astype(np.complex)
				banddft = (np.complex(0,-1)**order) * lodft
				banddft *= anglemask
				banddft *= himask
				orients[b, :, :] = np.fft.ifft2(np.fft.ifftshift(banddft)).real

			# ================== Subsample lowpass ============================
			dims = np.array(lodft.shape)
			
			lostart = np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)
			loend = lostart + np.ceil((dims-0.5)/2)

			lostart = lostart.astype(int)
			loend = loend.astype(int)

			log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = np.abs(np.sqrt(1 - Yrcos*Yrcos))
			lomask = self.pointOp(log_rad, YIrcos, Xrcos)

			lodft = lomask * lodft

			coeff = self.buildSFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht-1)
			coeff.insert(0, orients)
		return coeff

	def reconSFPyrLevs(self, coeff, log_rad, Xrcos, Yrcos, angle):
		if (len(coeff) == 1):
			return np.fft.fftshift(np.fft.fft2(coeff[0]))
		else:
			Xrcos = Xrcos - 1
    		
			# ========================== Orientation residue==========================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = np.pi * np.array(range(-(2*lutsize+1),(lutsize+2)))/lutsize
			order = self.nbands - 1
			const = np.power(2, 2*order) * np.square(sc.factorial(order)) / (self.nbands * sc.factorial(2*order))
			Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

			orientdft = np.zeros(coeff[0][0].shape, 'complex')

			for b in range(int(self.nbands)):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + (np.pi*b)/self.nbands)
				banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
				orientdft += ((np.complex(0,1)**(order)) * banddft * anglemask * himask)

			# ============== Lowpass component are upsampled and convoluted ============
			dims = np.array(coeff[0][0].shape)
			
			lostart = np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2) 
			loend = lostart + np.ceil((dims-0.5)/2) 

			nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
			lomask = self.pointOp(nlog_rad, YIrcos, Xrcos)

			nresdft = self.reconSFPyrLevs(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

			res = np.fft.fftshift(np.fft.fft2(nresdft))

			resdft = np.zeros(dims, 'complex')
			resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

			return resdft + orientdft

	def reconSFpyr(self, coeff):

		if ((self.nbands) != len(coeff[1])):
			raise Exception("Unmatched number of orientations")

		M, N = coeff[0].shape
		log_rad, angle = self.base(M, N)

		Xrcos, Yrcos = self.rcosFn(1, -0.5)
		Yrcos = np.sqrt(Yrcos)
		YIrcos = np.sqrt(np.abs(1 - Yrcos*Yrcos))

		lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
		hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)

		tempdft = self.reconSFPyrLevs(coeff[1:], log_rad, Xrcos, Yrcos, angle)

		hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
		outdft = tempdft * lo0mask + hidft * hi0mask

		return np.fft.ifft2(np.fft.ifftshift(outdft)).real


	def base(self, m, n):
		ctrm = np.ceil((m + 0.5)/2).astype(int)
		ctrn = np.ceil((n + 0.5)/2).astype(int)

		xv, yv = np.meshgrid((np.array(range(n)) + 1 - ctrn),
				(np.array(range(m)) + 1 - ctrm))
		xv = xv.astype(np.double)
		yv = yv.astype(np.double)
		xv *= (2.0/n)
		yv *= (2.0/m)

		rad = np.sqrt(xv**2 + yv**2)
		rad[ctrm - 1, ctrn-1] = rad[ctrm - 1, ctrn - 2]
		log_rad = np.log2(rad)

		angle = np.arctan2(yv, xv)
		
		return log_rad, angle

	def rcosFn(self, width, position):
		N = 256
		X = np.pi * np.array(range(-N-1, 2))
		X /= 2.0*N

		Y = np.cos(X)**2
		Y[0] = Y[1]
		Y[N+2] = Y[N+1]

		X = position + 2*width/np.pi*(X + np.pi/4)
		return X, Y

	def pointOp(self, im, Y, X):
		out = np.interp(im.flatten(), X, Y)
		return np.reshape(out, im.shape)

	#divisive normalization (same as DIIVINE)
	def normalize(self, coef, height, order):
		filtsize = (3, 3)
		norm_bands = []
		for pyr_h in xrange(height-2):
			inner_norm_bands = []
			sublevel = coef[pyr_h+1]
			for cband in xrange(order):
				child = coef[0]
				parent = []
				w, h = np.shape(sublevel[cband])
				if pyr_h > 0:
					child= scipy.misc.imresize(coef[pyr_h][cband], 50, interp='bilinear', mode='F')
				if pyr_h+3 < height:
					#parent = scipy.misc.imresize(coef[pyr_h+2][cband], 2.0, interp='bilinear', mode='F')
					parent = scipy.misc.imresize(coef[pyr_h+2][cband], 200, interp='bilinear', mode='F')
					parent = parent[1:-1, 1:-1]
					wp, hp = np.shape(parent)
					#print np.shape(parent)
					if wp > w-2:
						parent = parent[:w-2, :]
					if hp > h-2:
						parent = parent[:, :h-2]
					#print np.shape(parent)
					#print np.shape(parent)
					#exit(0)


				idx = np.hstack((np.arange(0, cband), np.arange(cband+1, order)))

				#stick it all in a matrix
				if parent == []:
					cov = np.array(np.hstack((
							#split image into overlapping blocks
							rolling_window(sublevel[cband], filtsize).reshape(((w-2)*(h-2), 9)),
							#grab coefficients from neighboring orientations
							sublevel.transpose(1, 2, 0)[1:-1, 1:-1, idx].reshape((w-2)*(h-2), order-1),
							#np.matrix(child[1:-1, 1:-1].reshape((w-2)*(h-2))).T,
					)))
				else:
					#parent sometimes gets an extra pixel 
					#print np.shape(np.matrix(parent[1:-1, 1:-1]))#.reshape((w-2)*(h-2))).T,
					#print w-2, h-2
					cov = np.array(np.hstack((
							rolling_window(sublevel[cband], filtsize).reshape(((w-2)*(h-2), 9)),
							#grab coefficients from neighboring orientations
							sublevel.transpose(1, 2, 0)[1:-1, 1:-1, idx].reshape((w-2)*(h-2), order-1),
							np.matrix(parent.reshape((w-2)*(h-2))).T,
							#np.matrix(child[1:-1, 1:-1].reshape((w-2)*(h-2))).T,
					)))
				_, _, cov_mat = mean_cov(cov)

				#actual N
				N = np.shape(cov_mat)[0]

				#N from the matlab code
				N = 10 - pyr_h

				#force positive semi-definite
				eigval, eigvec = np.linalg.eig(cov_mat)
				Q = np.matrix(eigvec)
				xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
				cov_mat = Q*xdiag*Q.T

				#reference code claims to be correcting the cov matrix, by below basic computation
				#L = diag(diag(L).*(diag(L)>0))*sum(diag(L))/(sum(diag(L).*(diag(L)>0))+(sum(diag(L).*(diag(L)>0))==0));
				cov_inv = np.linalg.pinv(cov_mat)

				# perform normalization by sqrt(Y^T * C_U * Y)
				z = np.sqrt(np.einsum('ij,ij->i', np.dot(cov, cov_inv), cov)/N)
				#inner = np.sum(np.multiply(np.dot(cov, cov_inv), cov), axis=1).T/N

				cov[:, 4] -= np.average(cov[:, 4])

				result = cov[:, 4]/z
				gb = 16/(2**(pyr_h))

				result = result.reshape(w-2, h-2)[gb:-(gb), gb:-(gb)]
				result -= np.average(result)
				inner_norm_bands.append(result)

			norm_bands.append(inner_norm_bands)
		norm_bands = np.array(norm_bands)
		return norm_bands

def load_sp5filters():
  harmonics = np.array([1, 3, 5])

  mtx = np.array([
    [0.3333, 0.2887, 0.1667, 0.0000, -0.1667, -0.2887],
    [0.0000, 0.1667, 0.2887, 0.3333, 0.2887, 0.1667],
    [0.3333, -0.0000, -0.3333, -0.0000, 0.3333, -0.0000],
    [0.0000, 0.3333, 0.0000, -0.3333, 0.0000, 0.3333],
    [0.3333, -0.2887, 0.1667, -0.0000, -0.1667, 0.2887],
    [-0.0000, 0.1667, -0.2887, 0.3333, -0.2887, 0.1667]
  ])

  hi0filt = np.array([ 
    [-0.00033429, -0.00113093, -0.00171484, -0.00133542, -0.00080639, -0.00133542, -0.00171484, -0.00113093, -0.00033429],
    [-0.00113093, -0.00350017, -0.00243812, 0.00631653, 0.01261227, 0.00631653, -0.00243812, -0.00350017, -0.00113093],
    [-0.00171484, -0.00243812, -0.00290081, -0.00673482, -0.00981051, -0.00673482, -0.00290081, -0.00243812, -0.00171484],
    [-0.00133542, 0.00631653, -0.00673482, -0.07027679, -0.11435863, -0.07027679, -0.00673482, 0.00631653, -0.00133542],
    [-0.00080639, 0.01261227, -0.00981051, -0.11435863, 0.81380200, -0.11435863, -0.00981051, 0.01261227, -0.00080639],
    [-0.00133542, 0.00631653, -0.00673482, -0.07027679, -0.11435863, -0.07027679, -0.00673482, 0.00631653, -0.00133542],
    [-0.00171484, -0.00243812, -0.00290081, -0.00673482, -0.00981051, -0.00673482, -0.00290081, -0.00243812, -0.00171484],
    [-0.00113093, -0.00350017, -0.00243812, 0.00631653, 0.01261227, 0.00631653, -0.00243812, -0.00350017, -0.00113093],
    [-0.00033429, -0.00113093, -0.00171484, -0.00133542, -0.00080639, -0.00133542, -0.00171484, -0.00113093, -0.00033429]
  ])

  lo0filt = np.array([
    [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614],
    [-0.01551246, 0.05586982, 0.15925570, 0.05586982, -0.01551246],
    [-0.03848215, 0.15925570, 0.40304148, 0.15925570, -0.03848215],
    [-0.01551246, 0.05586982, 0.15925570, 0.05586982, -0.01551246],
    [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614]
  ])

  lofilt = 2*np.array([
    [0.00085404, -0.00244917, -0.00387812, -0.00944432, -0.00962054, -0.00944432, -0.00387812, -0.00244917, 0.00085404],
    [-0.00244917, -0.00523281, -0.00661117, 0.00410600, 0.01002988, 0.00410600, -0.00661117, -0.00523281, -0.00244917],
    [-0.00387812, -0.00661117, 0.01396746, 0.03277038, 0.03981393, 0.03277038, 0.01396746, -0.00661117, -0.00387812],
    [-0.00944432, 0.00410600, 0.03277038, 0.06426333, 0.08169618, 0.06426333, 0.03277038, 0.00410600, -0.00944432],
    [-0.00962054, 0.01002988, 0.03981393, 0.08169618, 0.10096540, 0.08169618, 0.03981393, 0.01002988, -0.00962054],
    [-0.00944432, 0.00410600, 0.03277038, 0.06426333, 0.08169618, 0.06426333, 0.03277038, 0.00410600, -0.00944432],
    [-0.00387812, -0.00661117, 0.01396746, 0.03277038, 0.03981393, 0.03277038, 0.01396746, -0.00661117, -0.00387812],
    [-0.00244917, -0.00523281, -0.00661117, 0.00410600, 0.01002988, 0.00410600, -0.00661117, -0.00523281, -0.00244917],
    [0.00085404, -0.00244917, -0.00387812, -0.00944432, -0.00962054, -0.00944432, -0.00387812, -0.00244917, 0.00085404]
  ])

  bfilts = np.array([
    [
      [0.00277643, 0.00496194, 0.01026699, 0.01455399, 0.01026699, 0.00496194, 0.00277643],
      [-0.00986904, -0.00893064, 0.01189859, 0.02755155, 0.01189859, -0.00893064, -0.00986904],
      [-0.01021852, -0.03075356, -0.08226445, -0.11732297, -0.08226445, -0.03075356, -0.01021852],
      [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
      [0.01021852, 0.03075356, 0.08226445, 0.11732297, 0.08226445, 0.03075356, 0.01021852],
      [0.00986904, 0.00893064, -0.01189859, -0.02755155, -0.01189859, 0.00893064, 0.00986904],
      [-0.00277643, -0.00496194, -0.01026699, -0.01455399, -0.01026699, -0.00496194, -0.00277643]
    ],
    [
      [-0.00343249, -0.00640815, -0.00073141, 0.01124321, 0.00182078, 0.00285723, 0.01166982],
      [-0.00358461, -0.01977507, -0.04084211, -0.00228219, 0.03930573, 0.01161195, 0.00128000],
      [0.01047717, 0.01486305, -0.04819057, -0.12227230, -0.05394139, 0.00853965, -0.00459034],
      [0.00790407, 0.04435647, 0.09454202, -0.00000000, -0.09454202, -0.04435647, -0.00790407],
      [0.00459034, -0.00853965, 0.05394139, 0.12227230, 0.04819057, -0.01486305, -0.01047717],
      [-0.00128000, -0.01161195, -0.03930573, 0.00228219, 0.04084211, 0.01977507, 0.00358461],
      [-0.01166982, -0.00285723, -0.00182078, -0.01124321, 0.00073141, 0.00640815, 0.00343249]
    ],
    [
      [0.00343249, 0.00358461, -0.01047717, -0.00790407, -0.00459034, 0.00128000, 0.01166982],
      [0.00640815, 0.01977507, -0.01486305, -0.04435647, 0.00853965, 0.01161195, 0.00285723],
      [0.00073141, 0.04084211, 0.04819057, -0.09454202, -0.05394139, 0.03930573, 0.00182078],
      [-0.01124321, 0.00228219, 0.12227230, -0.00000000, -0.12227230, -0.00228219, 0.01124321],
      [-0.00182078, -0.03930573, 0.05394139, 0.09454202, -0.04819057, -0.04084211, -0.00073141],
      [-0.00285723, -0.01161195, -0.00853965, 0.04435647, 0.01486305, -0.01977507, -0.00640815],
      [-0.01166982, -0.00128000, 0.00459034, 0.00790407, 0.01047717, -0.00358461, -0.00343249]
    ],
    [
      [-0.00277643, 0.00986904, 0.01021852, -0.00000000, -0.01021852, -0.00986904, 0.00277643],
      [-0.00496194, 0.00893064, 0.03075356, -0.00000000, -0.03075356, -0.00893064, 0.00496194],
      [-0.01026699, -0.01189859, 0.08226445, -0.00000000, -0.08226445, 0.01189859, 0.01026699],
      [-0.01455399, -0.02755155, 0.11732297, -0.00000000, -0.11732297, 0.02755155, 0.01455399],
      [-0.01026699, -0.01189859, 0.08226445, -0.00000000, -0.08226445, 0.01189859, 0.01026699],
      [-0.00496194, 0.00893064, 0.03075356, -0.00000000, -0.03075356, -0.00893064, 0.00496194],
      [-0.00277643, 0.00986904, 0.01021852, -0.00000000, -0.01021852, -0.00986904, 0.00277643]
    ],
    [
      [-0.01166982, -0.00128000, 0.00459034, 0.00790407, 0.01047717, -0.00358461, -0.00343249],
      [-0.00285723, -0.01161195, -0.00853965, 0.04435647, 0.01486305, -0.01977507, -0.00640815],
      [-0.00182078, -0.03930573, 0.05394139, 0.09454202, -0.04819057, -0.04084211, -0.00073141],
      [-0.01124321, 0.00228219, 0.12227230, -0.00000000, -0.12227230, -0.00228219, 0.01124321],
      [0.00073141, 0.04084211, 0.04819057, -0.09454202, -0.05394139, 0.03930573, 0.00182078],
      [0.00640815, 0.01977507, -0.01486305, -0.04435647, 0.00853965, 0.01161195, 0.00285723],
      [0.00343249, 0.00358461, -0.01047717, -0.00790407, -0.00459034, 0.00128000, 0.01166982]
    ],
    [
      [-0.01166982, -0.00285723, -0.00182078, -0.01124321, 0.00073141, 0.00640815, 0.00343249],
      [-0.00128000, -0.01161195, -0.03930573, 0.00228219, 0.04084211, 0.01977507, 0.00358461],
      [0.00459034, -0.00853965, 0.05394139, 0.12227230, 0.04819057, -0.01486305, -0.01047717],
      [0.00790407, 0.04435647, 0.09454202, -0.00000000, -0.09454202, -0.04435647, -0.00790407],
      [0.01047717, 0.01486305, -0.04819057, -0.12227230, -0.05394139, 0.00853965, -0.00459034],
      [-0.00358461, -0.01977507, -0.04084211, -0.00228219, 0.03930573, 0.01161195, 0.00128000],
      [-0.00343249, -0.00640815, -0.00073141, 0.01124321, 0.00182078, 0.00285723, 0.01166982]
    ]
  ])[:, ::-1, ::-1]

  return lo0filt.astype(np.float32), hi0filt.astype(np.float32), lofilt.astype(np.float32), bfilts.astype(np.float32), mtx.astype(np.float32), harmonics.astype(np.float32)

class SpatialSteerablePyramid():
  def __init__(self, height = 4):
    """
    height is the total height, including highpass and lowpass
    """

    self.height = height

  def corr(self, A, fw):
    h, w = A.shape
    
    sy2 = np.int(np.floor((fw.shape[0] - 1) / 2))
    sx2 = np.int(np.floor((fw.shape[1] - 1) / 2))

    # pad the same as the matlabpyrtools
    newpad = np.vstack((A[1:fw.shape[0]-sy2, :][::-1], A, A[h-(fw.shape[0]-sy2):h-1, :][::-1]))#,
    newpad = np.hstack((newpad[:, 1:fw.shape[1]-sx2][:, ::-1], newpad, newpad[:, w-(fw.shape[1]-sx2):w-1][:, ::-1]))
    newpad = newpad.astype(np.float32)

    return scipy.signal.correlate2d(newpad, fw, mode='valid').astype(np.float32)

  def buildLevs(self, lo0, lofilt, bfilts, edges, mHeight): 
    if mHeight <= 0:
      return [lo0]

    bands = []
    for i in range(bfilts.shape[0]):
      filt = bfilts[i]
      bands.append(self.corr(lo0, filt))

    lo = self.corr(lo0, lofilt)[::2, ::2]
    bands = [bands] + self.buildLevs(lo, lofilt, bfilts, edges, mHeight-1)

    return bands

  def decompose(self, inputimage, filtfile='sp1Filters', edges='symm'):
    inputimage = inputimage.astype(np.float32)

    if filtfile == 'sp5Filters':
      lo0filt,hi0filt,lofilt,bfilts,mtx,harmonics = load_sp5filters()
    else:
      raise(NotImplementedError, "That filter configuration is not implemnted")

    h, w = inputimage.shape

    hi0 = self.corr(inputimage, hi0filt)
    lo0 = self.corr(inputimage, lo0filt)

    pyr = self.buildLevs(lo0, lofilt, bfilts, edges, self.height)
    pyr = [hi0] + pyr

    return pyr

  def extractSingleBand(self, inputimage, filtfile='sp1Filters', edges='symm', band=0, level=1): 
    inputimage = inputimage.astype(np.float32)

    if filtfile == 'sp5Filters':
      lo0filt,hi0filt,lofilt,bfilts,mtx,harmonics = load_sp5filters()
    else:
      raise(NotImplementedError, "That filter configuration is not implemnted")

    h, w = inputimage.shape

    if level == 0:
      hi0 = self.corr(inputimage, hi0filt)
      singleband = hi0
    else:
      lo0 = self.corr(inputimage, lo0filt)
      for i in range(1, level):
        lo0 = self.corr(lo0, lofilt)[::2, ::2]

      # now get the band
      filt = bfilts[band]
      singleband = self.corr(lo0, filt)

    return singleband

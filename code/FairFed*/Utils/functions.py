import logging
import math

import numpy as np

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger (__name__)

# surrogate functions defined in M. Jordan's paper
phi_zero_one = lambda a: 1.0 if a < 0.0 else 0.0
phi_hinge = lambda a: 1.0 - a if a < 1.0 else 0.0
phi_logistic = lambda a: math.log (1.0 + math.exp (-1.0 * a))  # , 2)
phi_squared = lambda a: (1.0 - a) ** 2
phi_exponential = lambda a: math.exp (-a)

kappa_zero_one = lambda a: 1.0 if a > 0.0 else 0.0
kappa_hinge = lambda a: a + 1.0 if a > -1.0 else 0.0
kappa_logistic = lambda a: math.log (1 + math.exp (a))  # , 2)
kappa_squared = lambda a: (1.0 + a) ** 2
kappa_exponential = lambda a: math.exp (a)
kappa_linear = lambda a: a

delta_zero_one = lambda a: 1 if a > 0.0 else 0.0
delta_hinge = lambda a: a if a < 1.0 else 1.0
delta_logistic = lambda a: 1 - math.log (1 + math.exp (-a))  # , 2)
delta_squared = lambda a: 1 - (1.0 - a) ** 2
delta_exponential = lambda a: 1.0 - math.exp (-a)
delta_linear = lambda a: a

identity = lambda a: a
Zafar_hinge = lambda a: max (a, 0.0)
INF = 100

def compute_risk_difference(s, h, s_var, kappa_name='zero_one', delta_name='zero_one'):
	if kappa_name == 'zero_one':
		kappa = kappa_zero_one
	elif kappa_name == 'hinge':
		kappa = kappa_hinge
	elif kappa_name == 'squared':
		kappa = kappa_squared
	elif kappa_name == 'logistic':
		kappa = kappa_logistic
	elif kappa_name == 'exponential':
		kappa = kappa_exponential
	elif kappa_name == 'Zafar_hinge':
		kappa = Zafar_hinge
	elif kappa_name == 'Zafar_identity':
		kappa = identity
	elif kappa_name == 'linear':
		kappa = kappa_linear
	else:
		logger.error ('kappa %s is not included.' % kappa_name)
		kappa = kappa_zero_one

	if delta_name == 'zero_one':
		delta = delta_zero_one
	elif delta_name == 'hinge':
		delta = delta_hinge
	elif delta_name == 'squared':
		delta = delta_squared
	elif delta_name == 'logistic':
		delta = delta_logistic
	elif delta_name == 'exponential':
		delta = delta_exponential
	elif delta_name == 'Zafar_hinge':
		delta = Zafar_hinge
	elif delta_name == 'Zafar_identity':
		delta = identity
	elif delta_name == 'linear':
		delta = delta_linear
	else:
		logger.error ('delta %s is not included.' % delta_name)
		delta = delta_zero_one

	alpha = np.multiply (s, h)
	rd_kappa = alpha.apply (lambda l: kappa (l))
	rd_delta = alpha.apply (lambda l: delta (l))
	num_total, num_pos, num_neg = s_var.count (s)
	weight = s.apply (lambda val: 1.0 / num_pos if val > 0 else 1.0 / num_neg)

	if kappa_name == 'zero_one':
		return kappa_name, (np.multiply (rd_kappa, weight)).sum () - 1
	else:
		return kappa_name, delta_name, (np.multiply (rd_kappa, weight)).sum () - 1, (np.multiply (rd_delta, weight)).sum () - 1

def compute_risk(y, h, phi_name='zero_one'):
	if phi_name == 'zero_one':
		phi = phi_zero_one
	elif phi_name == 'hinge':
		phi = phi_hinge
	elif phi_name == 'squared':
		phi = phi_squared
	elif phi_name == 'logistic':
		phi = phi_logistic
	elif phi_name == 'exponential':
		phi = phi_exponential
	else:
		logger.error ('phi %s is not included.' % phi_name)
		phi = phi_zero_one

	alpha = np.multiply (y, h)
	return phi_name, alpha.apply (lambda l: phi (l)).mean ()

def compute_minimal_crd(eta, p, kappa_name='zero_one'):
	if kappa_name == 'zero_one':
		m = -1 if eta > p else 1
		kappa = kappa_zero_one
	elif kappa_name == 'hinge':
		m = -1 if eta > p else 1
		kappa = kappa_hinge
	elif kappa_name == 'squared':
		m = (p - eta) / (eta + p - 2 * eta * p)
		kappa = kappa_squared
	elif kappa_name == 'logistic':
		if eta == 0.0:
			m = INF
		elif eta == 1.0:
			m = -INF
		else:
			m = math.log ((1 - eta) * p / (1 - p) / eta)
		kappa = kappa_logistic
	elif kappa_name == 'exponential':
		m = math.log ((1 - eta) * p / (1 - p) / eta) / 2
		kappa = kappa_exponential
	else:
		logger.error ('kappa is not included')
		logger.info ('hinge is the default setting')
		m = -1 if eta > p else 1
		kappa = kappa_hinge

	def minimum(n):
		return eta / p * kappa (n) + (1 - eta) / (1 - p) * kappa (-n) - 1

	return minimum (m), m

def compute_maximal_crd(eta, p, delta_name='zero_one'):
	if delta_name == 'zero_one':
		m = 1 if eta > p else -1
		delta = delta_zero_one
	elif delta_name == 'hinge':
		m = 1 if eta > p else -1
		delta = delta_hinge
	elif delta_name == 'squared':
		m = (eta - p) / (eta + p - 2 * eta * p)
		delta = delta_squared
	elif delta_name == 'logistic':
		if eta == 0.0:
			m = -INF
		elif eta == 1.0:
			m = INF
		else:
			m = math.log ((1 - eta) * p / (1 - p) / eta)
		delta = delta_logistic
	elif delta_name == 'exponential':
		m = math.log ((1 - eta) * p / (1 - p) / eta) / 2
		delta = delta_exponential
	else:
		logger.error ('delta is not included')
		logger.info ('hinge is the default setting')
		m = -1 if eta > p else 1
		delta = delta_hinge

	def maximum(n):
		return eta / p * delta (n) + (1 - eta) / (1 - p) * delta (-n) - 1

	return maximum (m), m

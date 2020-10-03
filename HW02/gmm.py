from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio



# Set random seed so output is all same
np.random.seed(1)



class GMM(object):
    def __init__(self, X, K, max_iters = 100): # No need to change
        """
        Args: 
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        
        self.N = self.points.shape[0]        #number of observations
        self.D = self.points.shape[1]        #number of features
        self.K = K                           #number of components/clusters

    #Helper function for you to implement
    def softmax(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """
        max_logits = np.amax(logit, axis=1)
        max_logits = np.reshape(max_logits, (max_logits.size, 1))
        logit_norm = logit - max_logits
        logit_exp = np.exp(logit_norm)
        denominator = np.sum(logit_exp, axis=1)
        prob = logit_exp / np.reshape(denominator, (denominator.size, 1))
        return prob

    def logsumexp(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        max_logits = np.amax(logit, axis=1)
        max_logits_r = np.reshape(max_logits, (max_logits.size, 1))
        logit_norm = logit - max_logits_r
        logit_exp = np.exp(logit_norm)
        logit_sum = np.sum(logit_exp, axis=1)
        s = np.log(logit_sum)
        s = s + max_logits
        return np.reshape(s, (s.size, 1))

    #for undergraduate student
    def normalPDF(self, logit, mu_i, sigma_i): #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array (or array of lenth D), the center for the ith gaussian.
            sigma_i: 1xDxD 3-D numpy array (or DxD 2-D numpy array), the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array (or array of length N), the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.diagonal() should be handy.
        """
        N, D = np.shape(logit)
        if len(np.shape(sigma_i)) == 2:
            sigma_i = np.diagonal(sigma_i)
        else:
            sigma_i = np.diagonal(sigma_i[0])
        pdf = np.ones((1, N))
        for i in range(D):
            exponent = ((-2 * sigma_i[i])**-1) * np.square(logit[:, i] - mu_i[i])
            pdf *= ((2 * np.pi * sigma_i[i])**-0.5) * np.exp(exponent)
        return np.reshape(pdf, (N,))
    
    #for grad students
    def multinormalPDF(self, logits, mu_i, sigma_i):  #[5pts]
        """
        Args:
            logit: N x D numpy array
            mu_i: 1xD numpy array (or array of lenth D), the center for the ith gaussian.
            sigma_i: 1xDxD 3-D numpy array (or DxD 2-D numpy array), the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array (or array of length N), the probability distribution of N data for the ith gaussian
         
        Hint: 
            np.linalg.det() and np.linalg.inv() should be handy.
        """
        raise NotImplementedError
    
    
    def _init_components(self, **kwargs): # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
                You will have KxDxD numpy array for full covariance matrix case
        """
        pi = np.ones((self.K,))
        logit_shuffle = np.copy(self.points)
        np.random.shuffle(logit_shuffle)
        mu = logit_shuffle[:self.K, :]
        sigma = np.identity(self.D)
        sigma = np.stack([sigma]*self.K)
        return pi, mu, sigma

    
    def _ll_joint(self, pi, mu, sigma, **kwargs): # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        log_likelihood = np.zeros((self.N, self.K))
        for k in range(self.K):
            pdf = self.normalPDF(self.points, mu[k], np.reshape(sigma[k], (1, self.D, self.D)))
            log_likelihood[:, k] = np.log(pi[k] + 1e-32) + np.log(pdf + 1e-32)
        return log_likelihood

    def _E_step(self, pi, mu, sigma, **kwargs): # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: 
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        ln_likelihood = self._ll_joint(pi, mu, sigma)
        return self.softmax(ln_likelihood)

    def _M_step(self, gamma, **kwargs): # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Hint:  
            There are formulas in the slide and in the above description box.
        """
        N, K = np.shape(gamma)
        tau_max = np.argmax(gamma, axis=1)
        pi = np.zeros(K)
        mu = np.zeros((K, self.D))
        sigma = np.zeros((K, self.D, self.D))
        for k in range(K):
            idxes = np.where(tau_max == k)
            X = self.points[idxes, :]
            tau_k = gamma[:, k]
            N_k = np.sum(tau_k, axis=0)
            mu_k = np.matmul(tau_k, self.points) / N_k
            mu[k] = mu_k
            pi_k = N_k / N
            pi[k] = pi_k
            sigma_k = np.diagonal(np.matmul(np.transpose(tau_k) * np.transpose(self.points - mu_k), (self.points - mu_k)) / N_k)
            np.fill_diagonal(sigma[k], sigma_k)
        return pi, mu, sigma
    
    
    def __call__(self, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)       
        
        Hint: 
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters. 
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))
        
        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma)
            
            # M-step
            pi, mu, sigma = self._M_step(gamma)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

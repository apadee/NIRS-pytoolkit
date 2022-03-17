# -*- coding: utf-8 -*-

"""General linear model for NIRS"""

# Authors: Anna Padée <anna.padee@unifr.ch>
#
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
from math import factorial

import scipy
from sklearn.linear_model import LinearRegression
from nilearn.glm.first_level import run_glm
import scipy.stats as stats
from statsmodels.tsa.stattools import acf
from nilearn.glm.contrasts import compute_contrast as _cc


class GLM_NIRS:
    def __init__(self):
        self.design_matrix = np.empty(0)
        self.fs = 1
        self.HRF = None
        self.HRFprime = None
        self.HRFbis = None

        #results
        self.beta_hat = np.empty(0)
        self.result_res_er = np.empty(0)
        self.result_tstat = np.empty(0)
        self.result_pval = np.empty(0)

    def create_design_matrix(self, paradigm=None, noise=None, first_derivative=True, sec_derivative=True, doPlot = False):
        #HRF parameters
        T0 = 0
        n = 4
        lamda = 2

        n_noise_reg = 0
        nTpts = paradigm.shape[0]

        if noise is not None:
            noise = np.atleast_2d(noise).T
            if noise.shape[0] != nTpts:
                raise ValueError("Wrong noise shape {}".format(noise.shape))
            n_noise_reg = noise.shape[1]

        if self.HRF is None:
            t = np.arange(0, 25, 1 / self.fs)
            self.HRF = np.power((t-T0), (n-1))*np.exp(-(t-T0)/lamda)/((pow(lamda,n)*factorial(n-1)))
        else:
            t = np.arange(0, self.HRF.shape[0]/self.fs, 1 / self.fs)

        n_derivatives = 0
        if first_derivative or sec_derivative:
            dx = t[1] - t[0]
            self.HRFprime = np.gradient(self.HRF, dx)
            self.HRFbis = np.gradient(self.HRFprime, dx)
            n_derivatives += 1
        if sec_derivative:
            n_derivatives += 1
        self.design_matrix = np.empty([nTpts, 3 + n_noise_reg + n_derivatives])
        self.design_matrix[:, 0] = np.convolve(paradigm, self.HRF)[0:nTpts]
        if first_derivative or sec_derivative:
            self.design_matrix[:, 1] = np.convolve(paradigm, self.HRFprime)[0:nTpts]
        if sec_derivative:
            self.design_matrix[:, 2] = np.convolve(paradigm, self.HRFbis)[0:nTpts]
        self.design_matrix[:, 1 + n_derivatives] = np.ones([nTpts])
        self.design_matrix[:, 2 + n_derivatives] = np.linspace(0, 1, nTpts)
        self.design_matrix[:, 0] = self.design_matrix[:, 0] / max(self.design_matrix[:, 0])

        if noise is not None:
            for i in range(noise.shape[1]):
                self.design_matrix[:, 3 + n_derivatives + i] = (noise[:, i] - np.mean(noise[:, i])) / np.sqrt(np.var(noise[:, i]))
        return

    def fit(self, time_series, do_whitening=False):
        if np.linalg.matrix_rank(self.design_matrix) != self.design_matrix.shape[1]:
            raise Warning("Design matrix rank is too low: {}".format(np.linalg.matrix_rank(self.design_matrix)))

        if do_whitening:
            Y = self.prewhiten(time_series)
        else:
            Y = np.copy(time_series)
        Y = np.atleast_2d(Y)

        nTpts = Y.shape[1]
        c = [0] * self.design_matrix.shape[1]
        c[0] = 1
        c = np.array(c, ndmin=2).T

        time_e = None
        #degress of freedom est:
        for i, val in enumerate(acf(Y[0, :], nlags=int(nTpts/3))):
            if val < 1/np.exp(1):
                time_e = i
                Te_val = val
                break
        if time_e is None:
            df_error = 100
        else:
            df_error = int(nTpts/(2*time_e))
        print("Deg of freedom:" + str(df_error))

        beta, res_er, *_ = scipy.linalg.lstsq(self.design_matrix, Y.T)
        #residue = Y.T - self.design_matrix @ beta
        #res_er = np.diag((residue @ residue.T))   #returned by scipy.linalg.lstsq
        self.result_beta = beta.T

        c_b_cov = c.T @ np.linalg.pinv(self.design_matrix.T @ self.design_matrix) @ c
        c_b_cov = c_b_cov[0][0]
        sigma2_er = res_er / df_error
        self.result_sigma2 = sigma2_er
        t_stat = (c.T @ beta) / np.sqrt(sigma2_er * c_b_cov)
        self.result_tstat = t_stat[0]

        t_dist = stats.t(df_error)
        self.result_pval = 1 - t_dist.cdf(self.result_tstat)
#        self.result_pval = np.empty(data.shape[0])
#        for i in range(data.shape[0]):
#            self.result_pval[i] = 1 - t_dist.cdf(self.result_tstat[i])
        return self.result_beta, self.result_res_er, self.result_tstat

    def old_fit(self, input, do_whitening=False):
        if np.linalg.matrix_rank(self.design_matrix) != self.design_matrix.shape[1]:
            raise Warning("Design matrix rank is too low: {}".format(np.linalg.matrix_rank(self.design_matrix)))

        if do_whitening:
            data = self.prewhiten(input)
        else:
            data = np.copy(input)
        data = np.atleast_2d(data)
        self.result_beta = np.empty((data.shape[0], self.design_matrix.shape[1]))
        self.result_var_e = np.empty(data.shape[0])
        self.result_tstat = np.empty(data.shape[0])
        self.result_pval = np.empty(data.shape[0])

        nTpts = data.shape[1]
        for i in range(data.shape[0]):
            self.beta_hat = np.linalg.inv(self.design_matrix.T.dot(self.design_matrix)).dot(self.design_matrix.T).dot(data[i, :])
            Var_e = (data[i, :] - self.design_matrix.dot(self.beta_hat)).T.dot((data[i, :] -
                        self.design_matrix.dot(self.beta_hat)) / (nTpts - np.linalg.matrix_rank(self.design_matrix)))
            print("Channel: \t", i)
            print("Beta: \t", self.beta_hat)
            c = [0] * self.design_matrix.shape[1]
            c[0] = 1
            c = np.array(c, ndmin=2).T
            t_stat = c.T * self.beta_hat / np.sqrt(Var_e * c.T.dot(np.linalg.inv(self.design_matrix.T.dot(self.design_matrix)).dot(c)))
            print("t stat: \t", t_stat)
            self.result_beta[i, :] = self.beta_hat
            self.result_var_e[i] = Var_e
            self.result_tstat[i] = t_stat[0][0]
            df_error = nTpts - np.linalg.matrix_rank(self.design_matrix)
            t_dist = stats.t(df_error)
            self.result_pval[i] = 1 - t_dist.cdf(t_stat[0][0])

        return self.result_beta, self.result_var_e, self.result_tstat

    def fit_nilearn(self, input, do_whitening=False, noise_model='ar1'):
        if do_whitening:
            data = self.prewhiten(input)
        else:
            data = np.copy(input)
        labels, glm_estimates = run_glm(data.T, self.design_matrix, noise_model=noise_model)
        results = glm_estimates[labels[0]]

        print("Estimate:", glm_estimates[labels[0]].theta[0], "  MSE:", glm_estimates[labels[0]].MSE,
              "  Error (uM):", 1e6 * (glm_estimates[labels[0]].theta[0] - 4 * 1e-6))

        c = [-1] * self.design_matrix.shape[1]
        c[0] = 1
        contrast = np.array(c, ndmin=2)


        return _cc(labels, results, contrast, contrast_type=None)

    def show_design(self, savepath=None):
        t = np.arange(0, self.HRF.shape[0] / self.fs, 1 / self.fs)
        plt.figure("HRF")
        plt.title("Hemodynamic response function")
        plt.plot(t, self.HRF)
        if self.HRFprime is not None:
            plt.plot(t, self.HRFprime, color="red")
        if self.HRFbis is not None:
            plt.plot(t, self.HRFbis, color="green")
        plt.grid()
        plt.xlabel("time [s]")

        plt.figure(figsize=(16, 9))
        plt.title("Design matrix: time courses")

        for i in range(self.design_matrix.shape[1]):
            ax = plt.subplot(self.design_matrix.shape[1], 1, i+1)
            plt.plot(np.linspace(0, self.design_matrix.shape[0] / self.fs, self.design_matrix.shape[0]), self.design_matrix[:, i], linewidth=2)
            plt.grid()
            if i < self.design_matrix.shape[1] - 1:
                plt.xticks([])
        plt.xlabel("time [s]")
        if savepath is not None:
            plt.savefig(savepath + "_design_m_time.png", dpi=300)
            plt.close()

        plt.figure("Design matrix")
        plt.imshow(self.design_matrix, cmap='Purples')
        plt.xlabel("Regressors")
        plt.ylabel("Samples")
        plt.axis("auto")
        plt.xticks([])
        if savepath is not None:
            plt.savefig(savepath + "_design_m_square.png", dpi=300)
            plt.close()

        if savepath is None:
            plt.show()
        else:
            plt.close()
        return

    def show_results_bar(self, order=np.empty(0), mode='t-test', colors=[], labels=np.empty(0), color_labels={}):
        """
        Plot results in a form of a bar plot, with possible reordering of the variables and putting them into groups.
        :param order: order in which plot the variables. np array with indices, less or equal in length as the number of variables
        :param mode: What to plot: "beta" or "t-test" (default)
        :param colors: list of colors (one for each variable)
        :param labels: text labels for each variable (np.array of str)
        :param color_labels: labels for each color, used to provide a legend for the plot (dictionary)
        :return:
        """
        if order.shape[0] == 0:
            order = np.array(list(range(self.result_beta.shape[0]))).astype(int)
        if labels.shape[0] == 0:
            labels = np.array(list(range(self.result_beta.shape[0]))).astype(str)
        if len(colors) == 0:
            colors = ["xkcd:deep blue"] * self.result_beta.shape[0]
        if len(color_labels) == 0:
            color_labels = {}
            for col in colors:
                color_labels[col] = ""

        x_pos = [i for i, _ in enumerate(self.result_beta[order, 1])]
        if mode == "beta":
            plt.figure("GLM results (Beta values)", figsize=(16, 9))
            plt.bar(x_pos, self.result_beta[order, 0], color=colors)
            plt.ylabel("Beta values")
        if mode == "t-test":
            plt.figure("T-test results", figsize=(16, 9))
            plt.bar(x_pos, self.result_tstat[order], color=colors)
            plt.ylabel("T-test scores")
            
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in color_labels.values()]
        plt.legend(markers, color_labels.keys(), numpoints=1, fontsize='x-small')
        plt.xticks(x_pos, labels[order], rotation='vertical')
        plt.yticks()

    def sklearn_vif(self):
        vifs = np.empty(self.design_matrix.shape[1])
        tolerances = np.empty(self.design_matrix.shape[1])
        for i in range(0, self.design_matrix.shape[1]):
            X = np.atleast_2d(self.design_matrix[:, i]).T
            y = np.delete(self.design_matrix, i, 1)
            r_squared = LinearRegression().fit(X, y).score(X, y)

            vif = 1/(1 - r_squared)
            vifs[i] = vif

            tolerance = 1 - r_squared
            tolerances[i] = tolerance
        return vifs, tolerances

    def fit_sklearn(self, data):
        result = LinearRegression().fit(data.T, self.design_matrix)
        coef_vals = result.coef_
        return coef_vals

    def prewhiten(self, input):
        data = np.copy(input)
        # Centering the columns (ie the variables)
        data_mean = data.mean(axis=-1)
        data -= data_mean[:, np.newaxis]
        u, d, _ = np.linalg.svd(data, full_matrices=False)
        del _
        K = (u / d).T[:data.shape[0]]  # see (6.33) p.140
        del u, d
        whitened_data = np.dot(K, data)
        whitened_data *= np.sqrt(data.shape[1])
        return whitened_data

    def prewhiten_from_residual(self):
        whitened_data = np.zeros(0)
        return whitened_data


def compute_residual(time_series, design_matrix):
    if np.linalg.matrix_rank(design_matrix) != design_matrix.shape[1]:
        raise Warning("Design matrix rank is too low: {}".format(np.linalg.matrix_rank(design_matrix)))

    Y = time_series
    beta, res_er, *_ = scipy.linalg.lstsq(design_matrix, Y.T)
    residue = Y.T - design_matrix @ beta
    return residue

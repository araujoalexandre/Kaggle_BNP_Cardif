"""
__file__

    LogisticRegression.py

__description__

    Logistic Regression with newton-cg method
    
__author__

    Araujo Alexandre < alexandre.araujo@wavestone.com >

"""
import numpy as np


class LogisticRegressionCustom:

    def __init__(self, fit_intercept=True, maxiter=100, tol=1e-4):

        self.fit_intercept=fit_intercept
        self.maxiter = maxiter
        self.tol = tol
        self.coefs_ = None
        self.n_iter_ = None


    def _sigmoid(self, x):
        """
            compute the sigmoid of x
        """
        return 1. / (1 + np.exp(-x))


    def _conjugate_gradient(self, grad, hess, maxiter=100, tol=1e-4):
        """
            Solve iteratively the linear system 'A . x = b'
            with a conjugate gradient descent.

            A = hess, b = - grad
            
            https://en.wikipedia.org/wiki/Conjugate_gradient_method
        """

        xi = np.zeros(len(grad))
        ri = grad
        pi = ri

        dri0 = ri @ ri

        i = 0
        while i <= maxiter:

            Ap = (hess @ pi) + pi

            alphai = dri0 / (pi @ Ap)

            xi = xi + alphai * pi
            ri = ri - alphai * Ap

            if np.sum(np.abs(ri)) <= tol:
                break

            dri1 = ri @ ri
            betai = dri1 / dri0 
            pi = ri + betai * pi

            i += 1
            dri0 = dri1

        return xi


    def _newton_cg(self, X, y):
        """
            train model on X

            INPUT:
                X: ndarray, array
                y: ndarray, array
        """

        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.column_stack((ones, X))

        m, n = X.shape
        beta = np.zeros(n)

        y[y == 0] = -1
        k = 0

        while k < self.maxiter:

            # calculate the gradient    
            yz = y * (X @ beta)
            z = self._sigmoid(yz)
            z0 = (z - 1) * y

            grad = (X.T @ z0) + beta
            hess = X.T @ ((z * (1 - z)).reshape(-1, 1) * X)

            absgrad = np.abs(grad)
            if np.max(absgrad) < self.tol:
                break

            xi = self._conjugate_gradient(grad, hess)
            beta = beta - xi

            k += 1

        return beta, k


    def fit(self, X, y, method='newton_cg'):
        """
            train model on X

            INPUT:
                X: ndarray, array
                y: ndarray, array
        """
        # we modify y so we make a copy
        self.y = y[:]
        if method == 'newton_cg':
            self.coefs_, self.n_iter_ = self._newton_cg(X, self.y)


    def predict_proba(self, X):
        """
            compute proba of X base on self.coefs_
        """

        # if not isinstance(self.coefs_, list):
        #     raise("You need to call fit metod first")
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.column_stack((ones, X))
            y_hat = self._sigmoid(X @ self.coefs_).flatten()
        else:
            y_hat = self._sigmoid(X @ self.coefs_).flatten()

        return np.column_stack((1 - y_hat, y_hat))
import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        dist = label - pred
        square_dist = np.square(dist)
        mse = np.mean(square_dist)
        rmse = np.sqrt(mse)
        return rmse

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """
        column = np.reshape(x, (x.shape[0], 1))
        X = np.tile(column, degree + 1)
        powers = np.array(range(degree + 1))
        feat = np.power(X, powers)
        return feat

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        prediction = np.dot(xtest, weight)
        return prediction


    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        pseudo_inverse = np.linalg.pinv(xtrain)
        weight = np.dot(pseudo_inverse, ytrain)
        return weight

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            dif = ytrain - np.dot(xtrain, weight)
            Sum = np.dot(xtrain.transpose(), dif)
            weight = weight + ((learning_rate / xtrain.shape[0]) * Sum)
        return weight


    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            n = np.random.randint(0, xtrain.shape[0])
            x = xtrain[n, :]
            y = ytrain[n][0]
            dif = y - np.dot(x, weight)
            dif = dif[0]
            second = learning_rate * x.transpose() * dif
            weight = weight + second
        return weight


    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        z = xtrain
        z_T = z.transpose()
        I = np.zeros((xtrain.shape[1], xtrain.shape[1]))
        np.fill_diagonal(I, c_lambda)
        I[0, 0] = 0
        new_z = np.dot(z_T, z)
        tobeinv = new_z + I
        inv = np.linalg.inv(tobeinv)
        pseudo = np.dot(inv, z_T)
        weight = np.dot(pseudo, ytrain)
        return weight

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        raise NotImplementedError

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        raise NotImplementedError

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        """
        Args:
            X : NxD numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : Nx1 numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: Float average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """
        y = y.flatten()
        rmseArr = []
        N = X.shape[0]
        subset_size = round(N / kfold)
        j = 0
        for i in range(kfold):
            if j + subset_size >= N:
                x_test = X[j:, :]
                x_train = np.delete(X, np.s_[j:], axis=0)
                y_test = y[j:]
                y_train = np.delete(y, np.s_[j:])
            else:
                x_test = X[j:j + subset_size, :]
                x_train = np.delete(X, np.s_[j:j + subset_size], axis=0)
                y_test = y[j:j + subset_size]
                y_train = np.delete(y, np.s_[j:j + subset_size])
            weight = Regression.ridge_fit_closed(self, x_train, y_train, c_lambda)
            prediction = Regression.predict(self, x_test, weight)
            rmse = Regression.rmse(self, prediction, y_test)
            rmseArr.append(rmse)
            j += subset_size
        meanError = np.mean(rmseArr)
        return meanError
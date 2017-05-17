def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta 
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss
    #TODO
    residual = np.dot(X,theta)-y
    residual2 = residual**2
    loss = np.sum(residual2)/(2*len(theta))
    return loss

def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    grad = np.zeros(X.shape[1])
    grad = np.dot(X.T,np.dot(X,theta)-y)/X.shape[0]
    #for i in range(X.shape[1]):
    #    grad[i]=sum((np.dot(X,theta)-y)*X[:,i])/X.shape[0]
    return grad

def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    eps_matrix = np.diag(np.ones(num_features))
    print (eps_matrix)
    for i in range(num_features):
        print(eps_matrix[i])
        J_diff_i = compute_square_loss(X, y, theta+epsilon*eps_matrix[i])-compute_square_loss(X, y, theta-epsilon*eps_matrix[i])
        grad_i = J_diff_i/(2*epsilon)
        approx_grad[i] = grad_i
    
    sum_square=np.sum((true_gradient-approx_grad)**2)
    eclid = np.sqrt(sum_square)
    if eclid<=tolerance:
        return True
    else:
        return False
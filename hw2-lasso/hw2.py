import numpy as np
import pandas as pd
from scipy.optimize import minimize

########################## 1.1 DataSet Construction ###############################
def generate_dataset(num_instance, dim_feature, theta, bias=False):

    X = np.random.rand (num_instance, dim_feature)
    epsilon = 0.1 * np.random.randn(num_instance) # Standard normal distribution
    if bias:
        X = np.hstack((X, np.ones((X.shape[0], 1)))) #Add bias term
    y = X.dot(theta) + epsilon # Generate y
    X_train, X_validation, X_test = X[:80],X[80:100],X[100:150]
    y_train, y_validation, y_test = y[:80],y[80:100],y[100:150]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test

############# 1.2 Experiments with Ridge Regression   #######################

class ridge_regression :   
    def _init_ ():
        self.theta = ''
        
    def fit(self, X, y, lambda_reg):
        m = X.shape[0]
        def obj(theta):
            return ((np.linalg.norm(np.dot(X,theta) - y))**2)/(2*m) + lambda_reg*(np.linalg.norm(theta))**2
        
        theta_init = np.zeros(X.shape[1])
        theta_opt = minimize(obj, theta_init).x        
        self.theta = theta_opt          
        return theta_opt
  
    def compute_loss(self, theta, X, y):
        m = X.shape[0]
        loss = (np.linalg.norm(np.dot(X,theta) - y))**2/(2 * m)
        return loss
    
    def find_best_lambda(self, X_train,y_train, X_validation, y_validation, lambdas):
        result = {}
        for Lambda in lambdas:
            theta = self.fit(X_train, y_train, Lambda)
            loss = self.compute_loss(theta, X_validation, y_validation)
            result[Lambda] = loss
        best_lambda = min(result, key=lambda k: result[k]) 
        return result, best_lambda

    def get_parameter(self):
        return self.theta
        
def print_sparsity(Lambda, theta, theta_true, tol=10**-4, loss= False):
    ''' Compute the sparsity of theta_estimate'''
    
    df_comp= pd.DataFrame([theta,theta_true]).T
    df_comp['ans1']= df_comp[df_comp[1]==0][0].apply(lambda x : abs(x)<=tol) # true value ==0, estimate < tol
    df_comp['ans2']=df_comp[df_comp[1]!=0][0] !=0  # true value !=0, estimate !=0
    spar1 = len(df_comp[df_comp['ans1']==True])
    spar2 = len(df_comp[df_comp['ans2']==True])
    if loss:     
        print 'lambda = %f, %d , %d , loss=%f' %(Lambda, spar1,spar2,loss)
    else:
        print 'lambda = %f, %d , %d '%(Lambda, spar1,spar2)
        

###############  2. Lasso: Shooting algorithm #######################
def compute_loss(theta, X, y):
    m = X.shape[0]
    loss = (np.linalg.norm(np.dot(X,theta) - y))**2/(2 * m)
    return loss

def soft(a, delta):  
    soft = np.sign(a) * max(abs(a) - delta, 0)    
    return soft
          
def compute_obj_loss(theta, X, y , lambda_reg):
    loss = compute_loss(theta, X,y)
    loss += lambda_reg * np.linalg.norm(theta, ord=1)
    return loss    

def shooting_algorithm(theta_init, X, y, lambda_reg, max_iter=10000,tol=10**-8):
    '''
    Coordinate descent for lasso, using vector presentation
    '''
    theta = theta_init # Initilization
    diff = 1    
    num_iter = 1
    while ( num_iter < max_iter and diff > tol):  
        loss_previous = compute_obj_loss(theta, X, y , lambda_reg)
        
        for j in range(theta_init.shape[0]):           
            a = X[:,j].dot(X[:,j])
            c = X[:,j].dot(y - X.dot(theta) + theta[j] * X[:,j])
            para_1, para_2 = c/a,lambda_reg/a
            theta[j] = soft(para_1, para_2)
        loss_current = compute_obj_loss(theta, X, y , lambda_reg)
        diff = abs(loss_previous - loss_current)
        num_iter += 1
              
    return theta

def shooting_algorithm_slow(theta_init, X, y, lambda_reg, max_iter=1000 ,tol=10**-8):
    '''
    Coordinate descent for lasso, slow version
    '''
    theta = theta_init # Initilization
    diff = 1    
    num_iter = 1
    while ( num_iter < max_iter and diff > tol ):  
        loss_previous = compute_obj_loss(theta, X, y , lambda_reg)
        
        for j in range(theta_init.shape[0]):           
            a,c =0,0 # initilization
            for i in range(X.shape[0]):
                a += X[i,j] * X[i,j]
                c += X[i,j] * (y[i] - X[i].dot(theta) + theta[j] * X[i,j])
            para_1, para_2 = c/a,lambda_reg/a
            theta[j] = soft(para_1, para_2)
        loss_current = compute_obj_loss(theta, X, y , lambda_reg)
        diff = abs(loss_previous - loss_current)
        num_iter += 1
               
    return theta  


def homotopy(theta_init, X_tr, y_tr,X_va,y_va, ceiling, type='fast'):
    '''regularization path approach'''
    theta = theta_init # Initilization
    loss_hist = {}
    theta_hist = {}
    Lambda = ceiling
    while Lambda >10**-5:        
        if type=='slow':
            theta = shooting_algorithm_slow(theta, X_tr, y_tr, Lambda)
        if type=='fast':
            theta = shooting_algorithm(theta, X_tr, y_tr, Lambda)

        loss_hist[Lambda] = compute_loss(theta, X_va, y_va) 
        theta_hist[Lambda] = theta
        Lambda *=0.1
    return theta_hist , loss_hist
        
        
############ 2.2  Projected SGD via Variable Splitting ############
def GD_lasso(X, y, alpha=0.01, lambda_reg =0.001, max_iter=10000 , tol=0 ):
    ''' Implement normal gradient descent'''
    
    def compute_stochastic_gradient(X, y, theta_1, theta_2, lambda_reg):
        num_features = X.shape[1]
        predict = X.dot(theta_1 - theta_2)
        error = y - predict

        grad_1 =  - X.T.dot(error)  + 2*lambda_reg * np.ones(num_features)
        grad_2 =   X.T.dot(error)  + 2*lambda_reg * np.ones(num_features)
        return grad_1, grad_2

    def floor(array):
        for i, e in enumerate(array):
            array[i] = max(e, 0)
        return array
    
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_1 = np.zeros(num_features) #Initialize theta+
    theta_2 = np.zeros(num_features) #Initialize theta-
    loss_hist =np.zeros(max_iter)
    
    for i in np.arange(max_iter):  
        theta_previous = theta_1 - theta_2
        grad_1, grad_2 = compute_stochastic_gradient(X, y, theta_1,theta_2,lambda_reg)
        theta_1 -= alpha * grad_1        
        theta_2 -= alpha * grad_2       
        theta_1 = floor(theta_1)
        theta_2 = floor(theta_2)            
        theta = theta_1 - theta_2            
        loss_hist[i] = compute_loss(theta, X, y)       
       
        diff = np.linalg.norm(theta_previous - theta,ord=1)
        
        if diff < tol:
            print 'Converged after %d iteration' %i
            break                    
    return theta, loss_hist
    
    
def SGD_lasso(X, y, alpha='1/t', lambda_reg =0.001, max_iter=1000 , tol=0 ):   
    ''' Implement stochastic gradient descent '''
    
    def compute_stochastic_gradient(x_i, y_i, theta_1, theta_2, lambda_reg):
        num_features = len(x_i)
        predict = x_i.dot(theta_1 - theta_2)
        error = y_i - predict

        grad_1 =  x_i * -1 * error  + 2*lambda_reg * np.ones(num_features)
        grad_2 =  x_i * error  + 2*lambda_reg * np.ones(num_features)
        return grad_1, grad_2

    def floor(array):
        for i, e in enumerate(array):
            array[i] = max(e, 0)
        return array
    
    from random import sample
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_1 = np.zeros(num_features) #Initialize theta+
    theta_2 = np.zeros(num_features) #Initialize theta-    
    loss_hist =np.zeros(max_iter)
    
    for i in np.arange(max_iter): 
        theta_previous = theta_1 - theta_2
        index = np.random.permutation(num_instances)
        for it,ix in enumerate(index):
            x_i, y_i = X[ix],y[ix]

            if isinstance(alpha, float):
                step_size = alpha
            elif alpha == '1/t':
                step_size = 1.0/(num_instances*i+it+1)
            else:
                step_size = 1.0/np.sqrt(num_instances*i+it+1)
                               
            grad_1, grad_2 = compute_stochastic_gradient(x_i, y_i, theta_1,theta_2,lambda_reg)
            theta_1 -= step_size * grad_1        
            theta_2 -= step_size * grad_2       
            theta_1 = floor(theta_1)
            theta_2 = floor(theta_2)
           
        theta = theta_1 - theta_2            
        loss_hist[i] = compute_loss(theta, X, y)      
        
        diff =   diff = np.linalg.norm(theta_previous - theta,ord=1)
        
        if diff < tol:
            print 'Converged after %d iteration' %i
            break
            
        
    return theta, loss_hist


######## main()#########
num_instance = 150
dim_feature = 75
theta_true = np.hstack((np.sign(np.random.randn(10))*10,np.zeros(dim_feature - 10)))
X_tr, X_va, X_te, y_tr, y_va, y_te = generate_dataset(num_instance, dim_feature ,theta_true)

########### Ridge Experiment ############
lambda_range = [10**x  for x in range(-5,6)]
theta_ridge = {}
loss_ridge = {}
for Lambda in lambda_range:
    ridge = ridge_regression ()
    ridge.fit(X_tr, y_tr, lambda_reg=Lambda) 
    theta = ridge.get_parameter()
    theta_ridge[Lambda] = theta
    loss_ridge[Lambda] = ridge.compute_loss(theta, X_va,y_va)
    
for Lambda in sorted(theta_ridge.iterkeys()):   
    theta,loss = theta_ridge[Lambda], loss_ridge[Lambda]
    print_sparsity (Lambda, theta, theta_true, tol=10**-3 ,loss=loss)
       
## Double check Use sklearn.linear model.Ridge.
best_lambda = 10**-5
from sklearn.linear_model import Ridge
clf = Ridge(alpha=best_lambda, fit_intercept=False)
clf.fit(X_tr, y_tr)
print 'Sklearn Ridge Double check: %d out of 75 features are ZERO(10^-4)' %len(clf.coef_[abs(clf.coef_)<10**-4])

############# Lasso :  shooting and homotopy ############
from time import clock
lambda_max = 2 * X_tr.T.dot(y_tr).max() # Calculate lambda_max

loss_shoot_slow = {}
start = clock()
for Lambda in lambda_range:
    theta_init = np.zeros(theta_true.shape[0])
    #theta_init = np.linalg.inv(X_tr.T.dot(X_tr)+np.diag(np.ones(dim_feature)*Lambda)).dot(X_tr.T).dot(y_tr)
    theta = shooting_algorithm_slow(theta_init, X_tr, y_tr, Lambda)
    loss = compute_loss(theta, X_va, y_va)
    loss_shoot_slow[Lambda] = loss
    
end = clock()
time_shooting = end - start
print 'Shooting algorithm(Slow): %f seconds' %time_shooting

loss_shoot_fast = {}
theta_shoot = {}   # theta_lasso
start = clock()
for Lambda in lambda_range:
    theta_init = np.zeros(theta_true.shape[0])
    #theta_init = np.linalg.inv(X_tr.T.dot(X_tr)+np.diag(np.ones(dim_feature)*Lambda)).dot(X_tr.T).dot(y_tr)
    theta = shooting_algorithm(theta_init, X_tr, y_tr, Lambda)  
    loss = compute_loss(theta, X_va, y_va)
    loss_shoot_fast[Lambda] = loss
    theta_shoot[Lambda] = theta 
    
end = clock()
time_shooting = end - start
print 'Shooting algorithm(fast): %f seconds' %time_shooting

theta_init = np.zeros(theta_true.shape[0])
start = clock()
loss_homotopy_slow = homotopy(theta_init, X_tr, y_tr,X_va,y_va,ceiling=10**4, type='slow')[1]
end = clock()
time_homotopy = end - start
print 'Homotopy algorithm(Slow): %f seconds' %(time_homotopy)

start = clock()
loss_homotopy_fast = homotopy(theta_init, X_tr, y_tr,X_va,y_va,ceiling=10**4,type='fast')[1]
end = clock()
time_homotopy = end - start
print 'Homotopy algorithm(fast): %f seconds' %time_homotopy

import matplotlib.pyplot as plt
%matplotlib inline
df = pd.DataFrame.from_dict(loss_ridge, orient='index')
df = df.sort_index()
plt.plot(lambda_range, df[0], label='ridge')

df = pd.DataFrame.from_dict(loss_shoot_fast, orient='index')
df = df.sort_index()
plt.plot(lambda_range, df[0], label='shooting')
'''
df = pd.DataFrame.from_dict(loss_homotopy_fast, orient='index')
df = df.sort_index()
plt.plot(lambda_range, df[0], label='homotopy')
'''
plt.xscale('log')
plt.yscale('log')
plt.xlabel('lambda')
plt.ylabel('validation error')
plt.legend(loc='best')

import pandas as pd
print 'lambda = _ , __ features(true value 0) is estimated as 0, __ features(true value non-zero) is estimated as non-zero.'
for Lambda in sorted(theta_shoot.iterkeys())[:-1]:
    theta = theta_shoot[Lambda]
    print_sparsity(Lambda, theta, theta_true)

###########  GD and SGD #########

result_SGD ={}
start = clock()
for Lambda in [0.01]:
    theta, loss_hist= SGD_lasso(X_tr, y_tr,alpha =0.0001,lambda_reg=Lambda,max_iter=5000)
    print_sparsity(Lambda, theta, theta_true)
    loss = compute_loss(theta, X_va, y_va)
    result_SGD[Lambda] = loss
end = clock()
time_SGD = end - start
print 'SGD  algorithm: %f seconds' %time_SGD

result_GD ={}
start = clock()
for Lambda in lambda_range:
    theta, loss_hist = GD_lasso(X_tr, y_tr,alpha =0.0005,lambda_reg=Lambda, max_iter =2000)
    print_sparsity(Lambda, theta, theta_true)
    loss = compute_loss(theta, X_va, y_va)
    result_GD[Lambda] = loss

end = clock()
time_GD = end - start
print 'GD  algorithm: %f seconds' %time_GD

df = pd.DataFrame.from_dict(result_SGD, orient='index')
df = df.sort_index()
plt.plot(lambda_range, df[0], label='SGD')
'''
df = pd.DataFrame.from_dict(result_GD, orient='index')
df = df.sort_index()
plt.plot(lambda_range, df[0], label='GD')
'''
df = pd.DataFrame.from_dict(loss_shoot_fast, orient='index')
df = df.sort_index()
plt.plot(lambda_range, df[0], label='shooting')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('lambda')
plt.ylabel('validation error')
plt.legend(loc='best')


    
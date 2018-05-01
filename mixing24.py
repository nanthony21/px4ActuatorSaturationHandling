
# coding: utf-8

# ### Created on Thu Mar 22 17:18:52 2018
# 
# ### @author: nanthony


import random
import numpy as np  
pinv = np.linalg.pinv


'''from https://list.coin-or.org/pipermail/ipopt/2013-February/003241.html
Given a QP of the form min x  0.5*x’*H*x + f’*x

 St.             rl <= A*x<= ru

                 lb <= x<= ub

Then the Hessian is:
>
Hess = 0.5*(H + H’)


 And objective gradient is:

 Grad = 0.5*(H + H’)*x + f



 And the Jacobian is simply the A matrix from your linear constraints. Note
 for linear constraints the Hessian of the Lagrangian only contains
 quadratic terms from the objective (as Hess above), and no constraints
 feature (given they are all linear). See Ogata Discrete Time Control
 Systems for more details on the above.****
'''

yaw = np.matrix([[0,0,1,0]],dtype=np.float)
pitch = np.matrix([[0,1,0,0]],dtype=np.float)
roll = np.matrix([[1,0,0,0]],dtype=np.float)
thrust = np.matrix([[0,0,0,1]],dtype=np.float)
rand = np.matrix([[random.uniform(-1,1) for i in range(4)]]) #r,p,y,t
sat = np.matrix([[1,1,1,1]],dtype=np.float)




g = np.zeros((1,4)) #contains the actuator outputs

B = np.matrix([ #The transfer matrix giving scalers for how much each command input should affect each rotor
                        [ -0.707107,  0.707107,  1.000000,  1.000000 ], #Columns are roll, pitch, yaw, thrust
                        [ 0.707107, -0.707107,  1.000000,  1.000000 ], #Rows are each of the 4 rotors
                        [ 0.707107,  0.707107, -1.000000,  1.000000 ],
                        [ -0.707107, -0.707107, -1.000000,  1.000000 ],
                    ])

H = np.diag([20,20,.1,1])

x = np.array([.5,0,0,1],dtype=np.float).squeeze()

g = B * np.matrix(x).transpose()
print("x: ")
print(x,'\n')
print("g: ")
print(g,'\n')



import pyipopt




nvar = 4 #we have four variables (roll,pitch,yaw,thrust)
x_L = np.array([-1,-1,-1,-1],dtype=np.float) #The variables may not drop below -1. For some reason ipopt doesn't recognize these if they are one.
x_U = np.array([1,1,1,1],dtype=np.float) #The variables may not go above 1
ncon = 4 #We have 4 constraints (the actuator outputs)
g_L = np.array([-1,-1,-1,-1],dtype=np.float) #the actuator outputs must lie between -1 and 1
g_U = np.array([1,1,1,1],dtype=np.float)

def eval_f(X):
'''
we want to minimize the sum of squares of the original command vector minus the calculated command vector. 
'''
    xx = np.matrix(x - X)
    res = xx * H * xx.transpose()
    res = np.array(res,dtype=np.float).squeeze()
    return res

def eval_grad_f(X): #This gives the gradient of the objective function
    res = 2 * H * np.matrix(X - x).transpose()
    res = np.array(res,dtype=np.float).squeeze()
    return res

def eval_g(X): #return the constrained values (the actuator outputs)
    res = np.array(B * np.matrix(X).transpose(),dtype=np.float).squeeze()
    return res

def eval_jac_g(x,flag):
    #if flag is true then return a tuple of 2 arrays (rows,columns) giving the structure of the jacobian 
    #if flag is true return a 1d array of the values of the jacobian.
    
    #In this case the jacobian is just the B matrix
    if flag:
        return (np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]),
                np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]))
    else:
        assert len(x)==4
        res = np.array([
            B[0,0],
            B[0,1],
            B[0,2],
            B[0,3],
            B[1,0],
            B[1,1],
            B[1,2],
            B[1,3],
            B[2,0],
            B[2,1],
            B[2,2],
            B[2,3],
            B[3,0],
            B[3,1],
            B[3,2],
            B[3,3],
        ],dtype=np.float)
        #print("B: ",res)
        return res

def eval_h(x, lagrange, obj_factor, flag,user_data=None):
	if flag:
		return (np.array([0,1,2,3]),np.array([0,1,2,3]))
	else:
		res = np.array([
				2*H[0,0],
				2*H[1,1],
				2*H[2,2],
				2*H[3,3]
				],dtype=np.float)
		#print("Hess: ",res)		
		return res
    
def apply_new(x):
	return None

nnzj = 16 #This is the number of non-zero elements of the jacobian. In most cases none of them are zero.


nnzh = 4 #Number of non-zero elements of the hessian.

nlp = pyipopt.create(nvar,x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g,eval_h)

newx, zl, zu, constraint_multipliers, obj, status = nlp.solve(x)
nlp.close()


print("Solution of the primal variables, x")
print(newx)

print("New Outputs")
print(eval_g(newx))

print("Solution of the bound multipliers, z_L and z_U")
print(zl)
print(zu)

print("Solution of the constraint multipliers, lambda")
print(constraint_multipliers)

print("Objective value")
print("f(x*) = {}".format(obj))





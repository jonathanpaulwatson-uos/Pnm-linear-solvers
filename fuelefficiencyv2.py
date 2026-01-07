#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 10:57:29 2025

@author: jonathan
"""

from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy
import numpy as np
import matplotlib.pyplot as plt
"""
Convergence Theorem
"""
def spectral_radius(G):
    eigvals=np.linalg.eigvals(G)
    absolutval=abs(eigvals)
    return max(absolutval)
def convergence_theorem(G):
    if spectral_radius(G)<1:
        return True
    else:
        return False

"""
Cholesky Solve
"""
data=sns.load_dataset("mpg")
data=data.dropna()
X=data[['cylinders','displacement','horsepower','weight','acceleration','model_year']].to_numpy()
y=data[['mpg']].to_numpy()
X=StandardScaler().fit_transform(X)
y=y-np.mean(y)
X=np.hstack([np.ones((X.shape[0],1)),X])
X_t=np.transpose(X)
A=np.dot(X_t,X)
b=np.dot(X_t,y)
def cholesky(A,b):
    c_and_lower,low=scipy.linalg.cho_factor(A)
    beta=scipy.linalg.cho_solve((c_and_lower,low),b)
    return beta
cholesky(A,b)
"""
Richardson Solve
"""
def Richardson(A,b,max_iterations=1000, tolerance=1e-6):
    n=len(b)
    x=np.zeros(n)
    x_old=np.zeros(n)
    G_richardson=np.identity(n)-A
    if convergence_theorem(G_richardson)==False:
        return 'The Convergence theorem is not satisfied and therefore the sequence will not converge'
    for k in range(max_iterations):
        x_old[:]=x
        r=b-A@x_old
        x=x_old+r
        if np.linalg.norm(x - x_old)<tolerance:
            print("Converged after",k+1,"iterations.")
            break
        if k == max_iterations:  # never hit tolerance
            print("reached max_iterations but did NOT converge")
    return x
print(Richardson(A,b))
"""
Weighted Richardson Solve
"""
def weighted_richardson(A,b,wfreq,max_iterations=1000, tolerance=1e-6):   #wfreq= what step size is wanted between different weightings
    n=len(b)
    x_old=np.zeros(n)
    m = int(round(1 / wfreq))
    weighted=np.zeros((m,n),dtype=float)
    w_vals=np.linspace(wfreq,1,num=m)
    for j, w in enumerate(w_vals):
        x=np.zeros(n)
        G_richardson=np.identity(n)-(w*A)
        if convergence_theorem(G_richardson)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations):
            x_old[:]=x
            r=b-A@x_old
            x=x_old+w*r 
            if np.linalg.norm(x - x_old)<tolerance:
                print("Converged after",k+1,"iterations.")
                break
            if k == max_iterations:  # never hit tolerance
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
        weighted[j,:]=x
    reference=cholesky(A,b)
    distances=np.linalg.norm(weighted-np.asarray(reference).reshape(-1),axis=1)
    plt.figure()
    plt.plot(w_vals,distances,marker='o')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\|x_\omega - \beta_{\mathrm{Cholesky}}\|_2$')
    plt.title('Weighted Richardson: Error vs $\omega$')
    plt.grid(True)
    plt.show()
    return w_vals,weighted
print(weighted_richardson(A, b,0.02))
"""
Jacobi Solve
"""
def jacobi(A,b,max_iterations=1000, tolerance=1e-6):
    n=len(b)
    x=np.zeros(n)
    x_old=np.zeros(n)
    Adiag=np.diagflat(np.diag(A))
    Aoffdiag=A-Adiag
    G_jacobi=np.dot(np.linalg.inv(Adiag),Aoffdiag)
    if convergence_theorem(G_jacobi)==False:
        return 'The Convergence theorem is not satisfied and therefore the sequence will not converge'
    for k in range(max_iterations):
        x_old[:]=x
        for i in range (n):
            x[i]=(b[i]-np.dot(Aoffdiag[i],x_old))/A[i,i]
        if np.linalg.norm(x - x_old)<tolerance:
            print("Converged after",k+1,"iterations.")
            break
        if k == max_iterations:  # never hit tolerance
            print("reached max_iterations but did NOT converge")
    return x
print(jacobi(A,b))
"""
Weighted Jacobi Solve
"""
def weighted_jacobi(A,b,wfreq,max_iterations=1000, tolerance=1e-6):   #wfreq= what step size is wanted between different weightings
    n=len(b)
    x_old=np.zeros(n)
    m = int(round(1 / wfreq))
    weighted=np.zeros((m,n),dtype=float)
    Adiag=np.diagflat(np.diag(A))
    Aoffdiag=A-Adiag
    w_vals=np.linspace(wfreq,1,num=m)
    iters=np.zeros(m,dtype=float)
    for j, w in enumerate(w_vals):
        x=np.zeros(n)
        G_jacobi=np.identity(n)-(w*np.dot(np.linalg.inv(Adiag),A))
        if convergence_theorem(G_jacobi)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations+1):
            x_old[:]=x
            for i in range (n):
                b_i=float(b[i,0])
                diagel=float(Adiag[i,i])
                matmult=float(np.dot(Aoffdiag[i], x_old))
                x[i]=w*(b_i-matmult)/diagel+(1-w)*x[i]
            if np.linalg.norm(x - x_old)<tolerance:
                iters[j]=k+1
                print("Converged after",k+1,"iterations.")
                break
            if k == max_iterations:# never hit tolerance
                iters[j]=1000
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
        weighted[j,:]=x
    reference=cholesky(A,b)
    distances=np.linalg.norm(weighted-np.asarray(reference).reshape(-1),axis=1)
    plt.figure()
    plt.plot(w_vals,distances,marker='o')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\|x_\omega - \beta_{\mathrm{Cholesky}}\|_2$')
    plt.title('Weighted Jacobi: Error vs $\omega$')
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.plot(w_vals,iters,marker='o')
    plt.xlabel(r'$\omega$')
    plt.ylabel('Iterations')
    plt.title('Jacobi: iterations vs $\omega$')
    plt.grid(True)
    plt.show()
    return w_vals,weighted 
print(weighted_jacobi(A, b,0.02))
"""
Gauss-Seidel Solve
"""
def Gauss_Seidel(A,b,max_iterations=1000, tolerance=1e-6):
    n=len(b)
    x=np.zeros(n)
    x_old=np.zeros(n)
    D=np.diag(np.diag(A))
    L=np.tril(A,-1) 
    U=np.triu(A,1)
    G_GS=-np.linalg.solve(D+L,U)
    if convergence_theorem(G_GS)==False:
        return 'The Convergence theorem is not satisfied and therefore the sequence will not converge'
    for k in range(max_iterations):
        x_old[:]=x
        for i in range (n):
            new_val_elements=np.dot(A[i,:i],x[:i])
            old_val_elements=np.dot(A[i,i+1:],x_old[i+1:])
            x[i]=(b[i]-new_val_elements-old_val_elements)/A[i,i]
        if np.linalg.norm(x - x_old)<tolerance:
            print("Converged after",k+1,"iterations.")
            break
        if k == max_iterations:  # never hit tolerance
            print("reached max_iterations but did NOT converge")
    return x
print(Gauss_Seidel(A,b))
"""
SOR Solve
"""
def SOR(A,b,wfreq,max_iterations=1000, tolerance=1e-6):   #wfreq= what step size is wanted between different weightings
    n=len(b)
    x_old=np.zeros(n)
    m = int(round(1 / wfreq))
    weighted=np.zeros((m,n),dtype=float)
    w_vals=np.linspace(wfreq,1,num=m)
    iters=np.zeros(m,dtype=float)
    for j, w in enumerate(w_vals):
        x=np.zeros(n)
        D=np.diag(np.diag(A))
        L=np.tril(A,-1) 
        U=np.triu(A,1)
        G_SOR=np.dot(np.linalg.inv(D+w*L),(((1-w)*D)-w*U))
        if convergence_theorem(G_SOR)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations+1):
            x_old[:]=x
            for i in range (n):
                new_val_elements=np.dot(A[i,:i],x[:i])
                old_val_elements=np.dot(A[i,i+1:],x_old[i+1:])
                b_i = float(b[i,0])
                x[i]=((1-w)*x[i])+(w*(b_i-new_val_elements-old_val_elements)/A[i,i])
            if np.linalg.norm(x - x_old)<tolerance:
                iters[j]=k+1
                print("Converged after",k+1,"iterations.")
                break
            if k == max_iterations:  # never hit tolerance
                iters[j]=1000
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
        weighted[j,:]=x
    reference=cholesky(A,b)
    distances=np.linalg.norm(weighted-np.asarray(reference).reshape(-1),axis=1)
    plt.figure()
    plt.plot(w_vals,distances,marker='o')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\|x_\omega - \beta_{\mathrm{Cholesky}}\|_2$')
    plt.title('SOR: Error vs $\omega$')
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.plot(w_vals,iters,marker='o')
    plt.xlabel(r'$\omega$')
    plt.ylabel('Iterations')
    plt.title('SOR: iterations vs $\omega$')
    plt.grid(True)
    plt.show()
    return w_vals,weighted
print(SOR(A, b,0.02))



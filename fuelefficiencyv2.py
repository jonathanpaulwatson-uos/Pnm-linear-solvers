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
np.set_printoptions(threshold=np.inf)
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
        if k == max_iterations-1:  
            print("reached max_iterations but did NOT converge")
    return x
print(Richardson(A,b))
"""
Weighted Richardson Solve
"""
def weighted_richardson(A,b,wfreq,max_iterations=1000, tolerance=1e-6):   #wfreq= what step size is wanted between different weightings
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
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
            if k == max_iterations-1:  
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
        if k == max_iterations-1:  
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
    converged = np.zeros(m, dtype=bool)
    for j, w in enumerate(w_vals):
        x=np.zeros(n)
        G_jacobi=np.identity(n)-(w*np.dot(np.linalg.inv(Adiag),A))
        if convergence_theorem(G_jacobi)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations):
            x_old[:]=x
            for i in range (n):
                b_i=float(b[i,0])
                diagel=float(Adiag[i,i])
                matmult=float(np.dot(Aoffdiag[i], x_old))
                x[i]=w*(b_i-matmult)/diagel+(1-w)*x[i]
            if np.linalg.norm(x - x_old)<tolerance:
                iters[j]=k+1
                converged[j] = True
                print("Converged after",k+1,"iterations.")
                break
            if k == max_iterations-1:
                iters[j]=max_iterations
                converged[j]= True
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
        weighted[j,:]=x
    reference=cholesky(A,b)
    distances=np.linalg.norm(weighted-np.asarray(reference).reshape(-1),axis=1)
    w_conv = w_vals[converged]
    distances_conv = distances[converged]
    iters_conv = iters[converged]
    plt.figure()
    plt.plot(w_conv,distances_conv,marker='o')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\|x_\omega - \beta_{\mathrm{Cholesky}}\|_2$')
    plt.title('Weighted Jacobi: Error vs $\omega$')
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.plot(w_conv,iters_conv,marker='o')
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
        if k == max_iterations-1:  
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
        for k in range(max_iterations):
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
            if k == max_iterations-1:  
                iters[j]=max_iterations
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
"""
Richardson for working weight parameters
"""
def weighted_richardson_suitable_weight_params(A,b,wfreq,max_iterations=1000, tolerance=1e-6):   #wfreq= what step size is wanted between different weightings
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n=len(b)
    x_old=np.zeros(n)
    m = int(round(0.001/ wfreq))
    weighted=np.zeros((m,n),dtype=float)
    w_vals=np.linspace(wfreq,0.001,num=m)
    iters=np.zeros(m,dtype=float)
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
                iters[j]=k+1
                print("Converged after",k+1,"iterations.")
                break
            if k == max_iterations-1:  
                iters[j]=max_iterations
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
    plt.figure()
    plt.plot(w_vals,iters,marker='o')
    plt.xlabel(r'$\omega$')
    plt.ylabel('Iterations')
    plt.title('Weighted Richardson: iterations vs $\omega$')
    plt.grid(True)
    plt.show()
    return w_vals,weighted
print(weighted_richardson_suitable_weight_params(A, b,0.0002))
"""
Probabilistic Richardson
"""
def probabilistic_richardson(A,b,wfreq,max_iterations=100, tolerance=1e-6):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    w_vals,x=weighted_richardson_suitable_weight_params(A,b,wfreq, max_iterations=100, tolerance=1e-6)
    n=len(b)
    sigma_old=np.identity(n)
    frobenius_norm = np.zeros(len(w_vals))
    r_var = np.zeros((len(w_vals), n, n), dtype=float)
    converged = np.zeros(len(w_vals), dtype=bool)
    for j, w in enumerate(w_vals):
        sigma=np.identity(n)
        G_richardson=np.identity(n)-(w*A)
        if convergence_theorem(G_richardson)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations):
            sigma_old[:]=sigma
            sigma=G_richardson@sigma_old@np.transpose(G_richardson)
            if np.linalg.norm(sigma - sigma_old)<tolerance:
                print("Converged after",k+1,"iterations.")
                converged[j]= True
                break
            if k == max_iterations-1:  
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
                converged[j]= True
        r_var[j, :, :] = sigma
        frobenius_norm[j]=np.linalg.norm(sigma,ord='fro')
    frob_conv = frobenius_norm[converged]
    w_conv = w_vals[converged]
    plt.figure(figsize=(10, 6))
    plt.plot(w_conv, frob_conv, marker="o")
    plt.xlabel("Weight parameter ω")
    plt.ylabel("Frobenius norm of covariance matrix")
    plt.title("Covariance Frobenius Norm for Each Richardson Weight")
    plt.grid(True)
    plt.show()
    return w_vals,x,r_var
print(probabilistic_richardson(A, b,0.0002))
w_vals_r,x_r,r_var=probabilistic_richardson(A,b,0.0002)
idx0 = np.where(np.isclose(w_vals_r, 0.0010))[0][0]
print(r_var[idx0])
print(x_r[idx0])
"""
Probabilistic Jacobi
"""
def probabilistic_jacobi(A,b,wfreq,max_iterations=100, tolerance=1e-6):
    w_vals,x=weighted_jacobi(A,b,wfreq,max_iterations=100, tolerance=1e-6)
    n=len(b)
    sigma_old=np.identity(n)
    D = np.diag(np.diag(A))
    D_inv=np.linalg.inv(D)
    j_covs = np.zeros((len(w_vals), n, n), dtype=float)
    frobenius_norm = np.zeros(len(w_vals))
    converged = np.zeros(len(w_vals), dtype=bool)
    for j, w in enumerate(w_vals):
        sigma=np.identity(n)
        G_jacobi = np.identity(n) - w * (D_inv @ A)
        if convergence_theorem(G_jacobi)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations):
            sigma_old[:]=sigma
            sigma=G_jacobi@sigma_old@np.transpose(G_jacobi)
            if np.linalg.norm(sigma - sigma_old)<tolerance:
                print("Converged after",k+1,"iterations.")
                converged[j]= True
                break
            if k == max_iterations-1:  
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
                converged[j]= True
        j_covs[j, :, :] = sigma
        frobenius_norm[j]=np.linalg.norm(sigma,ord='fro')
    frob_conv = frobenius_norm[converged]
    w_conv = w_vals[converged]
    plt.figure(figsize=(10, 6))
    plt.plot(w_conv, frob_conv, marker="o")
    plt.xlabel("Weight parameter ω")
    plt.ylabel("Frobenius norm of covariance matrix")
    plt.title("Covariance Frobenius Norm for Each Jacobi Weight")
    plt.grid(True)
    plt.show()
    return w_vals,x,j_covs
print(probabilistic_jacobi(A, b,0.02))
w_vals, x, j_covs = probabilistic_jacobi(A, b, 0.02)
idx2 = np.where(np.isclose(w_vals, 0.46))[0][0]
print(j_covs[idx2])
print(x[idx2])
"""
probabilistic SOR
"""
def probabilistic_SOR(A,b,wfreq,max_iterations=100, tolerance=1e-6):
    w_vals,x=SOR(A,b,wfreq,max_iterations=100, tolerance=1e-6)
    n=len(b)
    sigma_old=np.identity(n)
    j_covs = np.zeros((len(w_vals), n, n), dtype=float)
    frobenius_norm = np.zeros(len(w_vals))
    for j, w in enumerate(w_vals):
        sigma=np.identity(n)
        D=np.diag(np.diag(A))
        L=np.tril(A,-1) 
        U=np.triu(A,1)
        G_SOR=np.dot(np.linalg.inv(D+w*L),(((1-w)*D)-w*U))
        if convergence_theorem(G_SOR)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations):
            sigma_old[:]=sigma
            sigma=G_SOR@sigma_old@np.transpose(G_SOR)
            if np.linalg.norm(sigma - sigma_old)<tolerance:
                print("Converged after",k+1,"iterations.")
                break
            if k == max_iterations-1:  
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
        j_covs[j, :, :] = sigma
        frobenius_norm[j]=np.linalg.norm(sigma,ord='fro')
    plt.figure(figsize=(10, 6))
    plt.plot(w_vals, frobenius_norm, marker="o")
    plt.xlabel("Weight parameter ω")
    plt.ylabel("Frobenius norm of covariance matrix")
    plt.title("Covariance Frobenius Norm for Each SOR Weight")
    plt.grid(True)
    plt.show()
    return w_vals,x,j_covs
print(probabilistic_SOR(A, b,0.02))
w_vals_sor,x_sor,j_covs_sor=probabilistic_SOR(A,b,0.02)
idx3 = np.where(np.isclose(w_vals, 1.))[0][0]
print(j_covs_sor[idx3])
print(x_sor[idx3])
"""
probabilistic preconditioned richardson
"""
def probabilistic_preconditioned_richardson(A,b,precon_prior,wfreq,max_iterations=100, tolerance=1e-6):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    w_vals,x=weighted_richardson_suitable_weight_params(A,b,wfreq, max_iterations=100, tolerance=1e-6)
    n=len(b)
    sigma_old=np.identity(n)
    r_var = np.zeros((len(w_vals), n, n), dtype=float)
    frobenius_norm = np.zeros(len(w_vals))
    converged = np.zeros(len(w_vals), dtype=bool)
    for j, w in enumerate(w_vals):
        sigma=precon_prior
        G_richardson=np.identity(n)-(w*A)
        if convergence_theorem(G_richardson)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations):
            sigma_old[:]=sigma
            sigma=G_richardson@sigma_old@np.transpose(G_richardson)
            if np.linalg.norm(sigma - sigma_old)<tolerance:
                print("Converged after",k+1,"iterations.")
                converged[j]= True
                break
            if k == max_iterations-1:  
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
                converged[j]= True
        r_var[j, :, :] = sigma
        frobenius_norm[j]=np.linalg.norm(sigma,ord='fro')
    frob_conv = frobenius_norm[converged]
    w_conv = w_vals[converged]
    plt.figure(figsize=(10, 6))
    plt.plot(w_conv, frob_conv, marker="o")
    plt.xlabel("Weight parameter ω")
    plt.ylabel("Frobenius norm of covariance matrix")
    plt.title("Covariance Frobenius Norm for Each Preconditioned Richardson Weight")
    plt.grid(True)
    plt.show()
    return w_vals,x,r_var
print(probabilistic_preconditioned_richardson(A, b,np.linalg.inv(A),0.0002))
w_vals_r_precon,x_r_precon,r_var_precon=probabilistic_preconditioned_richardson(A,b,np.linalg.inv(A),0.0002)
idx0_precon = np.where(np.isclose(w_vals_r_precon, 0.0010))[0][0]
print(r_var_precon[idx0_precon])
print(x_r_precon[idx0_precon])
"""
probabilistic preconditioned jacobi
"""
def probabilistic_preconditioned_jacobi(A,b,precon_prior,wfreq,max_iterations=100, tolerance=1e-6):
    w_vals,x=weighted_jacobi(A,b,wfreq,max_iterations=100, tolerance=1e-6)
    n=len(b)
    sigma_old=np.identity(n)
    D = np.diag(np.diag(A))
    D_inv=np.linalg.inv(D)
    j_covs = np.zeros((len(w_vals), n, n), dtype=float)
    frobenius_norm = np.zeros(len(w_vals))
    converged = np.zeros(len(w_vals), dtype=bool)
    for j, w in enumerate(w_vals):
        sigma=precon_prior
        G_jacobi = np.identity(n) - w * (D_inv @ A)
        if convergence_theorem(G_jacobi)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations):
            sigma_old[:]=sigma
            sigma=G_jacobi@sigma_old@np.transpose(G_jacobi)
            if np.linalg.norm(sigma - sigma_old)<tolerance:
                print("Converged after",k+1,"iterations.")
                converged[j]= True
                break
            if k == max_iterations-1:  
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
                converged[j]= True
        j_covs[j, :, :] = sigma
        frobenius_norm[j]=np.linalg.norm(sigma,ord='fro')
    frob_conv = frobenius_norm[converged]
    w_conv = w_vals[converged]
    plt.figure(figsize=(10, 6))
    plt.plot(w_conv, frob_conv, marker="o")
    plt.xlabel("Weight parameter ω")
    plt.ylabel("Frobenius norm of covariance matrix")
    plt.title("Covariance Frobenius Norm for Each Preconditioned Jacobi Weight")
    plt.grid(True)
    plt.show()
    return w_vals,x,j_covs
print(probabilistic_preconditioned_jacobi(A, b,np.linalg.inv(A),0.02))
w_vals_precon,x_precon,j_covs_precon=probabilistic_preconditioned_jacobi(A,b,np.linalg.inv(A),0.02)
idx4 = np.where(np.isclose(w_vals, 0.46))[0][0]
print(j_covs_precon[idx4])
print(x_precon[idx4])
"""
probabilistic preconditioned SOR
"""
def probabilistic_preconditioned_SOR(A,b,precon_prior,wfreq,max_iterations=100, tolerance=1e-6):
    w_vals,x=SOR(A,b,wfreq,max_iterations=100, tolerance=1e-6)
    n=len(b)
    sigma_old=np.identity(n)
    j_covs = np.zeros((len(w_vals), n, n), dtype=float)
    frobenius_norm = np.zeros(len(w_vals))
    for j, w in enumerate(w_vals):
        sigma=precon_prior
        D=np.diag(np.diag(A))
        L=np.tril(A,-1) 
        U=np.triu(A,1)
        G_SOR=np.dot(np.linalg.inv(D+w*L),(((1-w)*D)-w*U))
        if convergence_theorem(G_SOR)==False:
            print('The Convergence theorem is not satisfied and therefore the sequence will not converge')
            continue
        for k in range(max_iterations):
            sigma_old[:]=sigma
            sigma=G_SOR@sigma_old@np.transpose(G_SOR)
            if np.linalg.norm(sigma - sigma_old)<tolerance:
                print("Converged after",k+1,"iterations.")
                break
            if k == max_iterations-1:  
                print(f"w={w:.3f}: reached max_iterations but did NOT converge")
        j_covs[j, :, :] = sigma
        frobenius_norm[j]=np.linalg.norm(sigma,ord='fro')
    plt.figure(figsize=(10, 6))
    plt.plot(w_vals, frobenius_norm, marker="o")
    plt.xlabel("Weight parameter ω")
    plt.ylabel("Frobenius norm of covariance matrix")
    plt.title("Covariance Frobenius Norm for Each Preconditioned SOR Weight")
    plt.grid(True)
    plt.show()
    return w_vals,x,j_covs
print(probabilistic_preconditioned_SOR(A, b,np.linalg.inv(A),0.02))
w_vals_sor_precon,x_sor_precon,j_covs_sor_precon=probabilistic_preconditioned_SOR(A,b,np.linalg.inv(A),0.02)
idx5 = np.where(np.isclose(w_vals, 1.))[0][0]
print(j_covs_sor_precon[idx5])
print(x_sor_precon[idx5])




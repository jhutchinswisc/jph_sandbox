import pandas as pd
import numpy as np
from numpy import isnan, matrix, concatenate, dot
from numpy.linalg import inv
import statsmodels.api as sm
from tabulate import tabulate
from rpy2.robjects import r
import pandas.rpy.common as com
import math as m

# I don't like the statsmodels OLS, so this one automatically adds the constant and successfully ignores the NA values in the data.
# This model matches R and Stata.

def ols(y,x, data, vcv = 'robust', group=None):
    ''' 
    Runs ordinary least squares regression on pandas dataframe
    with the option of robust standard errors.
    y     - label of the column
    x     - tuple of labels of covariates
    data  - pandas dataframe object
    vcv   - variance-covariance matrix.  Options: None, 'robust', and 'cluster'
    group - group variable if vcv = 'cluster' 
    '''
    df_ = data
    # Only use non-NA (breaks otherwise)
    for i in x:
        df_ = df_[(np.isnan(df_[i])==False)]
    df_ = df_[(np.isnan(df_[y])==False)]
    n   = len(df_)
    X = np.ones((n,1))
    for i in x:
        x_ = np.matrix((df_[i])).reshape(-1,1)
        X  = np.concatenate((X,x_),axis=1)
    Y = matrix((df_[y])).T
    XX_i = (X.T*X).I
    XY   = X.T*Y
    beta = XX_i*XY
    e = Y - X*beta
    k = X.shape[1]
    
    if vcv=='robust':
        a   = n/(n-k)
        u   = np.multiply(X,np.tile(e,(1,k)))
        vcv = a*XX_i*(u.T*u)*XX_i
    
    if vcv=='cluster':
        m     = df_[group].nunique() 
        a     = (m/(m-1))*((n-1)/(n-k))
        mlist = list(set(df_[group]))
        psi   = np.zeros((k,k))
        df_['e'] = e
        for g in mlist:
            x_g  = np.ones((len(df_[(df_[group]==g)]),1))
            x_g1 = matrix(df_[(df_[group]==g)][x])
            x_g  = np.concatenate((x_g,x_g1),axis=1)
            e_g  = matrix(df_[(df_[group]==g)]['e']).T
            p_   = dot(dot(x_g.T,e_g),dot(e_g.T,x_g))
            psi  = p_ +psi
        vcv = XX_i * psi * XX_i

    else:
        vcv = (1/(n-k))*(e.T*e).item()*((X.T*X).I)
    
    stderr = np.sqrt(np.diagonal(vcv))
    
    #print("Betas:",list(beta.A[:,0]))
    #print("Standard Errors:", list(np.array(stderr.T)))
    #print("Obs:",n)
    print(tabulate({"Var Name":['constant']+ x,
                    "Betas": beta.A[:,0],
                    "Std Errors": stderr,
                    "T-stat":np.absolute((np.divide(beta.A[:,0],stderr)))} , headers="keys"))
    print("Obs:",n)

# This should really be a class with the methods for estimating betas and the standard errors.  
# Future form of the model should dictate how the class should work.
# Should include: estimate betas and standard errors, table summary, post-estimation tests

# Here is an attempt at a "within" fixed effects model using the demeaning procedure explained in Hayashi (2000) pg. 324-331


def fe_lm(y=None,x=None,df=None,t_v=None,i_v=None,vcv='robust',group=None):
    # get dummy variables
    #df_c  = (pd.concat([df,pd.get_dummies(df[t_v])], axis=1)).sort([t_v,i_v])
    #df_c = df_c[(np.isnan(df_c[y])==False)]
    # Balance panel
    non_index = list()
    for i in [y] + x:
        g = df.groupby([i_v,t_v]).count()[i].sum(level=0)
        non_index = non_index+list(g[g==0].index)
    df_c = df[-df[i_v].isin(set(non_index))]

    T  = df_c[t_v].nunique()
    n  = len(df_c)
    N  = df_c[i_v].nunique()
    tlist  = sorted(list(set(df_c[t_v])))
    ilist  = sorted(list(set(df_c[i_v])))

    F      = np.ones((n,1))
    for v in [y]+ x:
        V_ = np.zeros((T,1))
        for i in ilist:
            d1 = df_c[v][(df_c[i_v]==i)]
            d  = np.matrix(d1.map(lambda x: int(~np.isnan(x)))).T
            M  = d.sum()
            Q  = np.identity(T)- (d*d.T)/M
            v_ = Q*np.matrix(d1.fillna(0)).T
            V_ = np.column_stack((V_,v_))
        F  = concatenate((F,(V_[:,1:].flatten()).T),axis=1)   
    Y = F[:,1]
    X = F[:,2:2+len(x)]
    
    XX_i  = (X.T*X).I
    XY    = X.T*Y
    beta  = XX_i*XY
    e = Y - X*beta

    k = X.shape[1]
    
    if vcv=='robust':
        a   = n/(n-N-k)
        u   = np.multiply(X,np.tile(e,(1,k)))
        vcv = a*XX_i*(u.T*u)*XX_i
    
    if vcv=='cluster':
        #g_i = list(df_[group].value_counts()[(df[group].value_counts()<5)].index)
        #df_ = df_[-df_[group].isin(g_i)]
        m     = df_c[group].nunique() 
        a     = (m/(m-1))*((n-1)/(n-N-k))
        mlist = list(set(df_c[group]))
        psi   = np.zeros((k,k))
        df_c['e'] = e
        for g in mlist:
            x_g  = np.ones((len(df_c[(df_c[group]==g)]),1))
            x_g1 = matrix(df_c[(df_c[group]==g)][x])
            x_g  = np.concatenate((x_g,x_g1),axis=1)
            e_g  = matrix(df_c[(df_c[group]==g)]['e']).T
            p_   = dot(dot(x_g.T,e_g),dot(e_g.T,x_g))
            psi  = p_ +psi
        vcv = XX_i * psi * XX_i

    else:
        vcv = (1/(n-N-k))*(e.T*e).item()*XX_i
    
    stderr = np.sqrt(np.diagonal(vcv))
    print(tabulate({"Var Name":x,
                    "Betas": beta.A[:,0],
                    "Std Errors": stderr,
                    "T-stat":np.absolute((np.divide(beta.A[:,0],stderr)))} , headers="keys"))
    print("Obs:",n,"N:",N)

# Is close to R estimates though not the same.
# This may not work for unbalanced panels.  Needs to be robustly tested.
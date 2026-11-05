import numpy as np

def conformal_quantiles(y_cal, pred_cal, alpha=0.10):
    """Split-conformal absolute residual quantiles, separately per target."""
    residuals=np.abs(np.asarray(y_cal)-np.asarray(pred_cal))
    n=len(residuals)
    q=[]
    for j in range(residuals.shape[1]):
        k=min(n, int(np.ceil((n+1)*(1-alpha))))
        q.append(float(np.sort(residuals[:,j])[k-1]))
    return q

def interval_metrics(y_true, pred, quantiles):
    y=np.asarray(y_true); p=np.asarray(pred); q=np.asarray(quantiles)
    lo=p-q; hi=p+q
    return {"marginal_coverage":np.mean((y>=lo)&(y<=hi),axis=0).tolist(),
            "mean_interval_width":(2*q).tolist(),
            "simultaneous_coverage":float(np.mean(np.all((y>=lo)&(y<=hi),axis=1)))}

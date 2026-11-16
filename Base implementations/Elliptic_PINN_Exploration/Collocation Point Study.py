"""Controlled collocation-count study using Elliptic_PINN.PINN and compute_loss.

Exactly N/3 uniformly spaced midpoints in each requested interval. Equal
weights 1/N reproduce the source's mean residual objective: this is a
boundary-emphasised sampling measure, not uniform-domain quadrature.
Every count starts from identical parameters for a given seed. All points
are used (the source training routine drops its first/last points).
"""
from pathlib import Path
import argparse
import importlib.util
import time
import csv
import json
import numpy as np
import torch


def load_source(path):
    spec = importlib.util.spec_from_file_location('elliptic_pinn_source', path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def thirds_points(n, epsilon):
    if n < 3 or n % 3:
        raise ValueError('Point counts must be positive multiples of three.')
    if not 0 < epsilon < .5:
        raise ValueError('Require 0 < epsilon < 0.5.')
    q = (np.arange(n//3)+.5)/(n//3)
    x = np.concatenate([a+(b-a)*q for a,b in [(0,epsilon),(epsilon,1-epsilon),(1-epsilon,1)]])
    return torch.tensor(x[:,None],dtype=torch.float32), torch.full((n,1),1/n,dtype=torch.float32)


def reference(x, e):
    left,right = np.exp(-x/e),np.exp((x-1)/e)
    return 1-(left+right)/(1+np.exp(-1/e)), (left-right)/(e*(1+np.exp(-1/e)))


def evaluate(model, x, derivative=False):
    values, slopes = [],[]
    for part in np.array_split(x,max(1,int(np.ceil(len(x)/2048)))):
        t=torch.tensor(part[:,None],dtype=next(model.parameters()).dtype,requires_grad=derivative)
        with torch.set_grad_enabled(derivative):
            u=model(t)
            if derivative:
                slopes.append(torch.autograd.grad(u.sum(),t)[0].detach().numpy().ravel())
        values.append(u.detach().numpy().ravel())
    return np.concatenate(values), np.concatenate(slopes) if derivative else None


def errors(model, epsilon, order):
    # Evaluate the saved float32 parameter values in float64 for integration.
    model.double()
    breaks=np.linspace(0,1,max(1001,int(np.ceil(10/epsilon))+1))
    z,w=np.polynomial.legendre.leggauss(order)
    half=np.diff(breaks)/2; mid=(breaks[:-1]+breaks[1:])/2
    x=(mid[:,None]+half[:,None]*z).ravel(); weights=(half[:,None]*w).ravel()
    u,du=evaluate(model,x,True); target,dtarget=reference(x,epsilon)
    bc,_=evaluate(model,np.array([0.,1.]))
    return {'L2_error':float(np.sqrt(np.sum(weights*(u-target)**2))),
            'H1_seminorm_error':float(np.sqrt(np.sum(weights*(du-dtarget)**2))),
            'boundary_residual_norm':float(np.linalg.norm(bc))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',nargs='+',type=int,default=[30,90,300,900])
    p.add_argument('--epochs',type=int,default=20000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epsilon',type=float,default=.01)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parent.parent/'Elliptic_PINN.py')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'collocation_point_results')
    p.add_argument('--no-show',action='store_true')
    args=p.parse_args()
    if args.epochs<1 or args.counts!=sorted(set(args.counts)):
        p.error('Use positive epochs and increasing, distinct counts.')
    for n in args.counts:
        thirds_points(n,args.epsilon)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    source=load_source(args.source)
    args.output.mkdir(parents=True,exist_ok=True)
    x_plot=np.linspace(0,1,10001); curves={}; histories={}; rows=[]
    for n in args.counts:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model=source.PINN().float()
        x,w=thirds_points(n,args.epsilon)
        optimiser=torch.optim.Adam(model.parameters(),lr=.01)
        history=[]; start=time.perf_counter()
        for epoch in range(args.epochs):
            optimiser.zero_grad(set_to_none=True)
            x.grad=None
            loss=source.compute_loss(model,x,w,args.epsilon)
            if not torch.isfinite(loss).all():
                raise RuntimeError(f'Nonfinite loss: N={n}, epoch={epoch}')
            loss.backward(); optimiser.step()
            history.append(loss.item())
            if (epoch+1)%5000==0:
                print(f'N={n}, step={epoch+1}, loss={loss.item():.6e}',flush=True)
        seconds=time.perf_counter()-start
        final_loss=source.compute_loss(model,x,w,args.epsilon).item()
        np.savez(args.output/f'params_N{n}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        np.savez(args.output/f'training_N{n}.npz',x=x.detach().numpy(),weights=w.numpy(),loss=np.array(history))
        low=errors(model,args.epsilon,8); high=errors(model,args.epsilon,16)
        if not all(np.isclose(low[k],high[k],rtol=1e-6,atol=1e-10) for k in low):
            raise RuntimeError(f'Quadrature refinement failed for N={n}')
        row={'points':n,'seed':args.seed,'epochs':args.epochs,'training_loss':final_loss,'wall_seconds':seconds,**high}
        rows.append(row); curves[n]=evaluate(model,x_plot)[0]; histories[n]=np.array(history)
        with (args.output/'errors.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
        print(row,flush=True)
    (args.output/'settings.json').write_text(json.dumps({'epsilon':args.epsilon,'counts':args.counts,
        'epochs':args.epochs,'seed':args.seed,'optimiser':'Adam','lr':.01,'architecture':'1-100-1 tanh',
        'training_dtype':'float32','sampling':'equal thirds; uniform midpoints in each interval',
        'weights':'1/N; boundary-emphasised measure','source':str(args.source.resolve()),
        'budget':'same steps; cost grows with N; one paired initialisation'},indent=2))
    import matplotlib
    if args.no_show: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,layout='constrained')
    target,_=reference(x_plot,args.epsilon)
    axes[0].plot(x_plot,target,'k-',lw=2,label='Analytical solution')
    for n,u in curves.items():
        line,=axes[0].plot(x_plot,u,'--',label=f'N = {n}')
        axes[1].semilogy(x_plot,np.maximum(abs(u-target),1e-16),color=line.get_color())
    axes[0].set(ylabel='u(x)',title=rf'Collocation-count study: $\varepsilon={args.epsilon:g}$')
    axes[0].legend(); axes[1].set(xlabel='x',ylabel='Absolute error')
    for ax in axes: ax.grid(alpha=.25); ax.set_xlim(0,1)
    fig.savefig(args.output/'collocation_count_comparison.pdf'); fig.savefig(args.output/'collocation_count_comparison.png',dpi=180)
    fig2,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for n,h in histories.items(): ax.loglog(np.arange(1,len(h)+1),h,label=f'N = {n}')
    ax.set(xlabel='Adam step',ylabel='Training loss'); ax.legend(); ax.grid(alpha=.25)
    fig2.savefig(args.output/'training_losses.pdf')
    if not args.no_show: plt.show()
    plt.close('all')
    print(f'Outputs: {args.output.resolve()}')


if __name__=='__main__': main()

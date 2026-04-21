from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

INSTRUMENTS = {
    'tbill_91':  {'name': 'أذون خزانة 91 يوم',    'base_yield': 0.278, 'std': 0.008, 'liquidity': 'weekly'},
    'tbill_182': {'name': 'أذون خزانة 182 يوم',   'base_yield': 0.282, 'std': 0.007, 'liquidity': 'bi-weekly'},
    'mmf_ahly1': {'name': 'صندوق الأهلي الأول',    'base_yield': 0.245, 'std': 0.005, 'liquidity': 'daily'},
    'mmf_ahly3': {'name': 'صندوق الأهلي الثالث',  'base_yield': 0.192, 'std': 0.018, 'liquidity': 'weekly'},
    'cd_3m':     {'name': 'شهادة ادخار 3 أشهر',    'base_yield': 0.235, 'std': 0.003, 'liquidity': 'monthly'},
    'cd_1y':     {'name': 'شهادة ادخار سنة',       'base_yield': 0.250, 'std': 0.002, 'liquidity': 'annual'},
}
RISK = {'low': 0.20, 'mid': 0.40, 'high': 0.65}

def get_yields(rate):
    base = np.array([v['base_yield'] for v in INSTRUMENTS.values()])
    return np.clip(base + (rate-27)/100 * np.array([0.9,0.9,0.7,0.5,0.8,0.85]), 0.01, 0.50)

def build_cov():
    stds = np.array([v['std'] for v in INSTRUMENTS.values()])
    corr = np.eye(len(stds))*0.85 + np.ones((len(stds),len(stds)))*0.15
    return np.outer(stds,stds)*corr

def optimise(surplus, horizon, emerg, risk, rate):
    keys = list(INSTRUMENTS.keys()); n = len(keys)
    yields = get_yields(rate); cov = build_cov()
    investable = 1 - emerg; rf = 0.20
    def neg_sharpe(w):
        r = np.dot(w,yields)*investable; s = np.sqrt(w@cov@w)
        return -(r-rf)/s if s>1e-8 else 0
    constraints = [{'type':'eq','fun':lambda w:np.sum(w)-1.0}]
    constraints.append({'type':'ineq','fun':lambda w: RISK[risk]-w[keys.index('cd_1y')]})
    res = minimize(neg_sharpe, np.ones(n)/n, method='SLSQP',
                   bounds=[(0,1)]*n, constraints=constraints,
                   options={'ftol':1e-9,'maxiter':1000})
    w = np.clip(res.x, 0, 1); w /= w.sum()
    ret = float(np.dot(w,yields)); std = float(np.sqrt(w@cov@w))
    sharpe = round((ret*investable-rf)/(std+1e-8), 2)
    abs_ret = round(surplus*ret*investable*(min(horizon,12)/12), 3)
    def sc(shock): return round(float(np.dot(w, get_yields(rate+shock)))*investable*100, 2)
    allocs = [{'key':keys[i],'name':INSTRUMENTS[keys[i]]['name'],
               'weight_pct':round(float(w[i])*investable*100,1),
               'amount_m':round(float(w[i])*investable*surplus,3),
               'yield':round(float(yields[i])*100,2)}
              for i in range(n) if w[i]>0.01]
    # efficient frontier
    ef = []
    for t in np.linspace(yields.min(), yields.max(), 12):
        r2 = minimize(lambda w:float(w@cov@w), np.ones(n)/n, method='SLSQP',
                      bounds=[(0,1)]*n,
                      constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1},
                                   {'type':'eq','fun':lambda w:np.dot(w,yields)-t}])
        if r2.success:
            ef.append({'risk':round(float(np.sqrt(r2.fun))*100,3),'return':round(t*investable*100,2)})
    return {'allocations':allocs,'portfolio_return':round(ret*investable*100,2),
            'abs_return_m':abs_ret,'sharpe_ratio':sharpe,'portfolio_std':round(std*100,2),
            'emergency_amount_m':round(surplus*emerg,3),
            'scenarios':{'shock_minus_2':sc(-2),'base':sc(0),'shock_plus_2':sc(+2)},
            'efficient_frontier':sorted(ef,key=lambda p:p['risk'])}

@app.route('/health')
def health(): return jsonify({'status':'running'})

@app.route('/instruments')
def api_instruments():
    rate = float(request.args.get('rate',27)); yields = get_yields(rate)
    keys = list(INSTRUMENTS.keys())
    return jsonify([{**INSTRUMENTS[keys[i]],'key':keys[i],'yield':round(float(yields[i])*100,2)} for i in range(len(keys))])

@app.route('/optimise')
def api_optimise():
    return jsonify(optimise(
        float(request.args.get('surplus',50)), int(request.args.get('horizon',3)),
        float(request.args.get('emergency',0.15)), request.args.get('risk','mid'),
        float(request.args.get('rate',27.0))
    ))

if __name__ == '__main__':
    print("="*50)
    print("  Robo-Advisor API — شغّال!")
    print("  http://localhost:5001")
    print("  افتح RoboAdvisor.html في المتصفح")
    print("="*50)
    app.run(port=5001, debug=False)

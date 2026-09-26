"""Reusable channel moments and disk-backed polynomial banks for budget ladders.

Only small moment vectors are retained in RAM. Tuple rows are regenerated in
bounded chunks when needed: retaining all triple rows would require excessive
storage. Previously accumulated moments are never accumulated a second time.
"""
from collections import Counter
import hashlib
import json
import os
import pickle
from pathlib import Path
import numpy as np
try:
    from .budget_sweep import polynomial_bank
    from .encodings import feature_chunks, channel_count
except ImportError:  # standalone modules inside a frozen BREC snapshot
    from budget_sweep import polynomial_bank
    from brec_encodings import feature_chunks, channel_count


def channel_keys(q, order, variant):
    positions=[(0,1)] if order==2 else [(i,j) for i in range(3) for j in range(3) if i!=j]
    keys=[('one',)]
    keys += [('pair',s,i,j) for s in ('I','A',*range(1,q+1)) for i,j in positions]
    if variant!='polynomial':
        keys += [('endpoint',a,b,i,j) for i,j in positions
                 for a in range(q) for b in range(q-a) if (a,b)!=(0,0)]
    if variant=='joint':
        keys += [('joint',a,b,c) for a in range(q+1) for b in range(q+1) for c in range(q+1)]
    assert len(keys)==channel_count(order,q,variant)
    return keys


class BudgetCache:
    def __init__(self, directory):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.moments={}

    def dump(self, path):
        """Persist only sufficient statistics, never full tuple arrays."""
        payload=[(list(identity),[(list(key),list(map(float,value))) for key,value in state.items()])
                 for identity,state in self.moments.items()]
        path=Path(path)
        temp=path.with_suffix('.tmp')
        with temp.open('w') as output:
            json.dump(payload,output,allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        temp.replace(path)

    def restore(self, path):
        self.moments={tuple(identity):{tuple(key):value for key,value in state}
                      for identity,state in json.loads(Path(path).read_text())}

    def bank(self, adj, q, basis):
        identity=hashlib.sha256(adj.tobytes()).hexdigest()+'_'+basis+'_float32'
        path=self.directory/(identity+'.npz')
        changed=not path.exists()
        if path.exists():
            with np.load(path,allow_pickle=False) as stored:
                previous=stored['bank']
                if str(stored['sha256'])!=hashlib.sha256(previous.tobytes()).hexdigest():
                    raise ValueError('Polynomial cache checksum mismatch')
        else:
            previous=polynomial_bank(adj,1,basis)
        if len(previous)<=q:
            changed=True
            values=list(previous)
            matrix=values[1]
            for degree in range(len(values),q+1):
                values.append(matrix@values[-1] if basis=='monomial'
                              else 2*matrix@values[-1]-values[-2])
            previous=np.stack(values)
        if changed:
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as output:
                np.savez(output,bank=previous,sha256=hashlib.sha256(previous.tobytes()).hexdigest())
                output.flush()
                os.fsync(output.fileno())
            temp.replace(path)
        return identity,previous[:q+1]

    def signatures(self, adj, q, basis, order, decimals, chunk_size,
                   readouts, p_values, variants, joint_normalise, requested=None):
        identity,bank=self.bank(adj,q,basis)
        state=self.moments.setdefault((identity,order,joint_normalise),{})
        widest=max(variants,key=lambda v:channel_count(order,q,v))
        keys=channel_keys(q,order,widest)
        width=len(keys)
        power=max(p_values)
        requested=requested or {v:set(readouts) for v in variants}
        mean=any('mean' in requested[v] for v in variants)
        multi=any('multiset' in requested[v] for v in variants)
        mean_keys={key for v in variants if 'mean' in requested[v]
                   for key in channel_keys(q,order,v)}
        # Exact identities in real arithmetic. Numeric comparisons retain the
        # same rounding/tolerance; no rounding of intermediate quantities.
        n=len(adj)
        count=float(n**order)
        h=bank.sum(2)
        column_sums=bank.sum(1)
        endpoint_moments={}
        grams={}
        direct={}
        for key in mean_keys:
            old=state.get(key,[])
            target=min(power,2) if key[0]=='joint' else power
            if len(old)>=target:
                continue
            values=list(old)
            for exponent in range(len(old)+1,target+1):
                canonical=key[:2] if key[0]=='pair' else key[:3] if key[0]=='endpoint' else key
                token=(canonical,exponent)
                if token not in direct:
                    with np.errstate(over='raise',invalid='raise'):
                        if key[0]=='one':
                            value=1.
                        elif key[0]=='pair':
                            seed=key[1]
                            matrix=np.eye(n, dtype=np.float32) if seed=='I' else adj if seed=='A' else bank[seed]
                            value=np.mean(matrix**exponent)
                        elif key[0]=='endpoint':
                            a,b=key[1:3]
                            if exponent not in endpoint_moments:
                                endpoint_moments[exponent]=np.mean(h**exponent,axis=1)
                            value=endpoint_moments[exponent][a]*endpoint_moments[exponent][b]
                        else:
                            a,b,c=key[1:]
                            if exponent==1:
                                value=np.sum(column_sums[a]*column_sums[b]*column_sums[c])/n**3
                            else:
                                for index in (a,b,c):
                                    if index not in grams:
                                        grams[index]=bank[index].T@bank[index]
                                value=np.sum(grams[a]*grams[b]*grams[c])/n**3
                            if joint_normalise=='n':
                                value/=n**exponent
                    if not np.isfinite(value):
                        raise FloatingPointError('Nonfinite direct moment')
                    direct[token]=float(value*count)
                values.append(direct[token])
            state[key]=values
        # Stable channel identities, not a Q-dependent prefix assumption.
        missing=[i for i,key in enumerate(keys) if key in mean_keys and len(state.get(key,()))<power]
        multiset_width=max((channel_count(order,q,v) for v in variants
                           if 'multiset' in requested[v]),default=0)
        columns=sorted(set(range(multiset_width)) | set(missing))
        totals=np.zeros((power,len(columns)), dtype=np.float32)
        correction=np.zeros_like(totals)
        counters={v:{d:Counter() for d in decimals} for v in variants
                  if 'multiset' in requested[v]}
        starts=np.array([len(state.get(keys[i],())) if keys[i] in mean_keys else power
                         for i in columns])
        if columns:
            for rows in feature_chunks(adj,bank,q,order,widest,chunk_size,
                                       joint_normalise,columns=columns):
                if not np.isfinite(rows).all():
                    raise FloatingPointError('Nonfinite encoding')
                if missing:
                    powered=rows.copy()
                    with np.errstate(over='raise',invalid='raise'):
                        for exponent in range(power):
                            if exponent:
                                powered*=rows
                            needed=starts<=exponent
                            if not needed.any():
                                continue
                            increment=powered[:,needed].sum(0)-correction[exponent,needed]
                            updated=totals[exponent,needed]+increment
                            correction[exponent,needed]=(updated-totals[exponent,needed])-increment
                            totals[exponent,needed]=updated
                if multi:
                    for variant in counters:
                        sub=rows[:,:channel_count(order,q,variant)]
                        for d in decimals:
                            rounded=np.round(sub,d)
                            rounded[rounded==0]=0.
                            for row in rounded:
                                counters[variant][d][hashlib.sha256(row.astype('<f4',copy=False).tobytes()).digest()]+=1
            if mean:
                if not np.isfinite(totals).all():
                    raise FloatingPointError('Nonfinite aggregated readout')
                for col,index in enumerate(columns):
                    key=keys[index]
                    if key not in mean_keys:
                        continue
                    old=state.get(key,[])
                    state[key]=list(old)+list(totals[len(old):,col])
        results={}
        for variant in variants:
            result=({('multiset',d,1):counters[variant][d] for d in decimals}
                    if variant in counters else {})
            if 'mean' in requested[variant]:
                selected=channel_keys(q,order,variant)
                values=np.array([state[key][:power] for key in selected], dtype=np.float32).T / float(len(adj)**order)
                for p in p_values:
                    for d in decimals:
                        rounded=np.round(values[:p],d)
                        rounded[rounded==0]=0.
                        result[('mean',d,p)]=rounded.astype('<f4',copy=False).tobytes()
            results[variant]=result
        return results

    def sliced_signatures(self, adj, q, q_values, basis, order, decimals,
                          chunk_size, readouts, p_values, variants, joint_normalise,
                          scale_audit=False):
        """One Qmax feature pass per graph; persist only aggregate signatures.

        Low-budget columns are selected by channel identity, not by prefix.
        Pickles are private, locally generated temporary cache files only.
        Timing belongs to the shared pass, not an individual Q condition.
        """
        graph_hash = hashlib.sha256(adj.tobytes()).hexdigest()
        configuration = ("float32", graph_hash, adj.shape, sorted(q_values), basis, order,
                         decimals, chunk_size, readouts, p_values, variants, joint_normalise, scale_audit)
        token = hashlib.sha256(repr(configuration).encode()).hexdigest()
        paths = {budget: self.directory / f'slices-{token}-{budget}.pkl'
                 for budget in q_values}
        if not all(path.exists() for path in paths.values()):
            qmax = max(q_values)
            _, bank = self.bank(adj, qmax, basis)
            widest = max(variants, key=lambda v: channel_count(order, qmax, v))
            keys = channel_keys(qmax, order, widest)
            lookup = {key: i for i, key in enumerate(keys)}
            selections = {(budget, variant): np.array([
                lookup[key] for key in channel_keys(budget, order, variant)], dtype=int)
                for budget in q_values for variant in variants}
            counters = {condition: {d: Counter() for d in decimals}
                        for condition in selections} if 'multiset' in readouts else {}
            power = max(p_values)
            totals = np.zeros((power, len(keys)), dtype=np.float32)
            correction = np.zeros_like(totals)
            audit = scale_audit and order == 3 and 'joint' in variants and 4 in q_values
            joint_start = channel_count(order,qmax,'endpoint') if audit else 0
            audit_columns = [lookup[key] for key in channel_keys(4,3,'joint') if key[0]=='joint'] if audit else []
            audit_totals = np.zeros((2,power,len(audit_columns)),dtype=np.float32)
            audit_correction = np.zeros_like(audit_totals)
            for rows in feature_chunks(adj, bank, qmax, order, widest,
                                       chunk_size, None if audit else joint_normalise):
                if audit:
                    raw_joint = rows[:,audit_columns]
                    for mode, values in enumerate((raw_joint, raw_joint / np.float32(len(adj)))):
                        powered = values.copy()
                        with np.errstate(over='raise',invalid='raise'):
                            for exponent in range(power):
                                if exponent:
                                    powered *= values
                                increment = powered.sum(0) - audit_correction[mode,exponent]
                                updated = audit_totals[mode,exponent] + increment
                                audit_correction[mode,exponent] = (updated-audit_totals[mode,exponent])-increment
                                audit_totals[mode,exponent] = updated
                    if joint_normalise == 'n':
                        rows[:,joint_start:] /= np.float32(len(adj))
                if not np.isfinite(rows).all():
                    raise FloatingPointError('Nonfinite maximum-budget encoding')
                if 'mean' in readouts:
                    powered = rows.copy()
                    with np.errstate(over='raise', invalid='raise'):
                        for exponent in range(power):
                            if exponent:
                                powered *= rows
                            increment = powered.sum(axis=0) - correction[exponent]
                            updated = totals[exponent] + increment
                            correction[exponent] = (updated - totals[exponent]) - increment
                            totals[exponent] = updated
                # Round the maximum dictionary once per precision, then project.
                for d in decimals if counters else ():
                    rounded = np.round(rows, d)
                    rounded[rounded == 0] = 0.
                    for condition, columns in selections.items():
                        for row in rounded[:, columns]:
                            counters[condition][d][hashlib.sha256(
                                row.astype('<f4', copy=False).tobytes()).digest()] += 1
            if not np.isfinite(totals).all():
                raise FloatingPointError('Nonfinite maximum-budget moments')
            for budget, path in paths.items():
                result = {}
                for variant in variants:
                    condition = (budget, variant)
                    signatures = {('multiset', d, 1): value
                                  for d, value in counters.get(condition, {}).items()}
                    if 'mean' in readouts:
                        values = totals[:, selections[condition]] / float(len(adj)**order)
                        for power_limit in p_values:
                            for d in decimals:
                                rounded = np.round(values[:power_limit], d)
                                rounded[rounded == 0] = 0.
                                signatures[('mean', d, power_limit)] = rounded.astype(
                                    '<f4', copy=False).tobytes()
                    result[variant] = signatures
                if audit and budget == 4:
                    baseline = totals[:,selections[(4,'joint')]] / np.float32(len(adj)**3)
                    offset = channel_count(3,4,'endpoint')
                    audit_values = np.stack([baseline.copy() for _ in range(3)])
                    audit_values[:2,:,offset:] = audit_totals / np.float32(len(adj)**3)
                    with np.errstate(over='raise',invalid='raise'):
                        factors = np.power(np.float32(len(adj)),np.arange(1,power+1,dtype=np.float32))
                        audit_values[2,:,offset:] = audit_values[1,:,offset:] * factors[:,None]
                    if not np.isfinite(audit_values).all():
                        raise FloatingPointError('Nonfinite scale audit moments')
                    result['_scale_moments'] = audit_values
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as output:
                    pickle.dump(result, output, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(path)
        with paths[q].open('rb') as source:
            return pickle.load(source)


def scale_audit_decisions(samples, labels, p, decimals, atol, rtol=1e-10):
    """All sampled cross pairs and same-graph pairs; all vectors remain means.

    Samples: graph x [K, K/n, restored K/n] x power x full dictionary.
    Non-joint channels retain their primary scale in every mode.
    """
    results={}
    labels=np.asarray(labels)
    for mode, name, rounded in ((0,'K_rounded',True),(1,'K_over_n_rounded',True),
                                (0,'K_unrounded',False),(1,'K_over_n_unrounded',False),
                                (2,'restored_unrounded',False)):
        values=samples[:,mode,:p].reshape(len(samples),-1)
        if rounded:
            values=np.round(values,decimals)
        tolerance_r=0. if rounded else rtol
        def close(a,b):
            return bool(np.all(np.abs(a-b)<=atol+tolerance_r*np.maximum(np.abs(a),np.abs(b))))
        groups=[values[labels==label] for label in ('left','right','reliability')]
        same=True; max_same=0.; min_cross=float('inf'); cross=True
        for group in groups:
            for i in range(len(group)):
                for j in range(i):
                    same &= close(group[i],group[j])
                    max_same=max(max_same,float(np.max(np.abs(group[i]-group[j]))))
        for left in groups[0]:
            for right in groups[1]:
                cross &= not close(left,right)
                min_cross=min(min_cross,float(np.max(np.abs(left-right))))
        checked=any(len(group)>1 for group in groups)
        results[name]=dict(all_cross_separated=cross,
                           same_graph_checks_performed=checked,
                           same_graph_consistent=same if checked else None,
                           confirmed_separated=bool(cross and same),
                           numerical_outcome=('unreliable' if not same else
                                              'separated' if cross else 'not_separated'),
                           min_cross_linf=min_cross,max_same_linf=max_same if checked else None,
                           atol=atol,rtol=tolerance_r,
                           decimals=decimals if rounded else None)
    return results

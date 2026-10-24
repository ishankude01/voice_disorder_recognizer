"""Two-source binary experiment. All SVD and VOICED examples are development.

Stages: prepare, features, train, report. Historical v2 is never overwritten.
Training is nested, participant-disjoint, resumable and bounded to ten hours.
"""
from pathlib import Path
import argparse, hashlib, io, json, platform, random, shutil, time, warnings
import joblib
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from svd_training import SPEC, extract_features, load_audio
from svd_advanced import transform_fit, transform, raw_scores, cnn_network

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/v3'
OUT=ROOT/'artifacts_v3'
SEED=20261023
VERSION='two-source-v3-97features'
SOURCES=['SVD','VOICED']
LABELS=['Healthy','Pathological']
DEADLINE=float('inf')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str),encoding='utf-8')
def check_time():
    if time.time()>DEADLINE:raise TimeoutError('Ten-hour experiment budget reached; retain historical fallback.')
def source_hashes(): return {name:sha(ROOT/name) for name in ('svd_v3.py','svd_training.py','svd_advanced.py')}
def spec_for(sr):
    if sr not in (8000,16000):raise ValueError('Unsupported model sampling rate')
    return {**SPEC,'sample_rate':sr,'n_fft':256 if sr==8000 else 512,
        'win_length':int(.025*sr),'hop_length':int(.01*sr),'version':VERSION,
        'cpps_max_frequency':3800 if sr==8000 else 4000,'mel_shape':[40,128]}

def audio_bytes(path):
    path=Path(path)
    if path.suffix.lower()!='.hea':return path.read_bytes()
    import wfdb
    r=wfdb.rdrecord(str(path.with_suffix('')))
    b=io.BytesIO();sf.write(b,r.p_signal,int(r.fs),format='WAV',subtype='DOUBLE')
    return b.getvalue()

def extract(content,sr):
    """Exactly the same deterministic extraction in fitting and the website."""
    spec=spec_for(sr)
    base,meta=extract_features(content,spec)
    x,_=load_audio(content,spec)
    power=np.abs(librosa.stft(x,n_fft=spec['n_fft'],win_length=spec['win_length'],
        hop_length=spec['hop_length'],center=False))**2
    mel=librosa.feature.melspectrogram(S=power,sr=sr,n_fft=spec['n_fft'],n_mels=40,fmin=50,fmax=sr/2)
    logmel=librosa.power_to_db(mel,ref=1.)
    mfcc=librosa.feature.mfcc(S=logmel,n_mfcc=13)
    extra=[]
    for order in (1,2):
        v=librosa.feature.delta(mfcc,order=order,width=9)
        extra.extend(v.mean(1));extra.extend(v.std(1))
    sound=parselmouth.Sound(x,sampling_frequency=sr);q=np.full(3,np.nan)
    try:
        pc=call(sound,'To PowerCepstrogram',60,.002,spec['cpps_max_frequency'],50)
        q[0]=call(pc,'Get CPPS',True,.01,.001,60,500,.05,'Parabolic',.001,.05,'Straight','Robust')
    except parselmouth.PraatError:pass
    try:
        pp=call(sound,'To PointProcess (periodic, cc)',60,500)
        q[1]=call(pp,'Get jitter (local)',0,0,1/500,1/60,1.3)
        q[2]=call([sound,pp],'Get shimmer (local)',0,0,1/500,1/60,1.3,1.6)
    except parselmouth.PraatError:pass
    rich=np.r_[base,extra,q];rich[~np.isfinite(rich)]=np.nan
    fixed=np.stack([np.interp(np.linspace(0,1,128),np.linspace(0,1,len(v)),v) for v in logmel]).astype('float32')
    fingerprint=hashlib.sha256(np.round(x*1e6).astype('<i4').tobytes()).hexdigest()
    return rich,fixed,fingerprint,meta

def prepare(metadata):
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol already frozen; use existing manifest.')
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    history={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'artifacts_svd_v2').rglob('*') if p.is_file()}
    dump(OUT/'historical_v2_hashes.json',history)
    d=pd.read_csv(ROOT/'data/svd_v2/development_features.csv').fillna('')
    s=pd.read_csv(metadata).fillna('')
    d=d.merge(s,left_on='record_id',right_on='AufnahmeID',validate='one_to_one')
    d['age']=(pd.to_datetime(d.AufnahmeDatum)-pd.to_datetime(d.Geburtsdatum)).dt.days/365.2425
    rows=[]
    for r in d.itertuples():
        rows.append(dict(record_id='SVD:'+str(r.record_id),speaker_id='SVD:'+str(r.speaker_id),
            source='SVD',label=int(r.binary),diagnosis=r.source_diagnoses,age=float(r.age),
            sex={'w':'f','m':'m'}.get(r.Geschlecht,'unknown'),path=r.path,raw_hash=sha(r.path)))
    v=pd.read_csv(ROOT/'data/manifest.csv').fillna('')
    label_audit=[]
    for r in v.itertuples():
        path=ROOT/'data/raw'/f'{r.record_id}.hea'
        header=path.read_text();info=path.with_name(r.record_id+'-info.txt').read_text()
        hd=header.split('<diagnoses>:')[1].split('<medications>:')[0].strip().lower()
        idg=next(line.split(':',1)[1].strip().lower() for line in info.splitlines() if line.startswith('Diagnosis:'))
        healthy={'healthy','normal','healthy voice','normal voice'}
        if (hd in healthy)!=(idg in healthy) or idg!=r.diagnosis.lower() or int(r.label)!=int(idg not in healthy):
            raise ValueError('VOICED label discrepancy: '+r.record_id)
        label_audit.append({'record_id':r.record_id,'header_diagnosis':hd,'detailed_diagnosis':idg,
            'binary_agrees':True,'diagnosis_text_differs':hd!=idg})
        raw_hash=key([sha(path),sha(path.with_suffix('.dat'))])
        rows.append(dict(record_id='VOICED:'+r.record_id,speaker_id='VOICED:'+str(r.speaker_id),source='VOICED',
            label=int(r.label),diagnosis=r.diagnosis,age=float(r.age),sex=str(r.sex).lower(),path=str(path.resolve()),raw_hash=raw_hash))
    frame=pd.DataFrame(rows)
    pd.DataFrame(label_audit).to_csv(OUT/'voiced_label_audit.csv',index=False)
    if frame.speaker_id.duplicated().any():raise ValueError('Expected one audited recording per participant')
    frame['age_band']=pd.cut(frame.age,[-1,29,49,69,150],labels=['under30','30-49','50-69','70plus']).astype(str)
    frame.to_csv(DATA/'manifest.csv',index=False)
    frame.drop(columns='path').to_csv(OUT/'manifest.csv',index=False)
    frame.groupby(['source','label']).agg(recordings=('record_id','size'),speakers=('speaker_id','nunique'),
        median_age=('age','median'),min_age=('age','min'),max_age=('age','max')).to_csv(OUT/'dataset_audit.csv')
    frame.groupby(['source','label','sex','age_band']).size().rename('speakers').to_csv(OUT/'demographic_audit.csv')
    dump(OUT/'protocol.json',dict(seed=SEED,version=VERSION,source_hashes=source_hashes(),manifest_hash=sha(DATA/'manifest.csv'),
        source_metadata_hash=sha(metadata),outer_folds=4,inner_folds=3,candidates=candidates(),
        evaluation='All data are development, including all 208 VOICED records. No new untouched external test.',
        selection='Mean source-specific balanced accuracy; within 0.01 prefer GMM, MLP, CNN, ensemble.',
        identity='SVD participant IDs; VOICED release record IDs are participant proxies, not independently reverified.',
        voiced_counts='57 healthy/151 pathological verified against both release labels; website summary differs.',
        thresholds=[.6,.7,.8,.9],max_training_hours=10))
    provenance=OUT/'provenance';provenance.mkdir(exist_ok=True)
    for name in source_hashes():shutil.copy2(ROOT/name,provenance/name)
    import importlib.metadata
    dump(OUT/'versions.json',dict(python=platform.python_version(),packages={n:importlib.metadata.version(n)
        for n in ['numpy','scipy','pandas','scikit-learn','librosa','torch','praat-parselmouth','soundfile','wfdb','streamlit','joblib']}))
    print(frame.groupby(['source','label']).size().to_string(),flush=True)

def features():
    protocol=verify_protocol();frame=pd.read_csv(DATA/'manifest.csv');arrays={};fingerprints={}
    for sr in (16000,8000):
        xs=[];ms=[];hs=[]
        for i,row in frame.iterrows():
            path=Path(row.path);raw_hash=key([sha(path),sha(path.with_suffix('.dat'))]) if path.suffix=='.hea' else sha(path)
            if raw_hash!=row.raw_hash:raise ValueError('Source audio changed: '+row.record_id)
            ident=key([raw_hash,spec_for(sr),protocol['source_hashes']])
            cache=DATA/'feature_cache'/f'{ident}.npz';cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                with np.load(cache) as z:x,m,h=z['x'],z['mel'],str(z['fingerprint'])
            else:
                x,m,h,_=extract(audio_bytes(path),sr)
                np.savez_compressed(cache,x=x,mel=m,fingerprint=h)
            xs.append(x);ms.append(m);hs.append(h)
            if (i+1)%100==0:print('Features',sr,i+1,'/',len(frame),flush=True)
        if len(set(hs))!=len(hs):raise ValueError('Duplicate analyzed audio found. Audit required before splitting.')
        np.savez_compressed(DATA/f'features_{sr}.npz',x=np.stack(xs),mel=np.stack(ms))
        frame[f'audio_hash_{sr}']=hs
        fingerprints[str(sr)]=dict(file_hash=sha(DATA/f'features_{sr}.npz'),missing_values=int(np.isnan(xs).sum()),
            recordings_with_missing=int(np.isnan(xs).any(axis=1).sum()))
    frame.to_csv(DATA/'featured_manifest.csv',index=False)
    frame.drop(columns='path').to_csv(OUT/'featured_manifest.csv',index=False)
    dump(OUT/'feature_manifest.json',dict(arrays=fingerprints,manifest_hash=sha(DATA/'featured_manifest.csv')))

def verify_protocol():
    protocol=json.loads((OUT/'protocol.json').read_text())
    if protocol['source_hashes']!=source_hashes():raise ValueError('Source/config changed. Use a new --run-dir; stale caches are forbidden.')
    if protocol['manifest_hash']!=sha(DATA/'manifest.csv'):raise ValueError('Manifest changed; start a new experiment.')
    return protocol

def candidates():
    base=[dict(id='GMM_1',family='GMM',features='base',components=1,covariance='diag',pca=0),
        dict(id='GMM_4',family='GMM',features='rich',components=4,covariance='diag',pca=20),
        dict(id='GMM_tied',family='GMM',features='rich',components=2,covariance='tied',pca=12),
        dict(id='MLP_small',family='MLP',features='rich',layers=[32,16],alpha=1.,select=40),
        dict(id='MLP_medium',family='MLP',features='base',layers=[64,32],alpha=.1,select=0),
        dict(id='CNN_aug',family='CNN',features='rich',width=16,dropout=.5,augment=True)]
    return [{**c,'id':c['id']+'_'+str(sr),'sr':sr} for c in base for sr in (16000,8000)]

def splits(frame,n,seed):
    strata=frame.source.astype(str)+'_'+frame.label.astype(str)
    folds=list(StratifiedGroupKFold(n,shuffle=True,random_state=seed).split(frame,strata,frame.speaker_id))
    for a,b in folds:
        if set(frame.iloc[a].speaker_id)&set(frame.iloc[b].speaker_id):raise RuntimeError('Participant overlap')
        if set(strata.iloc[a])!=set(strata) or set(strata.iloc[b])!=set(strata):raise ValueError('Missing source/class stratum')
    return folds

def weights(y,src):
    strata=np.array([str(s)+'_'+str(c) for s,c in zip(src,y)])
    names,counts=np.unique(strata,return_counts=True)
    lookup=dict(zip(names,len(y)/(len(names)*counts)))
    return np.array([lookup[s] for s in strata])

def fit_raw(x,mel,y,src,config,seed,stop=None,epochs=60):
    check_time();z,pre=transform_fit(x,y,config)
    bundle={'config':config,**pre,'classes':2,'temperature':1.}
    w=weights(y,src)
    if config['family']=='GMM':
        rng=np.random.default_rng(seed);models=[]
        for c in (0,1):
            groups=[np.flatnonzero((y==c)&(src==s)) for s in SOURCES];target=max(map(len,groups))
            ids=np.concatenate([np.r_[g,rng.choice(g,target-len(g),replace=True)] for g in groups])
            model=GaussianMixture(n_components=config['components'],covariance_type=config['covariance'],
                reg_covar=.1,n_init=3,max_iter=400,random_state=seed).fit(z[ids])
            if not model.converged_:raise RuntimeError('GMM failed convergence')
            models.append(model)
        bundle['model']=models
    elif config['family']=='MLP':
        model=MLPClassifier(hidden_layer_sizes=tuple(config['layers']),alpha=config['alpha'],max_iter=700,
            batch_size=32,learning_rate_init=.001,early_stopping=False,n_iter_no_change=25,random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(z,y,sample_weight=w)
        bundle['model']=model;bundle['warnings']=[str(v.message) for v in caught]
    else:
        import torch
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        bundle['mel_mean']=mel.mean(axis=(0,2),keepdims=True)
        bundle['mel_std']=np.maximum(mel.std(axis=(0,2),keepdims=True),1.)
        mm=torch.tensor(((mel-bundle['mel_mean'])/bundle['mel_std'])[:,None],dtype=torch.float32)
        qq=torch.tensor(z,dtype=torch.float32);yy=torch.tensor(y,dtype=torch.long);ww=torch.tensor(w,dtype=torch.float32)
        net=cnn_network(config['width'],config['dropout'],z.shape[1],2)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        ce=torch.nn.CrossEntropyLoss(reduction='none');best=float('inf');best_epoch=1;stale=0
        if stop is not None:
            sm=torch.tensor(((stop[1]-bundle['mel_mean'])/bundle['mel_std'])[:,None],dtype=torch.float32)
            sq=torch.tensor(transform(bundle,stop[0]),dtype=torch.float32);sy=torch.tensor(stop[2],dtype=torch.long)
            # Balanced validation objective; stop labels never fit preprocessing/model weights.
            sw=torch.tensor(weights(stop[2],stop[3]),dtype=torch.float32)
        for epoch in range(1,epochs+1):
            check_time();net.train()
            for ids in torch.randperm(len(y)).split(32):
                m=mm[ids].clone();m+=torch.randn_like(m)*.03
                t=int(torch.randint(0,120,(1,)));f=int(torch.randint(0,38,(1,)))
                m[:,:,:,t:t+8]=0;m[:,:,f:f+2,:]=0
                opt.zero_grad();loss=(ce(net(m,qq[ids]),yy[ids])*ww[ids]).mean();loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            if stop is not None:
                net.eval()
                with torch.no_grad():val=float((ce(net(sm,sq),sy)*sw).mean())
                if val<best-1e-4:best=val;best_epoch=epoch;stale=0
                else:stale+=1
                if stale>=8:break
        bundle['weights']={k:v.detach().numpy() for k,v in net.state_dict().items()}
        bundle['epochs']=best_epoch if stop is not None else epochs
    return bundle

def probability(bundle,x,mel):
    scores=np.asarray(raw_scores(bundle,x,mel),dtype='float64')
    return softmax(scores/bundle['temperature'],axis=1)

def fit_calibrated(frame,x,mel,config,seed):
    a,b=splits(frame,5,seed)[0];y=frame.label.to_numpy(int);src=frame.source.to_numpy()
    epochs=60;trial_info={}
    if config['family']=='CNN':
        c,d=splits(frame.iloc[a],4,seed+1)[0]
        trial=fit_raw(x[a[c]],mel[a[c]],y[a[c]],src[a[c]],config,seed,stop=(x[a[d]],mel[a[d]],y[a[d]],src[a[d]]))
        epochs=trial['epochs']
        trial_info={'stopping_train':frame.iloc[a[c]].speaker_id.tolist(),'stopping_validation':frame.iloc[a[d]].speaker_id.tolist()}
    model=fit_raw(x[a],mel[a],y[a],src[a],config,seed,epochs=epochs)
    scores=np.asarray(raw_scores(model,x[b],mel[b]),dtype='float64')
    def objective(t):
        p=softmax(scores/np.exp(t),axis=1)
        return np.mean([log_loss(y[b][src[b]==s],p[src[b]==s],labels=[0,1]) for s in SOURCES])
    opt=minimize_scalar(objective,bounds=(-2,6),method='bounded')
    model['temperature']=float(np.exp(opt.x))
    model['partition_audit']={'fitting':frame.iloc[a].speaker_id.tolist(),'calibration':frame.iloc[b].speaker_id.tolist(),**trial_info}
    return model

def metric(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);p=p/p.sum(1,keepdims=True);pred=p.argmax(1)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    auc=float(roc_auc_score(y,p[:,1])) if len(np.unique(y))==2 else None
    bins=np.minimum((p[:,1]*10).astype(int),9);ece=0.
    for k in range(10):
        mask=bins==k
        if mask.any():ece+=mask.mean()*abs(p[mask,1].mean()-y[mask].mean())
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
        sensitivity=float(tp/(tp+fn)) if tp+fn else None,specificity=float(tn/(tn+fp)) if tn+fp else None,
        roc_auc=auc,brier=float(brier_score_loss(y,p[:,1])),log_loss=float(log_loss(y,p,labels=[0,1])),ece=float(ece),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def source_score(frame,p):
    return float(np.mean([balanced_accuracy_score(frame.loc[frame.source==s,'label'],p[frame.source.to_numpy()==s].argmax(1)) for s in SOURCES]))

def choose(rows):
    best=max(r['score'] for r in rows);eligible=[r for r in rows if r['score']>=best-.01]
    return min(eligible,key=lambda r:({'GMM':0,'MLP':1,'CNN':2,'Ensemble':3}[r['family']],r['order']))

def rejection(frame,p):
    curve=[]
    for threshold in (.6,.7,.8,.9):
        eligible=True;row={'threshold':threshold}
        for source in SOURCES:
            mask=frame.source.to_numpy()==source;keep=p[mask].max(1)>=threshold
            acc=float((p[mask][keep].argmax(1)==frame.label.to_numpy()[mask][keep]).mean()) if keep.any() else None
            row[source]={'accepted':int(keep.sum()),'coverage':float(keep.mean()),'accuracy':acc}
            eligible &= bool(keep.sum()>=20 and keep.mean()>=.2 and acc is not None and acc>=.8)
        row['eligible']=eligible;curve.append(row)
    return min((r['threshold'] for r in curve if r['eligible']),default=1.01),curve

def train():
    global DEADLINE
    protocol=verify_protocol()
    if (OUT/'release.json').exists():raise RuntimeError('Completed release is frozen.')
    timing=OUT/'training_clock.json'
    if not timing.exists():dump(timing,{'started_unix':time.time(),'budget_seconds':36000})
    clock=json.loads(timing.read_text());DEADLINE=clock['started_unix']+clock['budget_seconds'];check_time()
    frame=pd.read_csv(DATA/'featured_manifest.csv');fm=json.loads((OUT/'feature_manifest.json').read_text())
    if sha(DATA/'featured_manifest.csv')!=fm['manifest_hash']:raise ValueError('Feature manifest changed')
    arrays={}
    for sr in (16000,8000):
        if sha(DATA/f'features_{sr}.npz')!=fm['arrays'][str(sr)]['file_hash']:raise ValueError('Feature array changed')
        with np.load(DATA/f'features_{sr}.npz') as z:arrays[sr]=(z['x'],z['mel'])
    configs=candidates();fingerprint=key([protocol,fm]);cache=DATA/'model_cache';cache.mkdir(exist_ok=True)
    comparisons=[];outer_rows=[];fold_info=[]
    def fit(ids,config,seed,tag):
        check_time();ident=key([fingerprint,frame.iloc[ids].record_id.tolist(),config,seed])
        dest=cache/(ident+'.joblib')
        if dest.exists():bundle=joblib.load(dest)
        else:
            print('FIT',tag,config['id'],len(ids),flush=True)
            x,m=arrays[config['sr']];bundle=fit_calibrated(frame.iloc[ids].reset_index(drop=True),x[ids],m[ids],config,seed)
            joblib.dump(bundle,dest)
        return bundle
    def predict_ids(bundle,ids):
        x,m=arrays[bundle['config']['sr']];return probability(bundle,x[ids],m[ids])
    def select(ids,seed,tag):
        part=frame.iloc[ids].reset_index(drop=True);folds=splits(part,3,seed);ps={};ranking=[]
        for order,c in enumerate(configs):
            p=np.zeros((len(ids),2))
            for fold,(a,b) in enumerate(folds):
                model=fit(ids[a],c,seed+fold,tag+f'/inner{fold}')
                assert not set(frame.iloc[ids[b]].speaker_id)&set(model['partition_audit']['fitting']+model['partition_audit']['calibration'])
                p[b]=predict_ids(model,ids[b])
            ps[c['id']]=p;row={'stage':tag,'id':c['id'],'family':c['family'],'order':order,'score':source_score(part,p)}
            ranking.append(row)
            for src in ['pooled']+SOURCES:
                mask=np.ones(len(part),bool) if src=='pooled' else part.source.to_numpy()==src
                comparisons.append({**row,'source':src,**metric(part.label.to_numpy()[mask],p[mask])})
            pd.DataFrame(comparisons).to_csv(OUT/'candidate_comparison.csv',index=False)
            print('SCORE',tag,c['id'],row['score'],flush=True)
        g=choose([r for r in ranking if r['family']=='GMM']);n=choose([r for r in ranking if r['family']!='GMM'])
        eid='Ensemble';ps[eid]=(ps[g['id']]+ps[n['id']])/2
        erow={'stage':tag,'id':eid,'family':'Ensemble','order':len(configs),'score':source_score(part,ps[eid])}
        ranking.append(erow)
        for src in ['pooled']+SOURCES:
            mask=np.ones(len(part),bool) if src=='pooled' else part.source.to_numpy()==src
            comparisons.append({**erow,'source':src,**metric(part.label.to_numpy()[mask],ps[eid][mask])})
        winner=choose(ranking);members=[g['id'],n['id']] if winner['id']==eid else [winner['id']]
        threshold,curve=rejection(part,ps[winner['id']])
        return {'winner':winner,'members':members,'best_gmm':g['id'],'best_dnn':n['id'],'threshold':threshold,
            'threshold_curve':curve,'ranking':ranking,'inner_splits':[{'train':part.iloc[a].speaker_id.tolist(),
            'validation':part.iloc[b].speaker_id.tolist()} for a,b in folds]}
    outer=splits(frame,4,SEED)
    for fold,(a,b) in enumerate(outer):
        selected=select(a,SEED+100*fold,f'outer{fold}')
        bundles={ident:fit(a,next(c for c in configs if c['id']==ident),SEED+1000+fold,'outer-refit')
            for ident in set(selected['members']+[selected['best_gmm'],selected['best_dnn']])}
        ps={ident:predict_ids(bundle,b) for ident,bundle in bundles.items()}
        chosen=np.mean([ps[ident] for ident in selected['members']],axis=0)
        for role,p in [('selection',chosen),('GMM',ps[selected['best_gmm']]),('DNN',ps[selected['best_dnn']])]:
            for i,idx in enumerate(b):
                outer_rows.append({'record_id':frame.iloc[idx].record_id,'speaker_id':frame.iloc[idx].speaker_id,
                    'source':frame.iloc[idx].source,'label':int(frame.iloc[idx].label),'fold':fold,'role':role,
                    'p_pathological':float(p[i,1]),'threshold':selected['threshold'],
                    'accepted':bool(p[i].max()>=selected['threshold']) if role=='selection' else False})
        fold_info.append({'fold':fold,'train':frame.iloc[a].speaker_id.tolist(),'validation':frame.iloc[b].speaker_id.tolist(),
            'selection':selected,'model_partitions':{k:v['partition_audit'] for k,v in bundles.items()}})
        pd.DataFrame(outer_rows).to_csv(OUT/'nested_predictions.csv',index=False);dump(OUT/'nested_folds.json',fold_info)
    final=select(np.arange(len(frame)),SEED+2000,'final-development')
    ids=set(final['members']+[final['best_gmm'],final['best_dnn']])
    models={ident:fit(np.arange(len(frame)),next(c for c in configs if c['id']==ident),SEED+3000,'final-refit') for ident in ids}
    bundle={'version':VERSION,'labels':LABELS,'selection':final,'models':models,'feature_specs':{str(sr):spec_for(sr) for sr in arrays},
        'source_hashes':source_hashes(),'manifest_hash':fm['manifest_hash'],'evaluation_scope':'Nested development CV; no untouched external test',
        'known_hashes':{str(sr):frame[f'audio_hash_{sr}'].tolist() for sr in arrays}}
    joblib.dump(bundle,OUT/'binary.joblib');loaded=joblib.load(OUT/'binary.joblib')
    for ident in models:np.testing.assert_allclose(predict_ids(models[ident],np.arange(3)),predict_ids(loaded['models'][ident],np.arange(3)))
    pd.DataFrame(comparisons).to_csv(OUT/'candidate_comparison.csv',index=False)
    dump(OUT/'release.json',{'status':'complete','version':VERSION,'selection':final,'model_sha256':sha(OUT/'binary.joblib'),
        'source_hashes':source_hashes(),'elapsed_training_seconds':time.time()-clock['started_unix']})
    print('COMPLETE',final['winner']['id'],'threshold',final['threshold'],flush=True)

def report():
    """Summarize saved outer predictions only; never rerun models for reporting."""
    predictions=pd.read_csv(OUT/'nested_predictions.csv');frame=pd.read_csv(DATA/'featured_manifest.csv')
    if predictions[predictions.role=='selection'].speaker_id.nunique()!=len(frame):raise ValueError('Incomplete outer evaluation')
    rows=[];summary={};rng=np.random.default_rng(SEED)
    for role in ['selection','GMM','DNN']:
        for source in ['pooled']+SOURCES:
            d=predictions[(predictions.role==role)&((predictions.source==source) if source!='pooled' else True)]
            y=d.label.to_numpy();p=np.c_[1-d.p_pathological,d.p_pathological]
            result=metric(y,p);folds=[metric(z.label.to_numpy(),np.c_[1-z.p_pathological,z.p_pathological]) for _,z in d.groupby('fold')]
            row={'role':role,'source':source,'n':len(d),**result}
            for name in ['accuracy','balanced_accuracy','macro_f1','sensitivity','specificity','roc_auc']:
                row[name+'_mean']=float(np.mean([f[name] for f in folds]));row[name+'_std']=float(np.std([f[name] for f in folds],ddof=1))
            if role=='selection':
                keep=d.accepted.to_numpy(bool);row['coverage']=float(keep.mean());row['accepted_count']=int(keep.sum())
                row['accepted_accuracy']=float((p[keep].argmax(1)==y[keep]).mean()) if keep.any() else None
                groups=[np.flatnonzero((d.source.to_numpy()==s)&(y==c)) for s in d.source.unique() for c in (0,1)]
                boot={k:[] for k in ['accuracy','balanced_accuracy','macro_f1','sensitivity','specificity','roc_auc']}
                for _ in range(1000):
                    idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);m=metric(y[idx],p[idx])
                    for k in boot:boot[k].append(m[k])
                row['confidence_intervals']={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in boot.items()}
                summary[source]=row
                pd.DataFrame(classification_report(y,p.argmax(1),target_names=LABELS,output_dict=True,zero_division=0)).T.to_csv(OUT/f'classification_{source}.csv')
                from svd_advanced import report_confusion
                report_confusion(y,p,LABELS,OUT/f'confusion_{source}.png',f'{source}: nested development CV')
            rows.append(row)
    pd.DataFrame([{k:v for k,v in r.items() if k!='confidence_intervals'} for r in rows]).to_csv(OUT/'nested_metrics.csv',index=False)
    dump(OUT/'nested_summary.json',summary)
    merged=predictions[predictions.role=='selection'].merge(frame[['record_id','age_band','sex','age','diagnosis']],on='record_id')
    sub=[]
    for attribute in ('sex','age_band'):
        for (source,value),d in merged.groupby(['source',attribute]):
            both=d.label.nunique()==2
            counts=d.label.value_counts();small=not both or counts.min()<20
            sub.append({'source':source,'attribute':attribute,'value':value,'n':len(d),'healthy':int(sum(d.label==0)),
                'pathological':int(sum(d.label==1)),'inconclusive_small_group':bool(small),
                **(metric(d.label.to_numpy(),np.c_[1-d.p_pathological,d.p_pathological]) if both else {})})
    pd.DataFrame(sub).to_csv(OUT/'subgroup_metrics.csv',index=False)
    merged['predicted']=(merged.p_pathological>=.5).astype(int)
    merged[merged.predicted!=merged.label].to_csv(OUT/'development_errors.csv',index=False)
    old=json.loads((OUT/'historical_v2_hashes.json').read_text())
    if any(sha(ROOT/p)!=h for p,h in old.items()):raise ValueError('Historical experiment was modified')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','features','train','report'])
    parser.add_argument('--source-meta',type=Path)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.run_dir:DATA=args.run_dir.resolve()/'data';OUT=args.run_dir.resolve()/'artifacts'
    with threadpool_limits(limits=2):
        try:
            if args.stage=='prepare':prepare(args.source_meta)
            elif args.stage=='features':features()
            elif args.stage=='train':train()
            else:report()
        except Exception as exc:
            if OUT.exists():dump(OUT/'last_error.json',{'stage':args.stage,'error':repr(exc),'time':time.time()})
            raise

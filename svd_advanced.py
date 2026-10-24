"""Audited speaker-independent SVD v2 experiment. Run stages in order.

python svd_advanced.py prepare --source-meta /path/voice_data.csv
python svd_advanced.py features
python svd_advanced.py train
python svd_advanced.py evaluate

Old SVD test speakers are development. The original reserved 42 VOICED
recordings form the external binary test. Subtype claims are grouped-CV only.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import random
import time
import warnings
import joblib
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, log_loss, brier_score_loss)
from threadpoolctl import threadpool_limits
from svd_training import load_audio, extract_features, DEFAULT_DATA, SPEC

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts_svd_v2'
DATA = ROOT/'data/svd_v2'
SEED = 20261022
SUBTYPES = ['Functional dysphonia group', 'Laryngitis', 'Vocal fold paresis']
LABELS = {'Functional dysphonia group': {'Funktionelle Dysphonie','Hyperfunktionelle Dysphonie'},
          'Laryngitis': {'Laryngitis'}, 'Vocal fold paresis': {'Rekurrensparese'}}
FEATURE_VERSION = 'svd-v2-97features-mel40x128-20261022'

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(source_meta, archive_meta=None, archive_index=None, voiced_root=None):
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Protocol already frozen. Use the saved manifests; do not resample the holdout.')
    source = pd.read_csv(source_meta).fillna('').rename(columns={'AufnahmeID':'record_id',
        'SprecherID':'speaker_id','AufnahmeTyp':'status','Pathologien':'source_diagnoses'})
    supplied = pd.read_csv(DEFAULT_DATA/'recordings.csv').rename(columns={
        'File':'file','Speaker ID':'supplied_speaker','Diagnosis':'supplied_diagnosis'})
    supplied['record_id'] = supplied.file.str.extract(r'^(\d+)').astype(int)
    merged = supplied.merge(source,on='record_id',how='left',validate='one_to_one')
    if merged.speaker_id.isna().any() or (merged.speaker_id!=merged.supplied_speaker).any():
        raise ValueError('Missing source row or incorrect speaker identity')
    mixed = set(source.groupby('speaker_id').status.nunique().loc[lambda s:s>1].index)
    prior = pd.read_csv(ROOT/'artifacts_svd_local/usable_recordings.csv')
    merged = merged[merged.file.isin(prior.file) & ~merged.speaker_id.isin(mixed)].copy()
    def subtype(text):
        labels = {s.strip() for s in text.split(',') if s.strip()}
        matches = [name for name, group in LABELS.items() if labels and labels<=group]
        return matches[0] if len(matches)==1 else ''
    merged['subtype'] = merged.source_diagnoses.map(subtype)
    merged['binary'] = (merged.status=='p').astype(int)
    merged['path'] = merged.file.map(lambda f:str((DEFAULT_DATA/'audio_wav'/f).resolve()))
    # Retain earliest valid session per person; no acoustic/model-based selection.
    merged = merged.sort_values('record_id').drop_duplicates('speaker_id').reset_index(drop=True)
    columns = ['file','record_id','speaker_id','supplied_diagnosis','source_diagnoses','binary','subtype','path']
    merged[columns].to_csv(DATA/'development.csv',index=False)
    audit = merged.groupby(['supplied_diagnosis','source_diagnoses'],dropna=False).agg(
        recordings=('file','size'),unique_speakers=('speaker_id','nunique')).reset_index()
    audit.to_csv(OUT/'label_audit.csv',index=False)
    merged.groupby(['binary','subtype'],dropna=False).agg(recordings=('file','size'),
        unique_speakers=('speaker_id','nunique')).to_csv(OUT/'development_class_counts.csv')
    original_splits=json.loads((ROOT/'data/splits.json').read_text())
    voiced=pd.read_csv(ROOT/'data/manifest.csv').fillna('')
    test=voiced[voiced.record_id.isin(original_splits['test'])].copy()
    if len(test)!=42:raise ValueError('Expected original untouched VOICED holdout of 42 recordings')
    test['file']=test.record_id+'.hea'
    test['speaker_id']='VOICED:'+test.speaker_id.astype(str)
    test['binary']=test.label.astype(int)
    test['source_diagnoses']=test.diagnosis
    audio_root=Path(voiced_root) if voiced_root else ROOT/'data/raw'
    test['path']=test.record_id.map(lambda f:str((audio_root/(str(f)+'.hea')).resolve()))
    test[['file','record_id','speaker_id','source_diagnoses','binary','path']].to_csv(DATA/'external_test_frozen.csv',index=False)
    dump(OUT/'protocol.json', {'seed':SEED,'feature_version':FEATURE_VERSION,'outer_folds':4,
        'model_selection':'mean four-fold grouped CV balanced accuracy; fixed candidates and 50/50 best-family ensemble',
        'calibration':'temperature fit on speaker-disjoint inner calibration subset, never outer validation',
        'subtype_task':'Three pathology-only labels; exclude comorbid diagnoses; merge functional/hyperfunctional as a broad functional group',
        'prior_test_policy':'ALL previously evaluated SVD speakers are development; one earliest usable recording per speaker',
        'test_policy':'Original reserved VOICED 42 recordings (12 healthy, 30 pathological), never used to fit/tune these SVD models. Separate Italian cohort; identities based on release participant proxies.',
        'subtype_final_test':'No independent subtype test with all three classes is available. Report grouped CV only.',
        'unavailable_svd_controls':'173 unused healthy speaker IDs exist in metadata, but their normal-pitch /a/ is absent from data.zip and healthy.zip; do not fabricate a new SVD holdout.',
        'source_metadata_sha256':digest(source_meta),'development_sha256':digest(DATA/'development.csv'),
        'external_test_sha256':digest(DATA/'external_test_frozen.csv'), 'previously_used_speakers':int(supplied.supplied_speaker.nunique())})
    print('Development',len(merged),'external binary holdout',len(test),flush=True)
    print(merged.groupby(['binary','subtype']).size().to_string(),flush=True)


def extract_advanced(path):
    base,meta=extract_features(path)
    x,_=load_audio(path)
    sr=16000
    power=np.abs(librosa.stft(x,n_fft=512,win_length=400,hop_length=160,center=False))**2
    mel=librosa.feature.melspectrogram(S=power,sr=sr,n_fft=512,n_mels=40,fmin=50,fmax=8000)
    logmel=librosa.power_to_db(mel,ref=1.)
    mfcc=librosa.feature.mfcc(S=logmel,n_mfcc=13)
    additions=[]
    for order in (1,2):
        delta=librosa.feature.delta(mfcc,order=order,width=9)
        additions.extend(delta.mean(axis=1)); additions.extend(delta.std(axis=1))
    sound=parselmouth.Sound(x,sampling_frequency=sr)
    quality=np.full(3,np.nan)
    try:
        pc=call(sound,'To PowerCepstrogram',60,.002,4000,50)
        quality[0]=call(pc,'Get CPPS',True,.01,.001,60,500,.05,'Parabolic',.001,.05,'Straight','Robust')
    except parselmouth.PraatError: pass
    try:
        pulses=call(sound,'To PointProcess (periodic, cc)',60,500)
        quality[1]=call(pulses,'Get jitter (local)',0,0,1/500,1/60,1.3)
        quality[2]=call([sound,pulses],'Get shimmer (local)',0,0,1/500,1/60,1.3,1.6)
    except parselmouth.PraatError: pass
    rich=np.concatenate([base,np.array(additions),quality])
    rich[~np.isfinite(rich)]=np.nan
    # Fixed-width network input through time-axis interpolation; never duplicates speakers.
    mel128=np.stack([np.interp(np.linspace(0,1,128),np.linspace(0,1,len(v)),v) for v in logmel]).astype('float32')
    ahash=hashlib.sha256(np.round(x*1e6).astype('<i4').tobytes()).hexdigest()
    return rich,mel128,ahash

def cache_features():
    frame=pd.read_csv(DATA/'development.csv').fillna('')
    cache=DATA/'cache';cache.mkdir(exist_ok=True)
    rows=[];vectors=[];mels=[];failures=[]
    for i,row in enumerate(frame.to_dict('records')):
        path=cache/f"{row['record_id']}_{FEATURE_VERSION}.npz"
        try:
            if path.exists():
                with np.load(path) as d: rich,mel,ahash=d['rich'],d['mel'],str(d['audio_hash'])
            else:
                rich,mel,ahash=extract_advanced(row['path'])
                np.savez_compressed(path,rich=rich,mel=mel,audio_hash=ahash)
            if any(r['audio_hash']==ahash for r in rows):
                raise ValueError('Duplicate analyzed audio; first retained')
            rows.append({**row,'audio_hash':ahash});vectors.append(rich);mels.append(mel)
        except (ValueError,parselmouth.PraatError) as exc:
            failures.append({**row,'error':str(exc)})
        if (i+1)%100==0: print('Features',i+1,'/',len(frame),flush=True)
    pd.DataFrame(rows).to_csv(DATA/'development_features.csv',index=False)
    pd.DataFrame(failures).to_csv(OUT/'feature_exclusions.csv',index=False)
    np.savez_compressed(DATA/'development_features.npz',rich=np.stack(vectors),mel=np.stack(mels))
    print('Feature shape',np.stack(vectors).shape,flush=True)

def groupsplit(y,groups,n=4,seed=SEED):
    result=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))
    for a,b in result:
        if set(groups[a])&set(groups[b]):raise RuntimeError('Speaker overlap')
        if len(np.unique(y[a]))!=len(np.unique(y)) or len(np.unique(y[b]))!=len(np.unique(y)):
            raise ValueError('A fold lacks a class')
    return result

def candidates():
    return [
      {'id':'GMM_baseline','family':'GMM','features':'base','components':1,'covariance':'diag','pca':0},
      {'id':'GMM_rich_2','family':'GMM','features':'rich','components':2,'covariance':'diag','pca':20},
      {'id':'GMM_rich_4','family':'GMM','features':'rich','components':4,'covariance':'diag','pca':20},
      {'id':'GMM_tied_2','family':'GMM','features':'rich','components':2,'covariance':'tied','pca':12},
      {'id':'MLP_baseline','family':'MLP','features':'base','layers':(64,32),'alpha':.1,'select':0},
      {'id':'MLP_rich_small','family':'MLP','features':'rich','layers':(32,16),'alpha':1.,'select':40},
      {'id':'MLP_rich_large','family':'MLP','features':'rich','layers':(128,64),'alpha':.1,'select':0},
      {'id':'CNN_mel','family':'CNN','features':'rich','width':8,'dropout':.3,'augment':False},
      {'id':'CNN_mel_aug','family':'CNN','features':'rich','width':16,'dropout':.5,'augment':True},
    ]

def transform_fit(x,y,config):
    raw=x[:,:42] if config['features']=='base' else x
    imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(raw)
    filled=imputer.transform(raw)
    scaler=StandardScaler().fit(filled)
    z=scaler.transform(filled)
    selector=None
    if config.get('select'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            selector=SelectKBest(f_classif,k=min(config['select'],z.shape[1])).fit(z,y)
        z=selector.transform(z)
    pca=None
    if config.get('pca'):
        pca=PCA(n_components=min(config['pca'],len(z)-1,z.shape[1]),whiten=True,random_state=SEED).fit(z)
        z=pca.transform(z)
    return z,{'imputer':imputer,'scaler':scaler,'selector':selector,'pca':pca}

def transform(bundle,x):
    raw=x[:,:42] if bundle['config']['features']=='base' else x
    z=bundle['scaler'].transform(bundle['imputer'].transform(raw))
    if bundle['selector'] is not None:z=bundle['selector'].transform(z)
    if bundle['pca'] is not None:z=bundle['pca'].transform(z)
    return z

def cnn_network(width,dropout,dim,classes):
    import torch
    from torch import nn
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Conv2d(1,width,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(width,width*2,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d((2,2)))
            self.head=nn.Sequential(nn.Linear(width*8+dim,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,classes))
        def forward(self,m,q):return self.head(torch.cat([self.encoder(m).flatten(1),q],dim=1))
    return Network()

def fit_raw(x,mel,y,config,seed,stop=None,epochs=60):
    z,pre=transform_fit(x,y,config)
    bundle={'config':config,**pre,'classes':int(max(y)+1),'temperature':1.}
    if config['family']=='GMM':
        models=[]
        for c in range(bundle['classes']):
            model=GaussianMixture(n_components=config['components'],covariance_type=config['covariance'],
                reg_covar=.1,n_init=3,max_iter=400,random_state=seed).fit(z[y==c])
            if not model.converged_:raise RuntimeError('GMM failed to converge')
            models.append(model)
        bundle['model']=models
    elif config['family']=='MLP':
        model=MLPClassifier(hidden_layer_sizes=config['layers'],alpha=config['alpha'],max_iter=350,
            batch_size=32,learning_rate_init=.001,early_stopping=False,n_iter_no_change=25,random_state=seed)
        weights=len(y)/(len(np.unique(y))*np.bincount(y)[y])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(z,y,sample_weight=weights)
        bundle['model']=model
        bundle['warnings']=[str(w.message) for w in caught]
    else:
        import torch
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        bundle['mel_mean']=mel.mean(axis=(0,2),keepdims=True)
        bundle['mel_std']=np.maximum(mel.std(axis=(0,2),keepdims=True),1.)
        m=torch.tensor(((mel-bundle['mel_mean'])/bundle['mel_std'])[:,None],dtype=torch.float32)
        q=torch.tensor(z,dtype=torch.float32);labels=torch.tensor(y,dtype=torch.long)
        net=cnn_network(config['width'],config['dropout'],z.shape[1],bundle['classes'])
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(bundle['classes']*np.bincount(y)),dtype=torch.float32))
        best_loss=float('inf');best_epoch=1;stale=0
        for epoch in range(1,epochs+1):
            net.train()
            for ids in torch.randperm(len(y)).split(32):
                mm=m[ids].clone()
                if config['augment']:
                    # Training-only mild spectrogram noise and short masks; no invented speakers.
                    mm+=torch.randn_like(mm)*.03
                    t=int(torch.randint(0,120,(1,)));mm[:,:,:,t:t+8]=0
                    f=int(torch.randint(0,38,(1,)));mm[:,:,f:f+2,:]=0
                opt.zero_grad();loss=criterion(net(mm,q[ids]),labels[ids]);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
                opt.step()
            if stop is not None:
                net.eval()
                with torch.no_grad():
                    ss=torch.tensor(((stop[1]-bundle['mel_mean'])/bundle['mel_std'])[:,None],dtype=torch.float32)
                    qq=torch.tensor(transform(bundle,stop[0]),dtype=torch.float32)
                    val=float(criterion(net(ss,qq),torch.tensor(stop[2],dtype=torch.long)))
                if val<best_loss-1e-4:best_loss=val;best_epoch=epoch;stale=0
                else:stale+=1
                if stale>=8:break
        bundle['weights']={k:v.detach().numpy() for k,v in net.state_dict().items()}
        bundle['epochs']=best_epoch if stop is not None else epochs
    return bundle

def raw_scores(bundle,x,mel):
    z=transform(bundle,x)
    family=bundle['config']['family']
    if family=='GMM':return np.column_stack([m.score_samples(z) for m in bundle['model']])
    if family=='MLP':return np.log(np.clip(bundle['model'].predict_proba(z),1e-9,1))
    import torch
    net=bundle.get('_runtime')
    if net is None:
        net=cnn_network(bundle['config']['width'],bundle['config']['dropout'],z.shape[1],bundle['classes'])
        net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in bundle['weights'].items()});net.eval()
    with torch.no_grad():
        mm=torch.tensor(((mel-bundle['mel_mean'])/bundle['mel_std'])[:,None],dtype=torch.float32)
        return net(mm,torch.tensor(z,dtype=torch.float32)).numpy()

def predict(bundle,x,mel):
    if 'members' in bundle:return np.mean([predict(m,x,mel) for m in bundle['members']],axis=0)
    return softmax(raw_scores(bundle,x,mel)/bundle['temperature'],axis=1)

def fit_calibrated(x,mel,y,groups,config,seed):
    fitting,calibration=groupsplit(y,groups,5,seed)[0]
    epochs=60
    if config['family']=='CNN':
        a,b=groupsplit(y[fitting],groups[fitting],4,seed+1)[0]
        trial=fit_raw(x[fitting[a]],mel[fitting[a]],y[fitting[a]],config,seed,
            stop=(x[fitting[b]],mel[fitting[b]],y[fitting[b]]))
        epochs=trial['epochs']
    bundle=fit_raw(x[fitting],mel[fitting],y[fitting],config,seed,epochs=epochs)
    scores=raw_scores(bundle,x[calibration],mel[calibration])
    # Temperature preserves argmax but improves score calibration on separate speakers.
    objective=lambda logt:log_loss(y[calibration],softmax(scores/np.exp(logt),axis=1),labels=np.arange(bundle['classes']))
    optimum=minimize_scalar(objective,bounds=(-2,6),method='bounded')
    bundle['temperature']=float(np.exp(optimum.x))
    bundle['fitting_speakers']=groups[fitting].tolist();bundle['calibration_speakers']=groups[calibration].tolist()
    return bundle

def measure(y,p):
    # Float32 CNN softmax can sum to 1 +/- 1e-7. Normalize for metrics only.
    p=np.asarray(p,dtype=np.float64)
    p=p/p.sum(axis=1,keepdims=True)
    pred=p.argmax(axis=1);k=p.shape[1]
    cm=confusion_matrix(y,pred,labels=np.arange(k))
    sensitivity=np.diag(cm)/np.maximum(cm.sum(axis=1),1)
    tn=cm.sum()-cm.sum(axis=0)-cm.sum(axis=1)+np.diag(cm)
    specificity=tn/np.maximum(cm.sum()-cm.sum(axis=1),1)
    return {'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'sensitivity':float(sensitivity[1] if k==2 else sensitivity.mean()),
        'specificity':float(specificity[1] if k==2 else specificity.mean()),
        'roc_auc':float(roc_auc_score(y,p[:,1])) if k==2 else float(roc_auc_score(y,p,multi_class='ovr',average='macro')),
        'log_loss':float(log_loss(y,p,labels=np.arange(k)))}

def report_confusion(y,p,labels,path,title):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    cm=confusion_matrix(y,p.argmax(axis=1),labels=np.arange(len(labels)))
    fig,ax=plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,cmap='Blues',colorbar=False,xticks_rotation=20)
    ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
    return cm.tolist()

def train():
    if (OUT/'selection_frozen.json').exists():raise RuntimeError('Selection already frozen; use existing run.')
    frame=pd.read_csv(DATA/'development_features.csv').fillna('')
    with np.load(DATA/'development_features.npz') as d:x,mel=d['rich'],d['mel']
    all_summaries=[];selections={}
    for task in ('binary','subtype'):
        ids=np.arange(len(frame)) if task=='binary' else np.flatnonzero(frame.subtype.isin(SUBTYPES))
        part=frame.iloc[ids].reset_index(drop=True)
        labels=['Healthy','Pathological'] if task=='binary' else SUBTYPES
        y=part.binary.to_numpy(dtype=int) if task=='binary' else np.array([labels.index(v) for v in part.subtype])
        xx,mm=x[ids],mel[ids];groups=part.speaker_id.to_numpy(dtype=int)
        splits=groupsplit(y,groups,4)
        dump(OUT/f'{task}_folds.json',[{'train_speakers':groups[a].tolist(),'validation_speakers':groups[b].tolist()} for a,b in splits])
        history=[];oofs={};bundles={}
        for config in candidates():
            name=config['id'];p=np.zeros((len(y),len(labels)));fold_rows=[];fitted=[]
            for fold,(a,b) in enumerate(splits):
                cache=DATA/'cv_models'/f'{task}_{name}_{fold}.joblib';cache.parent.mkdir(exist_ok=True)
                print(task,name,'fold',fold+1,flush=True)
                if cache.exists(): bundle=joblib.load(cache)
                else:
                    bundle=fit_calibrated(xx[a],mm[a],y[a],groups[a],config,SEED+fold)
                    joblib.dump(bundle,cache)
                assert not set(groups[b])&(set(bundle['fitting_speakers'])|set(bundle['calibration_speakers']))
                p[b]=predict(bundle,xx[b],mm[b]);fitted.append(bundle)
                fold_rows.append({'task':task,'model':name,'fold':fold,**measure(y[b],p[b])})
            row={'task':task,'model':name,'family':config['family']}
            for metric in measure(y,p):
                row[metric+'_mean']=float(np.mean([r[metric] for r in fold_rows]))
                row[metric+'_std']=float(np.std([r[metric] for r in fold_rows],ddof=1))
            history.extend(fold_rows);all_summaries.append(row);oofs[name]=p;bundles[name]=fitted
            print('CV',row,flush=True)
            pd.DataFrame(all_summaries).to_csv(OUT/'comparison.csv',index=False)
            pd.DataFrame(history).to_csv(OUT/f'{task}_fold_metrics.csv',index=False)
        ranked=[r for r in all_summaries if r['task']==task]
        gmm=max((r for r in ranked if r['family']=='GMM'),key=lambda r:r['balanced_accuracy_mean'])
        dnn=max((r for r in ranked if r['family']!='GMM'),key=lambda r:r['balanced_accuracy_mean'])
        ensemble=(oofs[gmm['model']]+oofs[dnn['model']])/2
        erows=[measure(y[b],ensemble[b]) for a,b in splits]
        erow={'task':task,'model':'Ensemble_50_50','family':'Ensemble'}
        for metric in erows[0]:
            erow[metric+'_mean']=float(np.mean([r[metric] for r in erows]))
            erow[metric+'_std']=float(np.std([r[metric] for r in erows],ddof=1))
        all_summaries.append(erow);ranked.append(erow);oofs[erow['model']]=ensemble
        winner=max(ranked,key=lambda r:r['balanced_accuracy_mean'])
        name=winner['model'];prob=oofs[name]
        model_ids=[gmm['model'],dnn['model']] if name=='Ensemble_50_50' else [name]
        # Selective accuracy and coverage are both reported; cannot promise a reliability target.
        curve=[]
        for threshold in (.5,.6,.7,.8,.9,.95):
            keep=prob.max(axis=1)>=threshold
            curve.append({'threshold':threshold,'coverage':float(keep.mean()),'accepted_count':int(keep.sum()),
                          'accuracy_accepted':float((prob.argmax(axis=1)[keep]==y[keep]).mean()) if keep.any() else None})
        eligible=[r for r in curve if r['accepted_count']>=30 and r['accuracy_accepted']>=.8 and r['coverage']>=.2]
        threshold=min((r['threshold'] for r in eligible),default=1.01)
        selections[task]={'model':name,'members':model_ids,'labels':labels,'uncertainty_threshold':threshold,
            'selection':winner,'selective_curve':curve,'subtype_scope':'Only three supported pathology labels; other causes cannot be ruled out'}
        pd.DataFrame(curve).to_csv(OUT/f'{task}_selective_curve.csv',index=False)
        predframe=part[['file','speaker_id','source_diagnoses']].copy()
        predframe['actual']=[labels[i] for i in y];predframe['predicted']=[labels[i] for i in prob.argmax(axis=1)]
        for i,label in enumerate(labels):predframe[f'p_{label}']=prob[:,i]
        predframe.to_csv(OUT/f'{task}_cv_predictions.csv',index=False)
        report=classification_report(y,prob.argmax(axis=1),labels=np.arange(len(labels)),target_names=labels,output_dict=True,zero_division=0)
        pd.DataFrame(report).T.to_csv(OUT/f'{task}_cv_classification.csv')
        report_confusion(y,prob,labels,OUT/f'{task}_cv_confusion.png',f'{task}: grouped development CV (selection-biased)')
        final=[]
        for ident in model_ids:
            config=next(c for c in candidates() if c['id']==ident)
            final.append(fit_calibrated(xx,mm,y,groups,config,SEED+100))
        bundle=final[0] if len(final)==1 else {'members':final}
        bundle.update(labels=labels,feature_version=FEATURE_VERSION,selection=selections[task],
                      development_speakers=groups.tolist(),development_hashes=part.audio_hash.tolist())
        joblib.dump(bundle,OUT/f'{task}.joblib')
        np.testing.assert_allclose(predict(bundle,xx[:3],mm[:3]),predict(joblib.load(OUT/f'{task}.joblib'),xx[:3],mm[:3]))
    pd.DataFrame(all_summaries).to_csv(OUT/'comparison.csv',index=False)
    dump(OUT/'selection_frozen.json',selections)
    dump(OUT/'model_hashes.json',{p.name:digest(p) for p in OUT.glob('*.joblib')})

def evaluate():
    if (OUT/'final_test_started.json').exists():raise RuntimeError('Final test already started. Use saved results; never retune on it.')
    selected=json.loads((OUT/'selection_frozen.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    for filename,key in [('development.csv','development_sha256'),('external_test_frozen.csv','external_test_sha256')]:
        if digest(DATA/filename)!=protocol[key]:raise RuntimeError('Frozen manifest changed: '+filename)
    for filename,expected in json.loads((OUT/'model_hashes.json').read_text()).items():
        if digest(OUT/filename)!=expected:raise RuntimeError('Frozen model changed: '+filename)
    test=pd.read_csv(DATA/'external_test_frozen.csv').fillna('')
    dev=pd.read_csv(DATA/'development_features.csv').fillna('')
    overlap=set(test.speaker_id)&set(dev.speaker_id)
    if overlap:raise RuntimeError('New test speaker overlap')
    # Evaluation begins after selection and model hashes are frozen.
    dump(OUT/'final_test_started.json',{'speaker_count':len(test),'selection_sha256':digest(OUT/'selection_frozen.json'),
        'evaluation_source_sha256':digest(__file__), 'feature_source_sha256':digest(ROOT/'svd_training.py')})
    bundle=joblib.load(OUT/'binary.joblib')
    hashes=[];scores=[]
    for row in test.to_dict('records'):
        import wfdb
        record=wfdb.rdrecord(str(Path(row['path']).with_suffix('')))
        encoded=io.BytesIO()
        sf.write(encoded,record.p_signal,int(record.fs),format='WAV',subtype='DOUBLE')
        rich,mel,ahash=extract_advanced(encoded.getvalue())
        if ahash in set(dev.audio_hash):raise RuntimeError('External audio duplicate overlaps development')
        scores.append(predict(bundle,rich[None],mel[None])[0]);hashes.append(ahash)
    p=np.stack(scores);y=test.binary.to_numpy(dtype=int)
    result=measure(y,p)
    rng=np.random.default_rng(SEED)
    boot=[]
    for _ in range(2000):
        idx=np.concatenate([rng.choice(np.flatnonzero(y==v),sum(y==v),replace=True) for v in (0,1)])
        boot.append(measure(y[idx],p[idx])['balanced_accuracy'])
    result['balanced_accuracy_95_ci']=np.percentile(boot,[2.5,97.5]).tolist()
    result['test_speakers']=len(test);result['test_healthy']=int(sum(y==0));result['test_pathological']=int(sum(y==1))
    result['confusion_matrix']=report_confusion(y,p,['Healthy','Pathological'],OUT/'external_binary_test_confusion.png','External VOICED reserved-speaker test')
    result['brier_score']=float(brier_score_loss(y,p[:,1]))
    threshold=selected['binary']['uncertainty_threshold'];keep=p.max(axis=1)>=threshold
    result['accepted_coverage']=float(keep.mean());result['accepted_accuracy']=float((p.argmax(axis=1)[keep]==y[keep]).mean()) if keep.any() else None
    result['scope']='External VOICED reserved recordings: 12 healthy, 30 pathological; 8 kHz acquisition resampled to 16 kHz. Small cross-database research evaluation, not clinical validation.'
    dump(OUT/'external_binary_test_metrics.json',result)
    pd.DataFrame(classification_report(y,p.argmax(axis=1),target_names=['Healthy','Pathological'],output_dict=True,zero_division=0)).T.to_csv(OUT/'external_binary_test_classification.csv')
    test['prediction']=p.argmax(axis=1);test['pathological_probability']=p[:,1]
    test['audio_hash']=hashes
    test.drop(columns=['path']).to_csv(OUT/'external_binary_test_predictions.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','features','train','evaluate'])
    parser.add_argument('--source-meta',type=Path)
    parser.add_argument('--archive-meta',type=Path)
    parser.add_argument('--archive-index',type=Path)
    parser.add_argument('--dataset-root',type=Path,default=DEFAULT_DATA,
        help='Directory containing recordings.csv and audio_wav/')
    parser.add_argument('--voiced-root',type=Path,help='Directory containing external VOICED .hea/.dat files')
    parser.add_argument('--run-dir',type=Path,help='Fresh experiment directory; never overwrite the frozen published run')
    args=parser.parse_args()
    DEFAULT_DATA=args.dataset_root
    if args.run_dir:
        DATA=args.run_dir.resolve()/'data';OUT=args.run_dir.resolve()/'artifacts'
        OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        if args.stage=='prepare':prepare(args.source_meta,args.archive_meta,args.archive_index,args.voiced_root)
        elif args.stage=='features':cache_features()
        elif args.stage=='train':train()
        else:evaluate()

import os
import glob
import numpy as np
#from scipy import interp, interpolate
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import figure

mean_fpr = np.linspace(0.0001, 1, 8000)
mean_rec = np.linspace(1, 0.0001, 8000)

# Repository root, relative to this script (works on any machine).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_roc_curve(pt_bin, folder, dataset):
    os.chdir(os.path.join(REPO_ROOT, folder))
    tprs = []
    aucs = []
    for file in glob.glob('output_training_all_*_roc.txt'):
#    for file in glob.glob('output_training_all_%s_*_roc.txt' % str(pt_bin)):
#    for file in glob.glob('train_%s_%s_*_roc.txt' % (str(pt_bin), str(dataset))):
        if 'pileup' in file and 'pileup' not in dataset:
            continue
        elif 'trackeff' in file and 'trackeff' not in dataset:
            continue
        fpr, tpr = np.loadtxt(file)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0001
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return mean_tpr, mean_auc, std_auc, std_tpr, tprs_upper, tprs_lower

def generate_pr_curve(pt_bin, folder, dataset):
    os.chdir(os.path.join(REPO_ROOT, folder))
    precs = []
    aucs = []
    for file in glob.glob('output_training_all_*_pr.txt'):
#    for file in glob.glob('output_training_all_%s_*_pr.txt' % str(pt_bin)):
#    for file in glob.glob('train_%s_%s_*_pr.txt' % (str(pt_bin), str(dataset))):
        if 'pileup' in file and 'pileup' not in dataset:
            continue
        elif 'trackeff' in file and 'trackeff' not in dataset:
            continue
        prec, rec = np.loadtxt(file)
        precs.append(np.interp(mean_rec[::-1], rec[::-1], prec[::-1]))
        precs[-1][0] = 1
        roc_auc = auc(rec, prec)
        aucs.append(roc_auc)

    mean_prec = np.mean(precs, axis=0)[::-1]
    mean_prec[-1] = 1
    mean_auc = auc(mean_rec, mean_prec)
    std_auc = np.std(aucs)
    std_prec = np.std(precs, axis=0)
    precs_upper = np.minimum(mean_prec + std_prec, 1)
    precs_lower = np.maximum(mean_prec - std_prec, 0)

    return mean_prec, mean_auc, std_auc, std_prec, precs_upper, precs_lower

r = 0.4
s = 'pp200GeV'
folder = 'hfvsl'
dataset = 'HardQCD'
jet = 'HF-jet vs udsg-jet '
#pt_bins = ['5_10', '10_15']
pt_bins = ['5to10']
for pt_bin in pt_bins:
    tpr_trkvtx, auc_trkvtx, std_auc_trkvtx, std_tpr_trkvtx, tprs_upper_trkvtx, tprs_lower_trkvtx = generate_roc_curve(pt_bin, 'training_runs/jetvlad_test_run', dataset.lower())
    prec_trkvtx, pr_auc_trkvtx, std_pr_auc_trkvtx, std_prec_trkvtx, precs_upper_trkvtx, precs_lower_trkvtx = generate_pr_curve(pt_bin, 'training_runs/jetvlad_test_run', dataset.lower())

    figure(num=None, figsize=(11, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(tpr_trkvtx, 1/(mean_fpr) , color='red', label=r'Tracking + Vertexing', lw=4, alpha=.8, linestyle='-')
    plt.fill_betweenx(1/(mean_fpr), tprs_lower_trkvtx, tprs_upper_trkvtx, color='red', alpha=.2)

    os.chdir(REPO_ROOT)


    plt.xlim([0, 1])
    plt.xticks(np.arange(0, 1.1, 0.1),fontsize=23)
    plt.yscale('log')
    plt.ylim([1e0, 1e5])
    plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4], fontsize=23)
    plt.grid(axis='x', color='0.95')
    plt.xlabel('Signal (Heavy-Flavor Jets) Efficiency', fontsize=23)
    plt.ylabel('Background (Light-Flavor Jets) Rejection', fontsize=23)
    plt.legend(loc="upper right",
               fancybox=True,
               framealpha=0.,
               fontsize=20,
               title='JetVLAD, Pythia8.235, %s\n' % (dataset if dataset == 'Balanced' else 'Cross-section weighted') + r'$\sqrt{s} = 200$ GeV, thermal bkg, anti-$k_{T}$ jets' +' R = %s\n' % r + r'$%s < p_{T,jet} < %s$ GeV/$c$, $|\eta_{jet}|$ < 0.6' % (pt_bin[:2], pt_bin[3:]),
               title_fontsize=20
               )
    fig = plt.gcf()
    plt.savefig("netvlad_rejection_pt_%s_%s.pdf" % (pt_bin, dataset.lower()), bbox_inches='tight')
    #plt.show()

    figure(num=None, figsize=(11, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(mean_rec, prec_trkvtx, color='red', label=r'Tracking + Vertexing', lw=4, alpha=.8, linestyle='-')
    plt.fill_between(mean_rec, precs_lower_trkvtx, precs_upper_trkvtx, color='red', alpha=.2)

    plt.xlim([0.2, 1])
    plt.ylim([-0.05, 1.05])
    plt.grid(axis='x', color='0.95')
    plt.xticks(np.arange(0.2, 1.1, 0.1),fontsize=23)
    plt.yticks(fontsize=23)
    plt.xlabel('Signal (Heavy-Flavor Jets) Efficiency', fontsize=23)
    plt.ylabel('Signal (Heavy-Flavor Jets) Purity', fontsize=23)
    plt.legend(loc=(0.025, 0.12),
               fancybox=True,
               framealpha=0.,
               fontsize=20,
               title='JetVLAD, Pythia8.314, %s\n' % (dataset if dataset == 'Balanced' else 'Cross-section weighted') + r'$\sqrt{s} = 200$ GeV, thermal bkg, anti-$k_{T}$ jets' +' R = %s\n' % r + r'$%s < p_{T,jet} < %s$ GeV/$c$, $|\eta_{jet}|$ < 0.6' % (pt_bin[:2], pt_bin[3:]),
               title_fontsize=20
               )
    fig = plt.gcf()
    plt.savefig("netvlad_purity_pt_%s_%s.pdf" % (pt_bin, dataset.lower()), bbox_inches='tight')

    print(pt_bin, dataset)
    idx_eff_81 = np.abs(0.80 - tpr_trkvtx).argmin()
    idx_pur_81 = np.abs(0.80 - mean_rec).argmin()
    idx_eff_50 = np.abs(0.50 - tpr_trkvtx).argmin()
    idx_pur_50 = np.abs(0.50 - mean_rec).argmin()
    print('Rejection@80eff: ', 1/mean_fpr[idx_eff_81])
    print('Prec@80eff: ', prec_trkvtx[idx_pur_81])
    print('Rejection@50eff: ', 1/mean_fpr[idx_eff_50])
    print('Prec@50eff: ', prec_trkvtx[idx_pur_50])

    print("")

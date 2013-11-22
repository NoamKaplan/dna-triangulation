
import functools

import sklearn.naive_bayes
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import triangulation
import collections
import argparse
import sys


def main():
    parser=argparse.ArgumentParser(description='Description',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-in',help='Hi-C interaction matrix',dest='infile',type=str,required=True)
    parser.add_argument('-out',help='prefix for output files',dest='outfile',type=str,required=True)
    parser.add_argument('-cv',help='evaluate by cross validation',dest='cv',action='store_true')
    parser.add_argument('-p',help='predict chromosome of unplace contigs',dest='predict_unplaced',action='store_true')
    parser.add_argument('-v',help='List of leave-out half-window sizes for CV (in bps)',dest='v_list',nargs='+',type=float,default=[0,0.5e6,1e6,2e6,5e6,10e6])
    parser.add_argument('-x',help='excluded chrs',dest='excluded_chrs',nargs='+',type=str,default=['chrM','chrY'])
    parser.add_argument('-pc',help='placed chrs',dest='placed_chrs',nargs='+',type=str,default=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15','chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22','chrX'])

      
    args=parser.parse_args()
      
    infile=args.infile
    outfile=args.outfile
    cv=args.cv
    predict_unplaced=args.predict_unplaced
    eval_on_train=args.eval_on_train
    v_list=args.v_list
    excluded_chrs=args.excluded_chrs
    placed_chrs=args.placed_chrs
    
    sys.stderr.write("Loading data\n")
    
    d,bin_chr,bin_position=triangulation.load_data_txt(infile,remove_nans=True,chrs=placed_chrs)
    bin_mean_position=np.mean(bin_position,1)
    chrs=np.unique(bin_chr)
    
    n=d.shape[0]

    d[np.diag_indices(n)]=0
    
    if cv:

        sys.stderr.write("Evaluating in cross-validation\n")
        
        d_sum=triangulation.func_reduce(d,bin_chr,func=np.sum).T
  
        for v in v_list:
            sys.stderr.write("leaving out bins within "+str(v)+" bps\n")
            
            predicted_chr=[]
            predicted_prob=[]

            for i in np.arange(n):
                
                eps=1e-8
               
                proximal_bins = (bin_chr==bin_chr[i]) & (bin_mean_position>=bin_mean_position[i]-v-eps) & (bin_mean_position<=bin_mean_position[i]+v+eps)

                train_vectors=d_sum.copy()
                train_vectors-=triangulation.func_reduce(d[proximal_bins,:],bin_chr[proximal_bins],func=np.sum,allkeys=chrs).T
                train_vectors/=triangulation.func_reduce(np.ones(len(~proximal_bins)),bin_chr[~proximal_bins],func=np.sum,allkeys=chrs).T
                train_vectors=train_vectors[~proximal_bins,:]
                train_labels=bin_chr[~proximal_bins]

                model=triangulation.AugmentationChrPredModel()

                model.fit(train_vectors,train_labels)

                test_d=d[i,~proximal_bins]
                test_bin_chr=bin_chr[~proximal_bins]

                test_vector=triangulation.average_reduce(test_d,test_bin_chr)

                pred_chr,pred_prob=model.predict(test_vector)
                predicted_chr.append(pred_chr[0])
                predicted_prob.append(pred_prob[0])
                
            predicted_chr=np.array(predicted_chr)
            predicted_prob=np.array(predicted_prob)
            np.savetxt(outfile+'_cvpred_v'+str(v)+'.tab',[bin_chr,bin_position,predicted_chr,predicted_prob],fmt='%s',delimiter='\t')


    if predict_unplaced:

        sys.stderr.write("predicting chromosome of unplaced contigs\n")
        
        # train on all data (without diagonal)
        model=triangulation.AugmentationChrPredModel()
       
        d_avg=triangulation.average_reduce(d,bin_chr).T
      
        model.fit(d_avg,bin_chr)
        
        d,bin_chr,bin_position=triangulation.load_data_txt(infile,remove_nans=True)

        chrs=np.unique(bin_chr)
    
        unplaced_chrs=np.unique((set(bin_chr)-set(placed_chrs))-set(excluded_chrs))
        
        unplaced_chr_bins=np.any(bin_chr[None].T==unplaced_chrs,1)

        d=d[unplaced_chr_bins,:]
        
        d_avg=triangulation.average_reduce(d.T,bin_chr).T

        d_avg=d_avg[:,np.any(chrs[None].T==np.array(placed_chrs),1)]

        pred_pos,pred_prob=model.predict(d_avg)
        
        res=np.c_[bin_chr[unplaced_chr_bins],bin_position[unplaced_chr_bins,:].astype(int),pred_pos,pred_prob]

        np.savetxt(outfile+'_predictions.tab',res,fmt='%s',delimiter='\t')
    

    
    

    
if __name__=="__main__":
    main()
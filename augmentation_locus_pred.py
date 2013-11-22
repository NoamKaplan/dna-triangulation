
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing
import triangulation
import sys
import argparse

def main():
    parser=argparse.ArgumentParser(description='locus prediction for genome augmentation from Hi-C data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-in',help='Hi-C interaction matrix input file',dest='infile',type=str,required=True)
    parser.add_argument('-out',help='prefix for out files',dest='outfile',type=str,required=True)
    parser.add_argument('-cv',help='evaluate in cross validation',dest='cv',action='store_true')
    parser.add_argument('-p',help='predict positions of unplaced contigs',dest='predict_unplaced',action='store_true')
    parser.add_argument('-v',help='List of leave-out half-window sizes for CV (in bps)',dest='v_list',nargs='+',type=float,default=[0,0.5e6,1e6,2e6,5e6,10e6])
    parser.add_argument('-xc',help='excluded chromosomes/contigs',dest='excluded_chrs',nargs='+',type=str,default=['chrM','chrY'])
    parser.add_argument('-pc',help='placed chromosomes',dest='placed_chrs',nargs='+',type=str,default=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15','chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22','chrX'])
    parser.add_argument('-pnum',help='numbers of processes to use for parallelizing CV',dest='pnum',type=int,default=1)
    parser.add_argument('-cf',help='file with chromosome assignment for each unplaced contig (contig_name\tchr)',dest='pred_chr_file',type=str)
    
    args=parser.parse_args()
      
    infile=args.infile
    outfile=args.outfile
    cv=args.cv
    predict_unplaced=args.predict_unplaced
    v_list=args.v_list
    excluded_chrs=args.excluded_chrs
    placed_chrs=args.placed_chrs
    pnum=args.pnum
    pred_chr_file=args.pred_chr_file
    
    sys.stderr.write("Loading data\n")
    
    d,bin_chr,bin_position=triangulation.load_data_txt(infile,remove_nans=True)
    bin_mean_position=np.mean(bin_position,1)
    chrs=np.unique(placed_chrs)

    unplaced_chrs=np.unique((set(bin_chr)-set(placed_chrs))-set(excluded_chrs))
    
    n=d.shape[0]

    d[np.diag_indices(n)]=0


    if cv:

        sys.stderr.write("Evaluating in cross-validation\n")

        for v in v_list:
            sys.stderr.write("leaving out bins within "+str(v)+" bps\n")

            fh=open(outfile+'_cvpred_v'+str(v)+'.tab','w')
            
            for c in ['chr20']:#np.unique(placed_chrs):
                sys.stderr.write("chr "+c+"\n")
                chr_bins=bin_chr==c
                chr_data=d[chr_bins,:][:,chr_bins].astype('float64')
                chr_bin_mean_position=bin_mean_position[chr_bins]
                chr_bin_num=np.sum(chr_bins)

                batch_size=chr_bin_num/pnum+1
                
 
                pool=multiprocessing.Pool(processes=pnum)
         
                jobs=[]
                
                for i in np.arange(0,chr_bin_num,batch_size):
                    
                    i_list=np.arange(i,min(i+batch_size,chr_bin_num))

                    jobs.append(pool.apply_async(cv_iter,args=[i_list,v,chr_bin_mean_position,chr_data]))

                pool.close()
                pool.join()
                
                predicted_pos=[]
                scales=[]
                for j in jobs:
                    predicted_pos+=j.get()[0]
                    scales+=j.get()[1]

                res=np.array([[c]*chr_bin_num,chr_bin_mean_position,predicted_pos,scales]).T
                
                np.savetxt(fh,res,fmt='%s',delimiter='\t')

            fh.close()

    if predict_unplaced:

        res=[]
        chr_bins={}
        chr_bin_mean_position={}
        chr_data={}
        
        models={}

        sys.stderr.write("training on placed contigs (estimating scale for each chromosome)...\n")

        
        for c in chrs:
                   
            chr_bins[c]=bin_chr==c
            chr_data[c]=d[chr_bins[c],:][:,chr_bins[c]].astype('float64')
            chr_bin_mean_position[c]=bin_mean_position[chr_bins[c]]

            models[c]=triangulation.AugmentationLocPredModel()
            models[c].estimate_scale(chr_bin_mean_position[c],chr_data[c])

        fh=open(pred_chr_file,'r')
        u_pred_chr_dict={}
        for line in fh:
            x=line.rstrip("\n").split("\t")
            u_pred_chr_dict[x[0]]=x[1]
        fh.close()

        sys.stderr.write("predicting on unplaced contigs...\n")
               
        unplaced_chr_bins=np.any(bin_chr[None].T==unplaced_chrs,1)
        placed_chr_bins=np.any(bin_chr[None].T==placed_chrs,1)

        for u in np.nonzero(unplaced_chr_bins)[0]:
            sys.stderr.write(bin_chr[u]+"\n")

            u_pred_chr=u_pred_chr_dict[bin_chr[u]]
          
            u_data=d[chr_bins[u_pred_chr],u].astype('float64')

            u_pos=chr_bin_mean_position[u_pred_chr]

            x0_array=np.mean(np.c_[u_pos[1:],u_pos[:-1]],1)
            x0_array=np.r_[-0.5e6,x0_array,u_pos[-1]+0.5e6]
            
            u_pred_pos=models[u_pred_chr].estimate_position(u_pos,u_data,x0_array)

            
            res.append(u_pred_pos)

        res=np.array(res)

        pdb.set_trace()
        
        np.savetxt(outfile+'_locus_pred.tab',np.c_[bin_chr[unplaced_chr_bins],bin_position[unplaced_chr_bins,:].astype(int),res],fmt='%s',delimiter='\t')



def cv_iter(i_list,v,chr_bin_mean_position,chr_data):
    
    pred_locs=[]
    scales=[]
    
    for i in i_list:

        eps=1e-8
        proximal_bins = (chr_bin_mean_position>=chr_bin_mean_position[i]-v-eps) & (chr_bin_mean_position<=chr_bin_mean_position[i]+v+eps)

        train_int_matrix=chr_data[:,~proximal_bins][~proximal_bins,:]
        train_positions=chr_bin_mean_position[~proximal_bins]

        
        model=triangulation.AugmentationLocPredModel()
        scale=model.estimate_scale(train_positions,train_int_matrix)

        test_vector=chr_data[i,~proximal_bins]
        test_positions=chr_bin_mean_position[~proximal_bins]

        x0_iter=np.linspace(0,np.max(test_positions),20)
        
        loc=model.estimate_position(test_positions,test_vector,x0_iter)
        
        scales.append(model.scale)
        pred_locs.append(loc)

    return pred_locs,scales
        


        
        
if __name__=="__main__":
    main()
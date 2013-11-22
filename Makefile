
# running examples:

augmentation_chr_pred:
	python augmentation_chromosome_pred.py -in Dixon12_hESC-R2A_hg19all_huref.tab -out aug_chr_pred_results -cv -v 0 1e6

augmentation_locus_pred:
	python augmentation_locus_pred.py -in Dixon12_hESC-R2A_hg19all_huref.tab -out aug_loc_pred_results -p -cf aug_chr_pred_results_predictions.tab -cv -v 0 1e6 -pnum 15

karyotype:
	python karyotype.py -in Dixon12_hESC-R2A_hg19all_huref.tab  -out karyotype_results -drop 10 -nchr 0 -s 0 -f 0.8 -n 20 -e

chromosome_scaffold:
	python chromosome_scaffold.py -in Dixon12_hESC-AllpathsLGcontigs.tab -out chromosome_scaffold_results -p 15 -it 50 -realpos contig_positions.tab 

# Allpaths-LG/GAGE contigs were obtained from http://gage.cbcb.umd.edu/data/Hg_chr14/Assembly.tgz
# We map the file Allpaths-LG/genome.ctg.fasta to hg19 chromosome 14 from http://gage.cbcb.umd.edu/data/Hg_chr14/Data.original/genome.fasta
# Mapping is performed using MUMmer following the GAGE pipeline (detailed in http://gage.cbcb.umd.edu/results/index.html), resulting in the file out.1coords

contig_positions:
        cat out.1coords | \
        awk '{if($$7>=95 && $$11>=95){print $$13"\t"$$1"\t"$$2}}' | \
        sed 's/^/chr_/' \
        > contig_positions.tab ; \

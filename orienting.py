# Main script for orienting
# Modules in orienting_mods.py which could be moved to triangulation.py later

# Example:
# python orienting.py -in Dixon12_hESC-AllpathsLGcontigs.tab -out results -pos contigs_pos.tab -real_ori contig_orientations.tab
########################################################################################################################

import orienting_mods
import triangulation
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt

def main():
	
	parser=argparse.ArgumentParser(description='Orient contigs within chromosome given interaction matrix.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-in',help='interaction frequency matrix file',dest='in_file',type=str,required=True)
	parser.add_argument('-out',help='out file prefix',dest='out_file',type=str,required=True)
	parser.add_argument('-pos',help='file with contig positions. "contig\tstart\tend"',dest='pos_file',type=str,required=True)
	parser.add_argument('-real_ori',help='file with real orientations. "contig\tsign"', dest='real_ori_file', type=str, default=None)
	
	args=parser.parse_args()
	in_file = args.in_file
	out_file = args.out_file
	pos_file = args.pos_file
	real_ori_file = args.real_ori_file

	# Read contig interacion file
	d,bin_chr,bin_position=triangulation.load_data_txt(in_file, remove_nans=True)
	
	# Read contig pos file into dictionary
	ID_col = 0
	start_col = 1
	end_col = 2
	IDs = []
	starts = []
	ends = []
	pos_fh = open(pos_file, 'r')
	for line in pos_fh:
		contig_line = line.split()
		IDs.append(contig_line[ID_col])
		starts.append(float(contig_line[start_col]))
		ends.append(float(contig_line[end_col]))
	pos_fh.close()

	# Create position dictionary for downstream analysis
	pos_dic = orienting_mods.make_pos_dic(IDs, starts, ends)

	# Sort contigs by their positions
	sorted_contigs_extra = orienting_mods.sort_by_pos(IDs, starts)
	
	# Use only contigs that are in interaction matrix
	sorted_contigs = []
	for contig in sorted_contigs_extra:
		if contig in bin_chr:
			sorted_contigs.append(contig)
	
	# Calculate bin centers
	bin_center = np.mean(bin_position, axis = 1)

	# Calculate the 4 orientation scores (edge wights) between each pair of contigs
	# Return the weighted directed acyclic graph object
	WDAG = orienting_mods.make_WDAG(d, bin_chr, bin_position, bin_center, sorted_contigs)

	# Create sorted node list for input into shortest_path function
	node_list = orienting_mods.sorted_nodes(sorted_contigs)
	
	# Find shortest path through WDAG
	orientation_results = orienting_mods.shortest_path(WDAG, node_list)

	# Create output file for predicted orientations
	OUT = open(out_file + '_pred_ori.txt', 'w+')
	# Remove start and end node from orientation result list
	orientation_results.remove("start")
	orientation_results.remove("end")

	# Format output results (Note contigs with single-bins default to forward)
	for contig in orientation_results:
		contig_ID = contig[:-3]
		orientation = contig[-2:]	
		if orientation == "fw":
			orientation = "+"
		elif orientation == "rc":
			orientation = "-" 
		else:
			print "Error in formatting output!"
		OUT.write(contig_ID + "\t" + orientation + "\n")
	OUT.close()

	if real_ori_file != None:
		# Open true orientation data to test results against
		true_fh = open(real_ori_file, 'r')
		ID_col = 0
		orient_col = 1
		true_dic = {}
		for line in true_fh:
			contig_line = line.split()
			contig_ID = contig_line[ID_col]
			orientation = contig_line[orient_col]
			true_dic[contig_ID] = orientation
		true_fh.close()
		# Record accuracy of prediction at different confidence thesholds
		# Get max confidence
		max_conf = orienting_mods.get_max_conf(WDAG, sorted_contigs)
		thresholds = np.arange(0.0, max_conf, max_conf/200.0)
		accuracy_list = []
		# Record percent of contigs removed
		percent_removed = []
		for threshold in thresholds:
			poor_conf = orienting_mods.poor_confidence(WDAG, sorted_contigs, threshold)
			percent_removed.append(float(len(poor_conf))/float(len(sorted_contigs)))
			# Calculate sensitivity, specificity, and accuracy such that fw is (+) and rc is (-)
			# Accuracy will be percent of orientations correctly predicted out of total contig orientations
			# Create prediction dictionary for orientation results
			pred_dic = orienting_mods.make_pred_dic(orientation_results, poor_conf)
		
			# Need to remove all contigs from true dictionary that are not in our prediction dictionary
			adj_true_dic = orienting_mods.adjust_true_dic(true_dic, pred_dic)

			# Calculate stats
			P, N, TP, TN, accuracy = orienting_mods.calc_stats(adj_true_dic, pred_dic)		
			accuracy_list.append(accuracy)
		# Plot results
		y_bottom = min(accuracy_list + percent_removed)
		fig, ax1 = plt.subplots()
		ax1.plot(thresholds, accuracy_list)
		ax1.set_xlabel("Confidence threshold")
		ax1.set_title("Accuracy vs Confidence")
		ax1.set_ylim(y_bottom-0.1, 1.0)
		ax1.set_ylabel("Accuracy", color='b')
		for t1 in ax1.get_yticklabels():
			t1.set_color('b')
		ax2 = ax1.twinx()
		ax2.plot(thresholds, percent_removed, 'r-')
		ax2.set_ylabel("Percent contigs removed", color='r')
		ax2.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
		ax2.set_ylim(y_bottom-0.1, 1.0)
		for t1 in ax2.get_yticklabels():
			t1.set_color('r')
		plt.savefig(out_file + '_acc_conf_plot.png')

		# Record accuracy of prediction at different contig size thresholds
		# Get max contig length of all contigs with positions
		max_length = orienting_mods.get_max_length(bin_chr, bin_position, sorted_contigs)
		contig_lengths = np.arange(0.0, max_length, max_length/200.0)
		accuracy_list = []
		percent_removed = []
		for contig_length in contig_lengths:
			# Get all contigs with length <= length of threshold
			small_contigs = orienting_mods.get_small_contigs(bin_chr, bin_position, sorted_contigs, contig_length)
			# Add all single bin/score zero contigs to list of contigs to be removed
			score_zeros = orienting_mods.poor_confidence(WDAG, sorted_contigs, 0.0)
			remove_contigs = list(set(small_contigs).union(set(score_zeros)))
			percent_removed.append(float(len(remove_contigs))/float(len(sorted_contigs)))
			pred_dic = orienting_mods.make_pred_dic(orientation_results, remove_contigs)
			# Need to remove all contigs from true dictionary that are not in our prediction dictionary
			adj_true_dic = orienting_mods.adjust_true_dic(true_dic, pred_dic)
			# Calculate stats
			P, N, TP, TN, accuracy = orienting_mods.calc_stats(adj_true_dic, pred_dic)		
			accuracy_list.append(accuracy)
		# Plot results
		y_bottom = min(accuracy_list + percent_removed)
		fig, ax1 = plt.subplots()
		ax1.plot(contig_lengths, accuracy_list)
		ax1.set_xlabel("Contig length threshold")
		ax1.set_title("Accuracy vs Contig Length")
		ax1.set_ylim(y_bottom-0.1, 1.0)
		ax1.set_ylabel("Accuracy", color='b')
		for t1 in ax1.get_yticklabels():
			t1.set_color('b')
		ax2 = ax1.twinx()
		ax2.plot(contig_lengths, percent_removed, 'r-')
		ax2.set_ylabel("Percent contigs removed", color='r')
		ax2.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
		ax2.set_ylim(y_bottom-0.1, 1.0)
		for t1 in ax2.get_yticklabels():
			t1.set_color('r')
		plt.savefig(out_file + '_acc_size_plot.png')

		# Record accuracy of prediction at different gap size thresholds
		# Get max gap size between all contigs and min gap size between all contigs
		max_gap, min_gap = orienting_mods.get_max_min_gap(sorted_contigs, pos_dic)
		gap_lengths = np.arange(max_gap, min_gap,  -max_gap/200.0)	
		accuracy_list = []
		percent_removed = []
		for gap_length in gap_lengths:
			# Get all contigs with gap size >= gap of threshold
			big_gaps = orienting_mods.get_big_gaps(pos_dic, sorted_contigs, gap_length)
			remove_contigs = list(set(big_gaps).union(set(score_zeros)))
			percent_removed.append(float(len(remove_contigs))/float(len(sorted_contigs)))
			pred_dic = orienting_mods.make_pred_dic(orientation_results, remove_contigs)
			adj_true_dic = orienting_mods.adjust_true_dic(true_dic, pred_dic)
			# Calculate stats
			P, N, TP, TN, accuracy = orienting_mods.calc_stats(adj_true_dic, pred_dic)		
			accuracy_list.append(accuracy)
		# Plot results
		y_bottom = min(accuracy_list + percent_removed)
		fig, ax1 = plt.subplots()
		ax1.plot(gap_lengths, accuracy_list)
		ax1.set_xlabel("Gap length threshold")
		ax1.set_title("Accuracy vs Gap Length")
		ax1.set_ylim(y_bottom-0.1, 1.0)
		ax1.set_ylabel("Accuracy", color='b')
		for t1 in ax1.get_yticklabels():
			t1.set_color('b')
		ax2 = ax1.twinx()
		ax2.plot(gap_lengths, percent_removed, 'r-')
		ax2.set_ylabel("Percent contigs removed", color='r')
		ax2.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
		ax2.set_ylim(y_bottom-0.1, 1.0)
		ax2.invert_xaxis()
		for t1 in ax2.get_yticklabels():
			t1.set_color('r')
		plt.savefig(out_file + '_acc_gaps_plot.png')
	

if __name__=="__main__":
	main()


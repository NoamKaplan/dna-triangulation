# Contains classes and functions for orienting contigs

import numpy as np

# Graph class for wieghted directed acyclic graph (WDAG)
class Graph:
	def __init__(self, graph_dict = {}):
		self.graph_dict = graph_dict
		self.D = {}		# dictionary for distance estimates
		self.Pi = {}	# dictionary for predecessors
	
	def nodes(self):
		return self.graph_dict.keys()
	
	def adjacent_nodes(self, node):
		return self.graph_dict[node].keys()

	def weight(self, u, v):
		return self.graph_dict[u][v]



# WDAG algorithm to find shortest path
def shortest_path(G, nodes):
	initialize_single_source(G, nodes)
	# Move through sorted nodes updating distances to adjacent nodes
	for i in range(len(nodes)):
		if i % 200 == 0:
			print "On node", i + 1, "in shortest path algorithm..."

	for u in nodes:
		# print "Node: "
		# print u
		for v in G.adjacent_nodes(u):
			relax(G, u, v)
		# print "D: "
		# print G.D
		# print "Pi: "
		# print G.Pi
		# print "\n"
	# Get the shortest path
	path = []
	start = nodes[0]
	end = nodes[-1]
	while end != start:
		path.append(end)
		end = G.Pi[end]
	path.append(start)
	path.reverse()
	return path

# Initialize single source and distance estimates for graph object
def initialize_single_source(G, nodes):
	for v in G.nodes():
		G.D[v] = float("inf")
	# Initialize source node to zero	
	G.D[nodes[0]] = 0;	

# Relax function
def relax(G, u, v):
	if G.D[v] > G.D[u] + G.weight(u,v):
		G.D[v] = G.D[u] + G.weight(u,v)
		G.Pi[v] = u

# Sort contigs by their predicted positions
def sort_by_pos(IDs, pos):
	IDs = np.array(IDs)
	pos = np.array(pos)
	sort_indices = np.argsort(pos)
	sorted_IDs = IDs[sort_indices]
	return sorted_IDs

# Calculate the 4 orientation scores (edge wights) between each pair of contigs
# Return the weighted directed acyclic graph object
def make_WDAG(HiC, bin_IDs, bin_positions, bin_centers, sorted_contigs):
	# Initialize WDAG dic such that start and end nodes have equal weights between contig orientations
	WDAG_dic = {
		"start" : {sorted_contigs[0] + "_fw" : 1.0, sorted_contigs[0] + "_rc" : 1.0},
		sorted_contigs[-1] + "_fw" : {"end" : 1.0},
		sorted_contigs[-1] + "_rc" : {"end" : 1.0},
		"end" : {}
	}
	# Get scores for every pair
	for i in range(len(sorted_contigs) - 1):
		if i % 200 == 0:
			print "Weighting contig", i + 1, "orientations..."
		
		# Get boolean indices for current contig A and neighbor B
		# print "\nOn contig " + sorted_contigs[i] + "..."
		indices_A = bin_IDs == sorted_contigs[i]
		indices_B = bin_IDs == sorted_contigs[i+1]

		########################################################################################
		# NOTE : We will assume the start position of a contig is always 1 (given by first bin)
		# as in example Dixon12_hESC-AllpathsLGcontigs.tab. If this is not the case we could get
		# errors during flips when calculating scores. Also, we assume the length of a contig
		# is given by the end position in last bin (again in agreement with example input file).
		# If input file is in a different format adjustments will need to be made
		########################################################################################

		# Get lengths contigs A and B
		A_length = np.amax(bin_positions[indices_A])
		B_length = np.amax(bin_positions[indices_B])
		
		# Assign direction labels
		Afw = sorted_contigs[i] + "_fw"
		Arc = sorted_contigs[i] + "_rc"
		Bfw = sorted_contigs[i+1] + "_fw"
		Brc = sorted_contigs[i+1] + "_rc"
		# Initialize sub dictionaries
		fw_dic = {}
		rc_dic = {}
		
		# Contig A forward interactions
		# Calculate score for forward:forward orientation
		# print "Fw --> Fw"
		Afw_Bfw_score = score(HiC[indices_A,:][:, indices_B], bin_centers[indices_A], bin_centers[indices_B], A_length) 
		# Calculate score for forward:reverse orientation
		# print "Fw --> Rc"
		# Note for reverse complement subtract bin center from length of contig + 1. ( use + 1 since bins all start at 1)
		Afw_Brc_score = score(HiC[indices_A,:][:, indices_B], bin_centers[indices_A], (B_length + 1) - bin_centers[indices_B], A_length)
		
		# Contig A reverse interactioms
		# Calculate score for reverse:forward orientation
		# print "Rc --> Fw"
		Arc_Bfw_score = score(HiC[indices_A,:][:, indices_B], (A_length + 1) - bin_centers[indices_A], bin_centers[indices_B], A_length) 
		# Calculate score for reverse:reverse orientation
		# print "Rc --> Rc"
		Arc_Brc_score = score(HiC[indices_A,:][:, indices_B], (A_length + 1) - bin_centers[indices_A], (B_length + 1) - bin_centers[indices_B], A_length) 
		
		# Save scores in dictionary
		fw_dic[Bfw] = Afw_Bfw_score
		fw_dic[Brc] =  Afw_Brc_score
		rc_dic[Bfw] = Arc_Bfw_score
		rc_dic[Brc] = Arc_Brc_score
		
		# Save in WDAG dictionary
		WDAG_dic[Afw] = fw_dic
		WDAG_dic[Arc] = rc_dic

	# print WDAG_dic	
	
	# Label orientations as unknown in WDAG if confidence score for orientations is below threshold
	# labeled_WDAG, unknown_indices = label_unknowns(WDAG_dic, sorted_contigs)


	# Make graph object
	WDAG = Graph(WDAG_dic)
	return WDAG

# Calculate edge weight equal to sum of all distances for every interaction between contig A and B
def score(HiC_sub, A_centers, B_centers, A_length):
	# Initialize distance marix
	D = []
	# print "HiC_sub: "
	# print HiC_sub
	# Add length of contig a to each bin center of contig B
	shift_B_centers = B_centers + A_length		
	# Loop through contig A bins
	for A_center in A_centers:
		# Initialize distance row
		D_row = []
		# Loop through contig B bins		
		for B_center in shift_B_centers:
			# Compute distance between bin centers and save in distance matrix
			distance = B_center - A_center
			# Note we should never get a negative distance but it'd be smart to throw an error if so (later)
			D_row.append(distance)
		
		D.append(D_row)
	# Convert distance matrix into numpy ndarray for element-wise multiplication
	D = np.array(D)
	# print "Distance matrix: "
	# print D
	# Compute sum of all distances for every interaction between contigs (edge-weight)
	weight = np.sum(D * HiC_sub) 
	# print "Distance matrix x HiC_sub = "
	# print D * HiC_sub
	# print "Sum = "
	# print weight
	return weight

# Create sorted node list for input into shortest_path function
def sorted_nodes(sorted_contigs):
	sorted_nodes = ["start"]
	for contig in sorted_contigs:
		sorted_nodes.append(contig + "_fw")
		sorted_nodes.append(contig + "_rc")
		
	sorted_nodes.append("end")
	return sorted_nodes

# Find contigs with low confidence for predicted orientation
def poor_confidence(WDAG, sorted_contigs, threshold):
	# Note this confidence score will be different for first and last contig
	# List of contigs with confidence less than or equal to threshold
	poor_contigs = []

	for i in range(len(sorted_contigs)):
		Bfw = sorted_contigs[i] + "_fw"
		Brc = sorted_contigs[i] + "_rc"
		
		# Calculate confidence score for first contig
		if i == 0:
			Cfw = sorted_contigs[i+1] + "_fw"
			Crc = sorted_contigs[i+1] + "_rc"
			conf = sum([abs(WDAG.weight("start", Bfw) - WDAG.weight("start",Brc)),
				abs(WDAG.weight(Bfw,Cfw) - WDAG.weight(Brc,Cfw)), abs(WDAG.weight(Bfw,Crc) - WDAG.weight(Brc,Crc))])
			# print conf
			# Record contigs below confidence threshold
			if conf <= threshold:
				poor_contigs.append(sorted_contigs[i])
				
		# Calculate confidence score for last contig
		elif i == len(sorted_contigs) - 1:
			Afw = sorted_contigs[i-1] + "_fw"
			Arc = sorted_contigs[i-1] + "_rc"
			conf = sum([abs(WDAG.weight(Afw,Bfw) - WDAG.weight(Afw,Brc)), abs(WDAG.weight(Arc,Bfw) - WDAG.weight(Arc,Brc)),
				abs(WDAG.weight(Bfw,"end") - WDAG.weight(Brc,"end"))])
			# print conf
			# Update WDAG_dic if score is too low
			if conf <= threshold:
				poor_contigs.append(sorted_contigs[i])

		# Calculate confidence for all other contigs
		else:

			Afw = sorted_contigs[i-1] + "_fw"
			Arc = sorted_contigs[i-1] + "_rc"
			Cfw = sorted_contigs[i+1] + "_fw"
			Crc = sorted_contigs[i+1] + "_rc"

			conf = sum([abs(WDAG.weight(Afw,Bfw) - WDAG.weight(Afw,Brc)), abs(WDAG.weight(Arc,Bfw) - WDAG.weight(Arc,Brc)),
				abs(WDAG.weight(Bfw,Cfw) - WDAG.weight(Brc,Cfw)), abs(WDAG.weight(Bfw,Crc) - WDAG.weight(Brc,Crc))])	
			# print conf
			if conf <= threshold:
				poor_contigs.append(sorted_contigs[i])
							
	return poor_contigs



# Create prediction dictionary for orientation results given contigs to remove
def make_pred_dic(orientation_results, contigs_to_remove):
	pred_dic = {}
	for contig in orientation_results:
		contig_ID = contig[:-3]
		if contig_ID not in contigs_to_remove:
			orientation = contig[-2:]	
			if orientation == "fw":
				orientation = "+"
			elif orientation == "rc":
				orientation = "-" 
			else:
				print "Error in formatting output!"
			pred_dic[contig_ID] = orientation
	return pred_dic

# Create position dictionary
def make_pos_dic(IDs, starts, ends):
	pos_dic = {}
	for i in range(len(IDs)):
		pos_dic[IDs[i]] = [starts[i], ends[i]]
	return pos_dic

# Calculate prediction statistics
def calc_stats(true_dic, pred_dic):
	# Calculate total number of (+) forward and (-) reverse complement orientations
	P = 0.0
	N = 0.0
	for contig in true_dic:
		if true_dic[contig] == "+":
			P += 1.0
		elif true_dic[contig] == "-":
			N += 1.0
		else:
			print "Error in calculating statistics!"
	# Calculate number of true positives and true negatives
	TP = 0.0
	TN = 0.0
	for contig in pred_dic:
		if pred_dic[contig] == "+" and true_dic[contig] == "+":
			TP += 1.0
		elif pred_dic[contig] == "-" and true_dic[contig] == "-":
			TN += 1.0

	# Statistics:
	#sensitivity = TP/P
	#specificity = TN/N
	if P + N != 0.0:
		accuracy = (TP + TN) / (P + N)
	else:
		accuracy = 0.0
	return P, N, TP, TN, accuracy

# Find contigs with length <= input length_threshold
def get_small_contigs(bin_IDs, bin_positions, sorted_contigs, length_threshold):
	small_contigs = []
	for contig in sorted_contigs:
		# Get length of contig
		length = np.amax(bin_positions[bin_IDs == contig])
		if length <= length_threshold:
			small_contigs.append(contig)
	return small_contigs

def get_big_gaps(pos_dic, sorted_contigs, gap_threshold):
	big_gaps = []
	for i in range(len(sorted_contigs)-1):
		if i == 0:
			# Get gap for contig
			gap = pos_dic[sorted_contigs[i+1]][0] - pos_dic[sorted_contigs[i]][1]
			if gap < 0:
				gap = 0
			if gap >= gap_threshold:
				big_gaps.append(sorted_contigs[i])
		else:
			gap1 = pos_dic[sorted_contigs[i]][0] - pos_dic[sorted_contigs[i-1]][1]
			if gap1 < 0:
				gap1 = 0
			gap2 = pos_dic[sorted_contigs[i+1]][0] - pos_dic[sorted_contigs[i]][1]
			if gap2 < 0:
				gap2 = 0
			if gap1 >= gap_threshold and sorted_contigs[i] not in big_gaps:
				big_gaps.append(sorted_contigs[i])
			if gap2 >= gap_threshold:
				if sorted_contigs[i] not in big_gaps:
					big_gaps.append(sorted_contigs[i])
				big_gaps.append(sorted_contigs[i+1])
	return big_gaps		

def adjust_true_dic(true_dic, pred_dic):
	adj_true_dic = dict(true_dic)
	contigs = adj_true_dic.keys()
	for contig in contigs:
		if contig not in pred_dic:
			adj_true_dic.pop(contig)
	return adj_true_dic

def get_max_length(bin_IDs, bin_positions, sorted_contigs):
	lengths = []
	for contig in sorted_contigs:
		# Get length of contig
		length = np.amax(bin_positions[bin_IDs == contig])	
		lengths.append(length)
	max_length = np.amax(lengths)
	return max_length

def get_max_min_gap(sorted_contigs, pos_dic):
	gaps = []
	for i in range(len(sorted_contigs)-1):
		# Get gap between current and next contig
		gap = pos_dic[sorted_contigs[i+1]][0] - pos_dic[sorted_contigs[i]][1]
		if gap < 0:
			gap = 0
		gaps.append(gap)
	return max(gaps), min(gaps)



def get_max_conf(WDAG, sorted_contigs):
	conf_list = []
	for i in range(len(sorted_contigs)):
		Bfw = sorted_contigs[i] + "_fw"
		Brc = sorted_contigs[i] + "_rc"		
		# Calculate confidence score for first contig
		if i == 0:
			Cfw = sorted_contigs[i+1] + "_fw"
			Crc = sorted_contigs[i+1] + "_rc"
			conf = sum([abs(WDAG.weight("start", Bfw) - WDAG.weight("start",Brc)),
				abs(WDAG.weight(Bfw,Cfw) - WDAG.weight(Brc,Cfw)), abs(WDAG.weight(Bfw,Crc) - WDAG.weight(Brc,Crc))])				
		# Calculate confidence score for last contig
		elif i == len(sorted_contigs) - 1:
			Afw = sorted_contigs[i-1] + "_fw"
			Arc = sorted_contigs[i-1] + "_rc"
			conf = sum([abs(WDAG.weight(Afw,Bfw) - WDAG.weight(Afw,Brc)), abs(WDAG.weight(Arc,Bfw) - WDAG.weight(Arc,Brc)),
				abs(WDAG.weight(Bfw,"end") - WDAG.weight(Brc,"end"))])
		# Calculate confidence for all other contigs
		else:
			Afw = sorted_contigs[i-1] + "_fw"
			Arc = sorted_contigs[i-1] + "_rc"
			Cfw = sorted_contigs[i+1] + "_fw"
			Crc = sorted_contigs[i+1] + "_rc"
			conf = sum([abs(WDAG.weight(Afw,Bfw) - WDAG.weight(Afw,Brc)), abs(WDAG.weight(Arc,Bfw) - WDAG.weight(Arc,Brc)),
				abs(WDAG.weight(Bfw,Cfw) - WDAG.weight(Brc,Cfw)), abs(WDAG.weight(Bfw,Crc) - WDAG.weight(Brc,Crc))])	
		conf_list.append(conf)						
	return max(conf_list)

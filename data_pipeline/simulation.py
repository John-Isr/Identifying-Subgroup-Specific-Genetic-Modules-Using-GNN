#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Imports
import networkx as nx
import numpy as np
import pandas as pd
import community as community_louvain
import random
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor #ransac needs to import LinearRegression
from scipy.stats import rankdata, norm, pearsonr
from sklearn.metrics import confusion_matrix, classification_report


# Block 1 - generate graph and cluster it 

#generate a power-law graph and cluster it
#NOTE - limit_num_clusters only works with 'greedy'
def barbasi_louvain(num_nodes, m0_value, clustering_choice = "louvain", max_clusters = 100, limit_num_clusters = False):
    G = nx.barabasi_albert_graph(num_nodes, m0_value)
    
    if clustering_choice == "louvain":
        comms = community_louvain.best_partition(G)
    elif clustering_choice == "greedy":
        if limit_num_clusters:
            comms = nx.algorithms.community.greedy_modularity_communities(G, best_n = max_clusters) 
        else:
            comms = nx.algorithms.community.greedy_modularity_communities(G) 
    else:
        raise ValueError("Invalid clustering choice. Choose one of {'louvain', 'greedy'}")
        #scale the distances
    
    return G, comms


# Block 1.5 - Create auxiliary dicts and choose "evil" cluster

#generates auxiliary dictionaries used in future functions
#comms_inverted - is an inverted dict, the keys are the cluster ids and the values are a set of all nodes in the cluster
#edges_in_cluster - a dictionary of cluster keys where the values are edges entirely within the given cluster

def aux_dicts(G, comms, clustering_choice):
    
    #num of clusters:
    if clustering_choice == 'louvain':
        num_of_clusters = np.unique(list(comms.values())).max() + 1
    elif clustering_choice == 'greedy':
        num_of_clusters = len(comms)

    #choose at ranodm one of the clusters as the "evil" cluster
    latent_process_cluster = np.random.randint(0, num_of_clusters)

    #comms_inverted is an inverted dict - the keys are the cluster ids and the values are a set of all nodes in the cluster
    comms_inverted = defaultdict(set)
    if clustering_choice == 'louvain':
        for node, cluster in sorted(comms.items(), key = lambda item: item[1]):
            comms_inverted[cluster].add(node)
    elif clustering_choice == 'greedy':
        for cluster in range(len(comms)):
            for node in comms[cluster]:
                comms_inverted[cluster].add(node)
        
    #makes a dictionary of cluster keys where the values are edges entirely within the given cluster
    #NOTE - the dict will not contain all edges - edges between clusters will not be included
    edges_in_cluster = defaultdict(list)
    for cluster in comms_inverted.keys():
        for node1 in comms_inverted[cluster]:
            for node2 in comms_inverted[cluster]:
                if G.has_edge(node1, node2):
                    if (node2, node1) not in edges_in_cluster[cluster]: #avoids adding edges twice
                        edges_in_cluster[cluster].append((node1, node2))
    
    return latent_process_cluster, num_of_clusters, comms_inverted, edges_in_cluster


# plotting G grpah stat functions:

# plots the node count per cluster
def plot_node_count_per_cluster(num_of_clusters, comms_inverted):
    print("total number of clusters: ", num_of_clusters)
    
    hist_count = np.zeros(num_of_clusters, int)
    for cluster_id in comms_inverted.keys():
        hist_count[cluster_id] = len(comms_inverted[cluster_id])

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x = np.arange(num_of_clusters), y = hist_count, color=sns.color_palette()[0])
    ax.bar_label(ax.containers[0], fontsize=10);
    ax.set_title("Num of Nodes per Cluster");
    ax.set_xlabel("cluster ID");
    ax.set_ylabel("node count");
    plt.show()

# Plotting the Edge distribution - the number of edges adjacent to the node
def plot_degree_distribution(G, nodes):
    deg = list(map(lambda x: x[1], G.degree))

    plt.figure(figsize=(6, 4))
    ax = sns.histplot(deg, bins = int(np.sqrt(nodes)), color = sns.color_palette()[0])
    ax.set_title("Degree Distribution");
    ax.set_xlabel("bins");
    ax.set_ylabel("edges per bin");
    plt.show()

    
# Block 2 - Assign edges as "evil" or not

#returns 1 or 0 based on given probabiliy p 
def mapper(p):
    x = np.random.uniform(0, 1)
    if x < p:
        result = 1
    else:
        result = 0
    return result

# mapping every edge to be part of the latent process or not
# if the edge is contained entirely inside the "evil" cluster, then it will be assigned as "evil" with a higher probability
# otherwise it might still be assigned as evil, but with a smaller probability 
def assign_edges_to_latent_process(G, edges_in_cluster, latent_process_cluster, 
                     p_cluster_NOT_part_of_process = 0.2, p_cluster_part_of_process = 0.9):
    m_hat = 0
    nx.set_edge_attributes(G, m_hat, "m_hat")
    for edge in G.edges():
        (a,b) = edge
        inverted_edge = (b,a)
        if (edge in edges_in_cluster[latent_process_cluster]) or (inverted_edge in edges_in_cluster[latent_process_cluster]):
            G.edges[edge]['m_hat'] = mapper(p_cluster_part_of_process)
        else:
            G.edges[edge]['m_hat'] = mapper(p_cluster_NOT_part_of_process)

    m_hat_edges = nx.get_edge_attributes(G, "m_hat")
    return m_hat_edges


# Block 3 - for each edge decide which patients contribute to it

# decides for each edge which patients contribute to it based on if the edge is "evil" and if the patient is sick or not
# m_hat_edges_values - a binary list the size of Edges saying which edges are "evil"
# edge_list_dict is just a dict where the keys are indexes 0-995, and the values are edges (70, 108), (2, 7) etc.
# m_patient_is_sick is a binary list saying for each patient if they are sick or not. 
# m_patient_part_of_edge_corr is a matrix of size Edges X Patients (for example 996 rows and 100 columns). It is a binary
# matrix that for each edge (each row) says which patients contributed to it

def assign_patients_to_edges(G, num_of_patients, m_hat_edges, prob_is_sick, Beta=0.9, Gamma=0.5, Delta=0.5, Sigma=0.5):

    edge_list = list(G.edges())
    edge_list_dict = {i: coord for i, coord in enumerate(edge_list)}
    m_hat_edges_values = np.array(list(m_hat_edges.values()))
    num_of_edges = (np.shape(edge_list)[0])
    
    m_patient_is_sick = random.choices([0, 1], k=num_of_patients, weights=(1 - prob_is_sick, prob_is_sick))
    m_patient_part_of_edge_corr = np.zeros((num_of_edges, num_of_patients))
    
    for edge_num in edge_list_dict:
        
        #if edge is evil
        if m_hat_edges_values[edge_num]: 
            for patient in range(num_of_patients):
                if m_patient_is_sick[patient]:
                    m_patient_part_of_edge_corr[edge_num][patient] = mapper(Beta)
                else:
                    m_patient_part_of_edge_corr[edge_num][patient] = mapper(Gamma)
        
        #edge is normal
        else: 
            for patient in range(num_of_patients):
                if m_patient_is_sick[patient]:
                    m_patient_part_of_edge_corr[edge_num][patient] = mapper(Delta)
                else:
                    m_patient_part_of_edge_corr[edge_num][patient] = mapper(Sigma)


    return edge_list_dict, num_of_edges, m_patient_is_sick, m_patient_part_of_edge_corr, m_hat_edges_values


# Sanity Check - check if "evil" cluster is getting correct assignment of sick 

# Test if the avg number of sick and healhty patients per edge for each cluster matches the expected values 
# We expect the "evil" cluster to have a higher average number of sick per edge relative to the other clusters
def sanity_check(G, prob_is_sick, Sigma, Delta, Beta, p_cluster_NOT_part_of_process, p_cluster_part_of_process,
                 edges_in_cluster, m_patient_part_of_edge_corr, latent_process_cluster, num_of_clusters, num_of_patients,
                 m_patient_is_sick, m_hat_edges, print_plot = True, print_stats = False):
    
    #num of sick in graph:
    num_sick = np.sum(m_patient_is_sick)
    num_healthy = num_of_patients - num_sick

    #calculate avg number of sick and heathy per edge for each cluster:
    edge_index_dict = {edge: index for index, edge in enumerate(list(G.edges()))}
    avg_sick_per_cluster = []
    avg_healthy_per_cluster = []
    evil_edge_ratio = []
    
    for cluster, edge_list in sorted(edges_in_cluster.items(), key = lambda item: item[0]):
        
        num_edges_in_cluster = len(edge_list)
        total_sick_in_cluster = 0
        total_healthy_in_cluster = 0
        num_evil_edges_in_cluster = 0


        for edge in edge_list:
            
            #reverse edge tuple to correct orientation
            if edge in edge_index_dict.keys():
                edge_index = edge_index_dict[edge]
            else:
                edge_index = edge_index_dict[edge[::-1]] #reverse tuple
            
            if edge not in m_hat_edges.keys():
                (a, b) = edge
                edge = (b, a)
            
            #update count of evil edges in cluster
            is_edge_evil = m_hat_edges[edge]
            num_evil_edges_in_cluster += is_edge_evil
            
            #update count of sick and healthy in cluster
            patients_of_edge = m_patient_part_of_edge_corr[edge_index]
            total_patients_of_edge = np.sum(patients_of_edge)
            num_sick_curr_edge = np.sum(patients_of_edge * m_patient_is_sick)
            num_healthy_curr_edge = total_patients_of_edge - num_sick_curr_edge
            total_sick_in_cluster += num_sick_curr_edge
            total_healthy_in_cluster += num_healthy_curr_edge
        
        #record ratio of evil edges for each cluster
        ratio_evil_edges_in_cluster = num_evil_edges_in_cluster / num_edges_in_cluster
        evil_edge_ratio.append(ratio_evil_edges_in_cluster)
        
        #record avg sick and healthy for each cluster
        avg_sick_in_cluster = total_sick_in_cluster / num_edges_in_cluster
        avg_healthy_in_cluster = total_healthy_in_cluster / num_edges_in_cluster
        avg_sick_per_cluster.append(avg_sick_in_cluster)
        avg_healthy_per_cluster.append(avg_healthy_in_cluster)

    
    #calculate the expected avg num of healthy and sick for normal vs "evil" clusters:
    #explanation of the calculation is provided further below 
    expected_avg_healthy_per_cluster = (num_of_patients * (1-prob_is_sick)) * Sigma
    expected_avg_sick_per_normal_cluster = (num_of_patients * prob_is_sick) * (Delta * (1 - p_cluster_NOT_part_of_process) + Beta * p_cluster_NOT_part_of_process)  
    expected_avg_sick_per_evil_cluster = (num_of_patients * prob_is_sick) * (Delta * (1- p_cluster_part_of_process) + Beta * p_cluster_part_of_process)  
    
    
    #calc error from expected:
    random_good_cluster = latent_process_cluster - 1 if latent_process_cluster > 0 else latent_process_cluster + 1
    
    ratio_evil_edges_normal_cluster = evil_edge_ratio[random_good_cluster]
    ratio_evil_edges_evil_cluster = evil_edge_ratio[latent_process_cluster]
    
    diff_avg_sick_evil_cluster = np.abs(expected_avg_sick_per_evil_cluster - avg_sick_per_cluster[latent_process_cluster])
    error_avg_sick_evil_cluster = diff_avg_sick_evil_cluster / expected_avg_sick_per_evil_cluster
    
    diff_avg_sick_normal_cluster = np.abs(expected_avg_sick_per_normal_cluster - avg_sick_per_cluster[random_good_cluster])
    error_avg_sick_normal_cluster = diff_avg_sick_normal_cluster / expected_avg_sick_per_normal_cluster
    
    diff_avg_healthy = np.abs(expected_avg_healthy_per_cluster - avg_healthy_per_cluster[random_good_cluster])
    error_avg_healthy = diff_avg_healthy / expected_avg_healthy_per_cluster
    
    
    if print_plot:
        #plot the actual avg sick / healthy per edge of each cluster (only taking into acount edges entirely within the cluster):
        print("Latent_Process_Cluster: ", latent_process_cluster)

        clusters_indexes = np.arange(num_of_clusters)
        sanity_check_data = pd.DataFrame({
            'Cluster': np.concatenate([clusters_indexes, clusters_indexes]),
            'Category': np.repeat(['Sick', 'Healthy'], num_of_clusters),
            'Counts': np.concatenate([avg_sick_per_cluster, avg_healthy_per_cluster])
        })

        plt.figure(figsize=(6, 4))
        ax = sns.barplot(data=sanity_check_data, x="Cluster", y="Counts", hue="Category")
        ax.set_title("Average Sick/Healthy per edge for each cluster");
        ax.set_xlabel("cluster ID");
        ax.set_ylabel("Average Patient");
        ax.plot(latent_process_cluster, avg_sick_per_cluster[latent_process_cluster] + 1, "*", markersize=14, color="r");
        plt.show()
        

    if print_stats:
        print(f"we expect avg healthy per cluster of: {expected_avg_healthy_per_cluster:.2f}")
        print(f"we expect avg sick per normal cluster of: {expected_avg_sick_per_normal_cluster:.2f}")
        print(f"we expect avg sick per 'evil' cluster of: {expected_avg_sick_per_evil_cluster:.2f}")
        print(f"we expect ratio of sick edges per normal cluster: {p_cluster_NOT_part_of_process:.2f}")
        print(f"we expect ratio of sick edges per sick cluster: {p_cluster_part_of_process:.2f}")
        
        print(f"\nerror % avg healthy per cluster: {error_avg_healthy:.2f}")
        print(f"error % avg sick per normal cluster: {error_avg_sick_normal_cluster:.2f}")
        print(f"error % avg sick per 'evil' cluster: {error_avg_sick_evil_cluster:.2f}")
        print(f"ratio of sick edges per normal cluster: {ratio_evil_edges_normal_cluster:.2f}")
        print(f"ratio of sick edges per evil cluster: {ratio_evil_edges_evil_cluster:.2f}")
    
    #explanation of the expected num of sick or healthy for a normal vs "evil" cluster:
    #If all edges have a 50% chance (Sigma or Gamma or Delta) of getting a healhty patient, then on average they should have:
    #Sigma * num_of_patients * (percent_of_healhty = 1 - prob_is_sick)

    #If a cluster is good, it has 20% (p_cluster_NOT_part_of_process) of it's edges as "evil" 
    #and 80% (1 - p_cluster_NOT_part_of_process) of it's edges as normal.
    #For each edge type the percent of evil is different: 50% (Delta) for the normal edges, 
    #or 90% (Beta) for "evil" edges. 
    #This means the expected sick for a good cluster is: 
    #[num_of_sick = num_of_patients * prob_is_sick] * [0.5 (Delta) * 0.8 (1 - p_cluster_NOT_part_of_process) + 0.9 (Beta) * 0.2 (p_cluster_NOT_part_of_process)]

    #As for an "evil" cluster, it has 90% (p_cluster_part_of_process) of its edges as evil, 
    #and only #10% (1- p_cluster_part_of_process) of its edges as good. 
    #This means that the expected sick for an "evil" cluster is:
    #[num_of_sick = num_of_patients * prob_is_sick] * [0.5 (Delta) * 0.1 (1- p_cluster_part_of_process) + 0.9 (Beta) * (0.9) (p_cluster_part_of_process)]

    return error_avg_sick_evil_cluster, error_avg_sick_normal_cluster, error_avg_healthy, ratio_evil_edges_normal_cluster, ratio_evil_edges_evil_cluster   
    

#runs all previous blocks multiple times to give the average result over many attempts 
def sanity_check_with_repeats(nodes, m0_value, clustering_choice, p_cluster_part_of_process, p_cluster_NOT_part_of_process, 
                              num_of_patients, prob_is_sick, Beta, Gamma, Delta, Sigma, epochs = 10, print_stats = False, 
                              max_clusters = 100, limit_num_clusters = False):
    
    
    total_error_sick_evil = 0
    total_error_sick_normal = 0
    total_error_healthy = 0
    total_ratio_normal = 0
    total_ratio_evil = 0
    
    for i in range(epochs):
        
        #generate graph and cluster into communities
        [G, comms] = barbasi_louvain(nodes, m0_value, clustering_choice, max_clusters, limit_num_clusters)

        #run aux_dicts and choose "evil" cluster
        latent_process_cluster, num_of_clusters, comms_inverted, edges_in_cluster = aux_dicts(G, comms, clustering_choice)

        #Run m_hat_edges
        m_hat_edges = assign_edges_to_latent_process(G, edges_in_cluster, latent_process_cluster, 
                                                     p_cluster_NOT_part_of_process, p_cluster_part_of_process)

        #for each edge decide which patients contribute to it
        (edge_list_dict, num_of_edges ,m_patient_is_sick, 
         m_patient_part_of_edge_corr, m_hat_edges_values) = assign_patients_to_edges(G, num_of_patients, m_hat_edges,
                                                                        prob_is_sick, Beta, Gamma, Delta, Sigma)
        
        #run the sanity check
        (error_sick_evil, error_sick_normal, error_healthy, ratio_normal,
         ratio_evil) = sanity_check(G, prob_is_sick, Sigma, Delta, Beta, p_cluster_NOT_part_of_process, 
                                          p_cluster_part_of_process, edges_in_cluster, m_patient_part_of_edge_corr, 
                                          latent_process_cluster, num_of_clusters, num_of_patients, 
                                          m_patient_is_sick, m_hat_edges, print_plot = False, print_stats = print_stats)
        
        total_error_sick_evil += error_sick_evil
        total_error_sick_normal += error_sick_normal
        total_error_healthy += error_healthy
        total_ratio_normal += ratio_normal
        total_ratio_evil += ratio_evil

    avg_error_sick_evil = total_error_sick_evil / epochs
    avg_error_sick_normal = total_error_sick_normal / epochs
    avg_error_healthy = total_error_healthy / epochs
    avg_ratio_normal = total_ratio_normal / epochs
    avg_ratio_evil = total_ratio_evil / epochs
    diff_ratio_normal = np.abs(avg_ratio_normal - p_cluster_NOT_part_of_process)
    diff_ratio_evil = np.abs(avg_ratio_evil - p_cluster_part_of_process)
    
    if print_stats:
        print(f"avg taken over {epochs} epochs")
        print(f"avg error of sick per evil cluster: {avg_error_sick_evil:.2f}")
        print(f"avg error of sick per normal cluster: {avg_error_sick_normal:.2f}")
        print(f"avg error of healthy per cluster: {avg_error_healthy:.2f}")

        print(f"\nwe expect ratio of sick edges per normal cluster: {p_cluster_NOT_part_of_process:.2f}")
        print(f"we expect ratio of sick edges per sick cluster: {p_cluster_part_of_process:.2f}")
        print(f"avg ratio of sick edges per normal cluster: {avg_ratio_normal:.2f}")
        print(f"avg ratio of sick edges per evil cluster: {avg_ratio_evil:.2f}")
        print(f"DIFF ratio of sick edges per normal cluster: {diff_ratio_normal:.2f}")
        print(f"DIFF ratio of sick edges per evil cluster: {diff_ratio_evil:.2f}")

    
# Block 4 - generate feature vectors for each edge

# generate the gene values (how much a patient has of a given gene) for healthy patients
# since the healthy ones are always sampled from a 2D gaussian with cov=0, we can instead
# sample each gene once for each patient from a 1D gaussian and save time wasted on repeated samplings. 
 
# NOTE - these values can be negative (they are not real). 
# They are only used for the correlation measurement when generating the feature vector of each edge. 
# Even if we raise the mean to make the values almost always positive, when measuring the correlation 
# we subtract the mean anyway

def generate_healthy_patients_gene_levels(num_nodes, num_patients, m_patient_is_sick, gene_vars, scale_healthy_var=1):
    m_patient_gene_levels = np.zeros(shape = (num_nodes, num_patients))

    for patient in range(num_patients):
        if m_patient_is_sick[patient]: #if patient is sick - skip them
            continue

        for gene in range(num_nodes):   
            gene_var = gene_vars[gene] * scale_healthy_var
            gene_std = np.sqrt(gene_var)
            gene_value = np.random.normal(scale = gene_std)
            m_patient_gene_levels[gene][patient] = gene_value

    return m_patient_gene_levels


#plots the data points and the line of best fit 
def scatterplot_with_regression(x, y, y_pred, distances, x_points_sick,
                                color_by = 'distance', title='', plot_outer_lines=False, spearman=False):

    #generate list for markers
    marker_names = []
    for i in range(len(x_points_sick)):
        marker = 'sick' if (x_points_sick[i] == 1) else 'healthy'
        marker_names.append(marker)
    
    #set color by distance or non at all
    if color_by == 'no color':
        length = len(distances)
        distances = np.zeros(length)
    
    if color_by not in ['distance', 'no color']:
        raise ValueError("Invalid color choice. Choose one of {'distance', 'no color'}")
    
    #plot scatter plot and regression line
    plt.figure(figsize=(6, 4))
    ax = sns.scatterplot(x=x, y=y, hue=distances, style=marker_names)
    ax.set_title(title);
    ax.set_xlabel("Gene A");
    ax.set_ylabel("Gene B");
    ax = sns.lineplot(x=x, y=y_pred, color='red')
        
    #plot outter lines
    if plot_outer_lines:
        abs_distances = np.abs(y - y_pred) 
        max_dist = abs_distances.max() 
        ax = sns.lineplot(x=x, y=y_pred + max_dist, color='orange')
        ax = sns.lineplot(x=x, y=y_pred - max_dist, color='orange')
        
        if spearman:
            data_size = len(x)
            lower_y_lim = 0 - 0.25 * data_size
            upper_y_lim = data_size * 1.25
            plt.ylim(lower_y_lim, upper_y_lim)
    
    
    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # Adjust the plot to make space for the legend
    plt.show()


#given an edge, it generates the data points for each patient that contributes to the edge.
#If the patient is healthy the values are taken from the pre-sampled gaussians of the healthy patients.
#if the patient is sick we sample from a multivariate gaussian with a correlation value that determines in turn the covariance.
#the correlation value is sampled from a uniform distribution (the range of which is a hyper parameter of the simulation)
def generate_data_points(edge, num_patients, m_patient_part_of_edge_corr,
                         m_patient_is_sick, edge_list, m_patient_gene_levels,
                         gene_vars, lower_correlation_bound, upper_correlation_bound):

    #grap variance values of the genes at the ends of current edge
    gene1, gene2 = edge_list[edge]
    var1 = gene_vars[gene1]
    var2 = gene_vars[gene2]

    mean_vec = np.array([0,0])
    x_values = np.zeros(num_patients)
    y_values = np.zeros(num_patients)
    
    #sampline correlation for sick patients from given range
    correlation = np.random.uniform(low = lower_correlation_bound, high = upper_correlation_bound) 
    cov = correlation * np.sqrt(var1) * np.sqrt(var2)        
    cov_mat = np.array([[var1, cov],
                        [cov, var2]])
    
    for patient in range(num_patients):

        #skip patients that don't contribute to the given edge
        if m_patient_part_of_edge_corr[edge][patient] == 0: 
            continue 

        curr_patient_is_sick = m_patient_is_sick[patient]

        #patient is healthy:
        if curr_patient_is_sick == 0: 
            x = m_patient_gene_levels[gene1][patient]
            y = m_patient_gene_levels[gene2][patient]
            x_values[patient] = x
            y_values[patient] = y

        #patient is sick
        else:
            #sample from bi-variate gaussian with correlation
            x, y = np.random.multivariate_normal(mean_vec, cov_mat)
            x_values[patient] = x
            y_values[patient] = y

    return x_values, y_values

#measures distance of points from regression line 
def dist_from_regression_line(x, y, num_patients, is_patient_part_of_edge, m_patient_is_sick, plot=False,  
                              scale_dists = False, model_choice = 'ransac', dist_measure = 'alpha score', 
                              scaler_choice = 'min_max', color_by = 'distance', spearman = True):
    """
    model_choice (str):  Choose one of {'ransac', 'huber', 'linear'}
    dist_measure (str):  Choose one of {'normal', 'inverse', 'inverse sqrt', outer lines, 'outliers', 'alpha score'}
    scaler_choice (str): Choose one of {'min_max', 'standard', 'robust'}

    """
    
    #indexes of patients that contributed to the current edge
    mask = np.where(is_patient_part_of_edge)[0]
    inverse_mask = np.where(is_patient_part_of_edge - 1)[0]
    
    #only use points of patients that contribute to the edge
    x_points = x[mask]
    y_points = y[mask]
    m_sick_and_contribute = m_patient_is_sick * is_patient_part_of_edge 
    x_points_sick = m_sick_and_contribute[mask]
    
    #if using spearmans correlation, convert data to ranked 
    if spearman:
        x_points = rankdata(x_points)
        y_points = rankdata(y_points)
    
    #initialize final feature vector
    feature_vector = np.zeros(num_patients)

    #fit regression and measure distances of points to the line
    x_points_for_model = x_points.reshape(-1,1) #the regression model needs x as a matrix of dx1 and not a 1D array
        
    #choose model
    if model_choice == 'ransac':
        model = RANSACRegressor(LinearRegression())
    elif model_choice == 'huber':
        model = HuberRegressor()
    elif model_choice == 'linear':
        model = LinearRegression()
    else:
        raise ValueError("Invalid model type. Choose one of {'ransac', 'huber', 'linear'}")

    #fit model, predict values and measure distances of points to the line 
    model.fit(x_points_for_model, y_points)
    y_pred = model.predict(x_points_for_model)
    residuals = y_points - y_pred
    abs_distances = np.abs(y_points - y_pred) 
    max_dist = abs_distances.max()
    
    #measure outliers
    if model_choice == 'ransac':
        inliers_mask = (model.inlier_mask_).astype(int)
    elif model_choice == 'huber':
        inliers_mask = (~model.outliers_).astype(int)
    elif model_choice == 'linear':
        inliers_mask = np.zeros(len(y_points), dtype=bool)
        
    eps = 1e-8
    #measure distances of points to the regression line - these are the values in the feature vector    
    if dist_measure == 'normal': #absolute errors, i.e. the residules
        distances = abs_distances
    elif dist_measure == 'outliers':
        distances = inliers_mask #inliers get 1, outliers get 0
    elif dist_measure == 'inverse':
        distances = 1 / abs_distances + eps
    elif dist_measure == 'inverse sqrt':
        distances = np.sqrt(1 / abs_distances + eps)
    elif dist_measure == 'outer lines':
        upper_line = y_pred + max_dist
        lower_line = y_pred - max_dist
        distances_to_upper_line = np.abs(y_points - upper_line)
        distances_to_lower_line = np.abs(y_points - lower_line)
        distances = np.minimum(distances_to_lower_line, distances_to_upper_line)
    elif dist_measure == 'alpha score':
        #scale residuals to normal distribution
        residual_scaler = StandardScaler()
        std_residuals = residuals.reshape(-1, 1) #needed for the scaler
        std_residuals = residual_scaler.fit_transform(std_residuals)
        std_residuals = std_residuals.flatten() #revets back to original shape
        # Calculate the CDF value for each standardized residual
        cdf_values = norm.cdf(std_residuals)
        # Calculate the two-tailed probability for each standardized residual
        two_tailed_probabilities = 2 * np.minimum(cdf_values, 1 - cdf_values)
        distances = two_tailed_probabilities
    else:
        raise ValueError("Invalid distance measure choice. Choose one of {'normal', 'inverse', 'inverse sqrt', 'outer lines', 'outliers', 'alpha score'}")
    
    #scale distances if option chosen by user:
    if scale_dists:         
        if scaler_choice == 'min_max':
            scaler = MinMaxScaler()
        elif scaler_choice == 'standard':
            scaler = StandardScaler()
        elif scaler_choice == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaler choice. Choose one of {'min_max', 'standard', 'robust'}")
    
    #scale distances via min_max for spearman
    if spearman:
        scaler = MinMaxScaler()
    
    #scale the distances
    if (dist_measure != 'alpha score') and (spearman or scale_dists):
        distances = distances.reshape(-1, 1) #needed for the scaler
        distances = scaler.fit_transform(distances)
        distances = distances.flatten() #revets back to original shape
        
    #store values in feature vector (array size of "num_of_patients")    
    #update values of contributing patients in feature vector
    i = 0
    for index in mask:
        feature_vector[index] = distances[i]
        i+=1
    
    #update values of non-contributing patients in feature vector
    j = 0
    non_contribute_value = 0 #good for distance scores where a high score means large contribution 
    for index in inverse_mask:
        
        #In this case, large distances mean a bad score:
        if dist_measure == 'normal': 
            non_contribute_value = distances.max() * 10
            #non_contribute_value = np.inf
        
        feature_vector[index] = non_contribute_value
        j += 1
    
    #plot the data points and the regresion line 
    if plot: 
        
        #print confusion matrix
        #x_points_sick - the true label
        #inliers_mask - the prediciton 
        if dist_measure == 'outliers':
            confusion_mat = confusion_matrix(x_points_sick, inliers_mask)
            report = classification_report(x_points_sick, inliers_mask)
            print("confusion matrix for single edge via outliers:")
            print(confusion_mat)
            print(report)

        #plot scatter plot with regression line
        plot_outer_lines = True if (dist_measure == 'outer lines') else False    
        plot_title = f'{model_choice} regression line colored by dist {dist_measure}'
        scatterplot_with_regression(x_points, y_points, y_pred, distances, 
                                    x_points_sick, color_by, plot_title, plot_outer_lines, spearman)
    
    return feature_vector

#returns the index of the first "evil" edge found
def index_single_evil_edge(m_hat_edges_values):
    
    for i in range(len(m_hat_edges_values)):
        if m_hat_edges_values[i] == 1:
            return i

    return len(m_hat_edges_values) - 1

# for each edge generate (or grab the existing) gene values for each patient contributing to the edge
# then use these points for a correlation scatter plot and fit a regression line to it
# then save the distances of the points from the line as the feature vector of the given edge

def generate_feature_vectors(G, m_patient_part_of_edge_corr, m_patient_is_sick, m_patient_gene_levels,
                             gene_vars, lower_correlation_bound, upper_correlation_bound, evil_edge_index, plot=False, 
                             scale_dists=False, model_choice='ransac', dist_measure='normal', scaler_choice='min_max', 
                             color_by='distance', spearman=True):

    edge_list = list(G.edges())
    num_edges = m_patient_part_of_edge_corr.shape[0]
    num_patients = m_patient_part_of_edge_corr.shape[1]

    for edge in range(num_edges):
        to_plot = False
        #if plot chosen, will only plot the first "evil" edge instead of plotting every edge.
        if plot and edge == evil_edge_index:
            to_plot = True
                  
        #generate data points based on sick or healthy gaussians
        x, y = generate_data_points(edge, num_patients, m_patient_part_of_edge_corr,
                                          m_patient_is_sick, edge_list, m_patient_gene_levels,
                                          gene_vars, lower_correlation_bound, upper_correlation_bound)
        
        
        #fit regression line to the data and returned measured distances of points to the line             
        is_patient_part_of_edge = m_patient_part_of_edge_corr[edge]
        feature_vector = dist_from_regression_line(x, y, num_patients, is_patient_part_of_edge, m_patient_is_sick, to_plot, 
                                                   scale_dists, model_choice, dist_measure, scaler_choice, color_by, spearman)
        
        #add feature vector to the edge in the graph
        G.edges[edge_list[edge]]['feature_vector'] = feature_vector
        
        #NEW FOR DEEP PROJECT
        edge_empirical_corr, _ = pearsonr(x,y)
        G.edges[edge_list[edge]]['correlation'] = edge_empirical_corr
        

# adds feature vector to each edge that is simply a binary vector of which patients contributed to it
# there's no gaussian sampling 
def add_simple_feature_vectors(G, m_patient_part_of_edge_corr):

    edge_list = list(G.edges())
    for edge_num in range(len(edge_list)):
        contributing_patients = m_patient_part_of_edge_corr[edge_num]
        G.edges[edge_list[edge_num]]['feature_vector'] = contributing_patients

        
        
#final - runs entire simulation 
# returns:
# G - the graph with the feature vectors - G.edges[edge]['feature_vector']
# m_patient_is_sick -  binary 1D array for which patient is sick or not
# latent_process_cluster - id of the "evil" cluster
# comms - dict where keys are nodes and values are cluster ids 
# comms_inverted - dict where keys are cluster ids and values are nodes in the cluster

def run_simulation(nodes = 500,
                   percent_of_nodes = 0.2,
                   prob_is_sick = 0.3,
                   m0_value = 2,
                   clustering_choice = "louvain",
                   max_clusters = 100,
                   limit_num_clusters = False,
                   num_clusters_in_latent_processes = 1,
                   p_cluster_NOT_part_of_process = 0.2,
                   p_cluster_part_of_process = 0.9,
                   Beta = 0.9,
                   Gamma = 0.5,
                   Delta = 0.5,
                   Sigma = 0.5,
                   lower_correlation_bound = 0.7,
                   upper_correlation_bound = 0.9,
                   lower_var_bound = 0.8,
                   upper_var_bound = 1,
                   scale_healthy_var = 1,
                   model_choice = 'ransac',
                   spearman = True,
                   dist_measure = 'alpha score',
                   scale_dists = False,
                   scaler_choice = 'min_max',
                   simple_feature_vector = False,
                   plot = False,
                   color_by = 'distance',
                   plot_graph_stats = False,
                   run_sanity_check = False, 
                   epochs_sanity_check = 10,
                   print_sanity_stats = False,
                   print_when_done = False):
                  
    
    """ 
    returns:
    G - the graph with the feature vectors - G.edges[edge]['feature_vector']
    m_patient_is_sick -  binary 1D array for which patient is sick or not
    latent_process_cluster - id of the "evil" cluster
    comms - dict where keys are nodes and values are cluster ids 
    comms_inverted - dict where keys are cluster ids and values are nodes in the cluster
    max_clusters - upper limit on num of clusters (only works with 'greedy')
    limit_num_clusters - if True will use upper limit on num of clusters
    
    inputs:
    nodes- number of nodes to be created for the network
    percent_of_nodes- percent of nodes as measure for number of patients 
    prob_is_sick -  probability to assign a patient as sick (part of the latent process)
    m0_value - Number of edges to attach from a new node to existing nodes (when building
    the initial graph)
    clustering_choice - Choose one of {'louvain', 'greedy'}
    num_clusters_in_latent_processes -  how many clusters are a part of the latent 
    processes (i.e. "evil")
    p_cluster_NOT_part_of_process - probability that an edge will be assigned as "evil" 
    when not entirly inside "evil" cluster
    p_cluster_part_of_process - probability that an edge will be assigned as "evil" when 
    entirly inside "evil" cluster

    Beta, Gamma, Delta, Simga - probability for patient to contribute to edge
    Beta - sick patient and the edge is part of the latent process (evil):
    Gamma - healthy and the edge is evil
    Delta - sick and the edge is normal 
    Sigma - healthy and the edge is normal 
    NOTE - as one can see, due to Beta being larger, sick patients are more likely to 
    contribute to evil edges. 
    which in turn, it is more likely for an edge to be evil if it's in the cluster that 
    was assigned as evil.
    all this goes to make sure that for that given cluster(s) there will be a higher 
    percentage of sick people contributing
    to it's edges. 

    lower_correlation_bound - lower bound for sampling the correlation for sick patients 
    gaussian of an edge
    upper_correlation_bound - upper bound for sampling the correlation for sick patients
    gaussian of an edge
    lower_var_bound - lower bound for the sampling the variance of each gene (node) 
    upper_var_bound - upper bound for the sampling the variance of each gene (node)
    scale_healthy_var - scales the variance of healthy before sampling 
    
    model_choice - regression line model - recommended to use one robust to outliers 
    choose one of  {'ransac', 'huber', 'linear'}
    spearman - if True will use Spearman's correlation when fitting the line to the data 
    (will convert data to ranked)

    dist_measure - measure used to calc feature vector values ("correlation gain") - 
    associated with dist to regression line
    choose one of {'normal', 'inverse', 'inverse sqrt', 'outer lines', 'outliers'}
    normal - absolute distance (will give noisy points the highest value)
    inverse - takes 1/abs_dist,  giving higher values to closer points - but might 
    explode or divide by 0 for points on the line 
    (the dividing by zero is mostly an issue when using Spearmans)
    inverse sqrt - the sqrt of inverse - makes the very large values smaller
    outer lines - distance to the closest of the regression line when moved +- the 
    max_distance of any point. It means points
    in the center will have the largest distance and points on the edges 
    will have the smallest.
    outliers - binary distance measure (1 for inliers, 0 for outliers) 
    alpha score - scales residuals to normal dist. then takes two-tail prob

    scale_dists - if True will scale/normalize the distances 
    scaler_choice - choose one of {'min_max', 'standard', 'robust'}
    min_max - scales data between [0,1] - sensitive to outliers (only scales)
    standard - normalizes the data (X-Mean)/STD 
    robust - also normalizes the data, but is more robust to outliers (X-Median)/IQR

    simple_feature_vector - if True will return binary vector of which patients 
    contributed to the edge as the feature vector
    no gauss sampling, the values are entirely based on:
    Beta, Gamma, p_cluster_part_of_process, and p_cluster_NOT_part_of_process

    plot - if true will plot scatterplot and regression line of the last edge
    color_by - color points in scatter plot - #Choose one of {'distance', 'no color'}
    plot_graph_stats - if True will plot simply stats about the G graph. 
    plot_graph_stats - if True will plot simple stats about the G graph 
    (num of nodes per cluster and degree dist)
    run_sanity_check - if True will run sanity check to see if patient assignment 
    was done correctly (blocks 2, 3)
    epochs_sanity_check - number of times sanity check runs - takes the average
    print_sanity_stats - if True will print sanity check stats
    print_when_done - if True will print when it is done
    
    """
    
    #set number of patients as percent of nodes
    num_of_patients = int(round(percent_of_nodes * nodes))
        
    #Block 1 - generate power-law graph and cluster it 
    [G, comms] = barbasi_louvain(nodes, m0_value, clustering_choice, max_clusters, limit_num_clusters)
        
    # Block 1.5 - Create auxiliary dicts and choose "evil" cluster
    #generate aux dicts for future use and randomly choose "evil" cluster
    # comms_inverted - is an inverted dict, the keys are the cluster ids and the values are a set of all nodes in the cluster
    # edges_in_cluster - a dictionary of cluster keys where the values are edges entirely within the given cluster
    latent_process_cluster, num_of_clusters, comms_inverted, edges_in_cluster = aux_dicts(G, comms, clustering_choice)

    #if chosen will plot simple graph G stats - number of nodes per cluster and degree distribution
    if plot_graph_stats:
        plot_node_count_per_cluster(num_of_clusters, comms_inverted)
        plot_degree_distribution(G, nodes)
    
    
    #Block 2 - Assign edges as "evil" or not
    #m_hat_edges - binary array of which edges are evil or not 
    m_hat_edges = assign_edges_to_latent_process(G, edges_in_cluster, latent_process_cluster, 
                                   p_cluster_NOT_part_of_process, p_cluster_part_of_process)
    
    
    #Block 3 - for each edge decide which patients contribute to it
    #run the assign_patients_to_edges function:
    # m_hat_edges_values -          a binary list the size of Edges saying which edges are "evil"
    # edge_list_dict -              a dict where the keys are indexes 0-995, and the values are edges (70, 108) for example
    # m_patient_is_sick -           is a binary list saying for each patient if they are sick or not. 
    # m_patient_part_of_edge_corr - is a matrix of size Edges X Patients (for example 996 rows and 100 columns). 
    #                               It is a binary matrix that for each edge (each row) says which patients contributed to it
    (edge_list_dict, num_of_edges ,m_patient_is_sick, 
     m_patient_part_of_edge_corr, m_hat_edges_values) = assign_patients_to_edges(G, num_of_patients, m_hat_edges,
                                                                    prob_is_sick, Beta, Gamma, Delta, Sigma)
        
    
    #sanity check - test if avg number of sick and healhty patients per edge for each cluster matches the expected values
    if run_sanity_check:
        #run sanity checks multiple times and return average result:
        sanity_check_with_repeats(nodes, m0_value, clustering_choice, p_cluster_part_of_process, 
                                  p_cluster_NOT_part_of_process, num_of_patients, prob_is_sick, 
                                  Beta, Gamma, Delta, Sigma, epochs_sanity_check, print_sanity_stats, 
                                  max_clusters, limit_num_clusters)
                                

        # We expect the "evil" cluster to have a higher average number of sick per edge relative to the other clusters
        _, _, _, _, _, = sanity_check(G, prob_is_sick, Sigma, Delta, Beta, p_cluster_NOT_part_of_process, 
                                      p_cluster_part_of_process, edges_in_cluster, m_patient_part_of_edge_corr, 
                                      latent_process_cluster, num_of_clusters, num_of_patients, m_patient_is_sick, m_hat_edges)

    
    # Block 4 - generate feature vectors for each edge
    if simple_feature_vector:
        add_simple_feature_vectors(G, m_patient_part_of_edge_corr)

    else:
        #randomally generate gene variance scores for each gene - samplint is uniform from given range
        gene_vars = np.random.uniform(low = lower_var_bound, high = upper_var_bound, size = nodes)

        #generate a matrix: Nodes.X.Patients (500 X 100 for example) with gene amounts for given patient.
        #Only stores values for the healthy patients. used to save on wastful sampling of healhty patients for each edge  
        m_patient_gene_levels = generate_healthy_patients_gene_levels(nodes, num_of_patients, 
                                                                      m_patient_is_sick, gene_vars, scale_healthy_var)

        #generate the actual feature vector for each edge
        evil_edge_index = index_single_evil_edge(m_hat_edges_values)
        generate_feature_vectors(G, m_patient_part_of_edge_corr, m_patient_is_sick, m_patient_gene_levels,
                                 gene_vars, lower_correlation_bound, upper_correlation_bound, evil_edge_index, plot, scale_dists,
                                 model_choice, dist_measure, scaler_choice, color_by, spearman)
        
    
    
    #NEW DEEP PROJECT
    #removes the "m_hat" feature from the edges 
    for edge in G.edges:
        del G.edges[edge]['m_hat']

    if print_when_done:
        print("done \naccess feature vectors with: G.edges[edge]['feature_vector']")
        
    return G, m_patient_is_sick, latent_process_cluster, comms, comms_inverted



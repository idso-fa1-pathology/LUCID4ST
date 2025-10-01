import os
from glob import glob
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import json

# to get spatial clusters based on DBSCAN
src_path = '/path/to/pm_readout'
num_spot = 1
min_spots = 10
pgmn_cutoff = 0 
dst_path = f'/path/to/csv_{pgmn_cutoff}a_dbscan{num_spot}spt_min{min_spots}spt'
st_path = '/path/to/spatial-transcriptome'
if not os.path.isdir(dst_path):
    os.makedirs(dst_path)


center_distance = 398
files = sorted(glob(os.path.join(src_path, '*.csv')))
for file in files:
    file_name = os.path.basename(file)
    print(file_name)
    output_file = os.path.join(dst_path, file_name)

    data = pd.read_csv(file)
    st_json_path = os.path.join(st_path, file_name[:8], 'outs/spatial', 'scalefactors_json.json')
    with open(st_json_path, 'r') as json_file:
        diameter = json.load(json_file)['spot_diameter_fullres']
    
    distance_threshold = center_distance * num_spot + np.ceil(diameter/2 + 1)  
    # Assign 'pgmn' type for spots above the cutoff
    data['type'] = 'non_pgmn'  # Initialize with default
    data.loc[data['anthracosis_spot_per'] > pgmn_cutoff, 'type'] = 'pgmn'
    pgmn_spots = data[data['type'] == 'pgmn']

    # DBSCAN clustering
    def find_clusters(spots_df, distance_threshold):
        if spots_df.empty:
            spots_df['cluster'] = np.nan
            return spots_df
        coordinates = spots_df[['x_loc', 'y_loc']].values
        clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(coordinates)
        spots_df['cluster'] = clustering.labels_
        return spots_df

    pgmn_clusters = find_clusters(pgmn_spots, distance_threshold)

    if not pgmn_clusters.empty:
        initial_cluster_count = pgmn_clusters['cluster'].nunique()
        
        # Filter clusters with more than `min_spots`
        pgmn_clusters_filtered = pgmn_clusters.groupby('cluster').filter(lambda x: len(x) >= min_spots)
        
        # Count the number of clusters after filtering
        if not pgmn_clusters_filtered.empty:
            remaining_clusters = pgmn_clusters_filtered['cluster'].nunique()
        else:
            remaining_clusters = 0

        filtered_clusters = initial_cluster_count - remaining_clusters

        # Print the results
        print(f"Initial number of clusters: {initial_cluster_count}")
        print(f"Filtered out clusters: {filtered_clusters}")
        print(f"Remaining clusters: {remaining_clusters}")
    else:
        print("No clusters found in the data.")
        pgmn_clusters_filtered = pd.DataFrame()  # Handle empty case

    
    # Save filtered clusters and full data
    pgmn_clusters_filtered.to_csv(output_file, index=False)








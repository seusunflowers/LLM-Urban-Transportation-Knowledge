import math
import numpy as np
import scipy
import pandas as pd
from pandas import Series, DataFrame
from sklearn.manifold import TSNE

import osmnx
import geopandas as gpd
import shapely

import torch
from torch_geometric.nn import Node2Vec


class SpatialFeatureTools(object):
    
    @staticmethod
    def gcj02_to_wgs84(lng, lat):
        ''' Transformation from GCJ02 coordinate system to the WGS84 coordinate system '''

        def _transformlat(lng, lat):
            ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
                  0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
            ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
                    np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
            ret += (20.0 * np.sin(lat * np.pi) + 40.0 *
                    np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
            ret += (160.0 * np.sin(lat / 12.0 * np.pi) + 320 *
                    np.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
            return ret

        def _transformlng(lng, lat):
            ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
                  0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
            ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
                    np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
            ret += (20.0 * np.sin(lng * np.pi) + 40.0 *
                    np.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
            ret += (150.0 * np.sin(lng / 12.0 * np.pi) + 300.0 *
                    np.sin(lng / 30.0 * np.pi)) * 2.0 / 3.0
            return ret
        
        x_pi = math.pi * 3000.0 / 180.0
        a = 6378245.0  # semi-major axis
        ee = 0.00669342162296594323  # square of eccentricity

        dlat = _transformlat(lng - 105.0, lat - 35.0)
        dlng = _transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * np.pi
        magic = np.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = np.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * np.pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * np.pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return lng * 2 - mglng, lat * 2 - mglat


    @staticmethod
    def get_pois(geometries:gpd.GeoDataFrame, radius:float=0):
        '''Retrieve POIs within or nearby the specified geometries'''
        geometries['geometry'] = geometries.buffer(radius)
        pois = osmnx.features.features_from_polygon(polygon=geometries.to_crs(epsg=4326).union_all(),
                                                    tags={'landuse': ['commercial', 'construction', 'education', 'fairground', 'industrial', 'residential', 'retail', 'retail', 'institutional']}).reset_index().to_crs(geometries.crs)
        pois = pois[pois['element']=='way']
        pois = pois[['id', 'geometry', 'landuse']]
        pois = gpd.sjoin(geometries, pois, how='left')
        pois.fillna({'landuse': 'empty'}, inplace=True)
        return pois


    @staticmethod
    def get_poi_distribution(geometries:gpd.GeoDataFrame, node_col_name:str, radius:float=0):
        '''Retrieve POI distribution within or nearby the specified geometries'''
        pois = SpatialFeatureTools.get_pois(geometries=geometries, radius=radius)
        pois['count'] = 1
        poi_distribution = pois[[node_col_name, 'landuse', 'count']].groupby(by=[node_col_name, 'landuse']).count().reset_index()
        poi_distribution = poi_distribution.pivot(index=node_col_name, columns='landuse', values='count')
        poi_distribution.fillna(0, inplace=True)
        if 'empty' in poi_distribution.columns:
            del poi_distribution['empty']
        return poi_distribution


    @staticmethod
    def get_connected_graph(coords:np.ndarray, saved_links_ratio:float=1, min_weight:float=0.05, random_state:int=42):
        '''coords: N x 2，N denotes node number
           saved_links_ratio: The proportion of edges retained  based on the TSNE metric, (=1 represents the retention of the top 1% of the most similar nodes)
           min_weight: Minimum edge weight. Let all edges with weights greater than 0 but less than min_weight have their weights set to min_weight
        '''
        N, _ = coords.shape
        coords_tsne = TSNE(n_components=1, learning_rate='auto', init='random', random_state=random_state).fit_transform(coords)
        ## Minimal connected graph construction
        dist = np.repeat(coords_tsne, repeats=N, axis=1) - np.repeat(coords_tsne.reshape(1,-1), repeats=N, axis=0)
        min_con_A = dist.copy()
        min_con_A[min_con_A <= 0] = np.nan
        min_con_A[min_con_A > np.nanmin(min_con_A, axis=1, keepdims=True)] = np.nan
        min_con_A[np.isnan(min_con_A)] = 0
        min_con_A = min_con_A + min_con_A.T # directed graph to undirected graph
        min_con_A[min_con_A == 0] = np.nan
        ## Connected graph construction. Retain OD pairs based on given saved_links_ratio
        A = np.fabs(dist)
        A[A > np.percentile(A, saved_links_ratio)] = np.nan
        A = np.where(np.isnan(A), min_con_A, A)
        A = np.exp(-(A / np.nanstd(A))**2)
        A[np.isnan(A)] = 0
        A[(A < min_weight) & (A > 0)] = min_weight
        ## Connectivity evaluation
        graph = scipy.sparse.csr_array(A)
        n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=graph, directed=False, return_labels=True)
        assert n_components == 1 # =1 indicates complete connectivity
        return A


    @staticmethod
    def get_node_rand_emb(node_num:int, node_e_dim:int, random_state:int=42):
        return torch.randn(node_num, node_e_dim, generator=torch.Generator().manual_seed(random_state))


    @staticmethod
    def get_node_onehot_emb(node_num:int):
        return torch.eye(node_num)[:,:-1]


    @staticmethod
    def get_node2vec_emb(edge_index:torch.Tensor, node_e_dim:int, batch_size:int, epoch:int=200, logging_steps:int=40, lr:float=0.01, walk_length:int=20, context_size:int=10, walks_per_node:int=10, 
        num_negative_samples:int=1, p:float=1.0, q:float=1.0, sparse:bool=True, random_state:int=42):

        import sys
        num_workers = 4 if sys.platform == 'linux' else 0

        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        
        def train_node2vec(model, loader):
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        node2vec_model = Node2Vec(edge_index, embedding_dim=node_e_dim, walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node,
                                  num_negative_samples=num_negative_samples, p=p, q=q, sparse=sparse)
        loader = node2vec_model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers) # num_workers = 4 if sys.platform == 'linux' else 0
        optimizer = torch.optim.SparseAdam(list(node2vec_model.parameters()), lr=lr)
        for epoch in range(1, epoch+1):
            loss = train_node2vec(node2vec_model, loader)
            if epoch % logging_steps == 0 or epoch == 1:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        with torch.no_grad():
            node2vec_model.eval()
            node_embeddings = node2vec_model()
        return node_embeddings


if __name__ == "__main__":
    spatial_tools = SpatialFeatureTools()
    print(spatial_tools.gcj02_to_wgs84(120.328249, 36.658946))
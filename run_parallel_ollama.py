import ollama
import calendar
import random
import tqdm
import subprocess
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def get_knearest_nodes(A:np.ndarray, k:int):
    assert k >= 1 and k <= A.shape[1] and A.shape[0] == A.shape[1]
    A = A.copy()
    A[np.arange(A.shape[0]), np.arange(A.shape[0])] = 0
    return np.argsort(A, axis=1)[:,-1:-1*(k+1):-1]


def prompt_based_ollama_inference(llm_name:str, data:DataFrame, hist_win:int, pred_win:int, neighbor_indices:np.ndarray, elem_iter:iter, query_template:str, var_unit:str,
                                  dayofweek_list:list=None, sample_ratio:float=1e-3, random_seed:int=0, gpu_id=0, ollama_host='http://127.0.0.1:11433'):
    """
    Executes time-series forecasting by prompting an LLM via the Ollama API.

    This function initializes a local Ollama server, iterates through spatial nodes, samples historical time windows, and constructs textual prompts containing 
    target and neighbor data for the LLM to predict future values.

    Args:
        llm_name (str): Name of the model registered in Ollama (e.g., 'qwen3:14b').
        data (DataFrame): Preprocessed time-series data where columns are nodes/features.
        hist_win (int): Number of historical time steps to provide in the prompt.
        pred_win (int): Number of future time steps the model is expected to predict.
        neighbor_indices (np.ndarray): NxK array containing indices of spatial neighbors for each node.
        elem_iter (iter): Iterator providing metadata (strings) for each node (e.g., names, locations).
        query_template (str): Format string for the prompt.
        var_unit (str): Unit of measurement for the values (e.g., 'km/h', 'pax/h').
        dayofweek_list (list, optional): List of strings for days (e.g., ['Mon', ...]). Defaults to None.
        sample_ratio (float): Probability of sampling a specific time window for inference. Defaults to 1e-3.
        random_seed (int): Seed for reproducibility of the random sampling. Defaults to 0.
        gpu_id (int): CUDA device ID to assign to the Ollama server. Defaults to 0.
        ollama_host (str): Network address for the Ollama server. Defaults to 'http://127.0.0.1:11433'.

    Returns:
        DataFrame: Results containing node_id, time index, the full prompt, and the raw LLM response.
    """
    # This flag tells Windows to open a brand new, visible console window
    CREATE_NEW_CONSOLE = 0x00000010
    subprocess.Popen(['powershell', '-Command', f'$env:CUDA_VISIBLE_DEVICES="{gpu_id}"; $env:OLLAMA_HOST="{ollama_host}"; ollama serve'], creationflags=CREATE_NEW_CONSOLE)
    client = ollama.Client(host=ollama_host)
    time.sleep(10)
    
    inference_results = []

    random.seed(random_seed)
    for node_id, ((nearest_node_id, second_nearest_node_id), elem) in tqdm.tqdm(enumerate(zip(neighbor_indices, elem_iter)),
                                                                                total=len(neighbor_indices), position=gpu_id, desc=f"GPU {gpu_id}"):
        elem = elem if isinstance(elem, (list, tuple)) else (elem,)
        for i in range(len(data) - (hist_win + pred_win) + 1):
            if random.random() < sample_ratio:
                hist_from_time, hist_to_time = data.iloc[i]['daytime'], data.iloc[i+hist_win-1]['daytime']
                if dayofweek_list is not None:
                    hist_from_time = dayofweek_list[data.iloc[i]['dayofweek']] + ' ' + hist_from_time
                    hist_to_time = dayofweek_list[data.iloc[i+hist_win-1]['dayofweek']] + ' ' + hist_to_time
                node_hist_pattern = data.iloc[i:i+hist_win, node_id]
                nearest_node_hist_pattern = data.iloc[i:i+hist_win, nearest_node_id]
                second_nearest_node_hist_pattern = data.iloc[i:i+hist_win, second_nearest_node_id]
                node_hist_pattern_text = f' {var_unit}, '.join(node_hist_pattern.round(0).astype(str)) + f' {var_unit}'
                nearest_node_hist_pattern_text = f' {var_unit}, '.join(nearest_node_hist_pattern.round(0).astype(str)) + f' {var_unit}'
                second_nearest_node_hist_pattern_text = f' {var_unit}, '.join(second_nearest_node_hist_pattern.round(0).astype(str)) + f' {var_unit}'
                query_prompt = query_template.format(*elem, hist_from_time, hist_to_time, node_hist_pattern_text, nearest_node_hist_pattern_text, second_nearest_node_hist_pattern_text)
                response = client.chat(model=llm_name, messages=[{"role": 'user', "content": query_prompt},])
                inference_results.append([node_id, i, query_prompt, response["message"]['content']])
    inference_results = DataFrame(inference_results, columns=['node_id', 'hist_time_index', 'prompt', 'llm_content'])
    return inference_results


def run_metrla(gpu_id:int, ollama_host:str, ollama_model:str):
    """ Pipeline for Los Angeles highway traffic speed forecasting. Fills missing values using the average speed for that specific time of day and workday status.
    """
    HIST_TIME_WINDOW = 12
    PRED_TIME_WINDOW = 24
    DAILY_TIME_STEPS = 288
    RAW_DATA_DICT = 'metr-la'
    VAR_UNIT = 'km/h'

    ## get highway traffic speed in LA
    station_info = pd.read_csv(f'{RAW_DATA_DICT}/graph_sensor_metadata.csv', usecols=[1, 5, 12, 13, 14, 15])
    NODE_NUM = len(station_info)
    data = pd.read_csv(f'{RAW_DATA_DICT}/metr-la.csv')
    data.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    data['time'] = pd.to_datetime(data['time'])
    data.replace(0, np.nan, inplace=True)
    # fill missing data
    dayofweek = data['time'].dt.weekday.values
    data['workday'] = (dayofweek < 5).astype(int)
    data['daytime'] = data['time'].dt.strftime('%H:%M')
    typical_data = data.iloc[:18*DAILY_TIME_STEPS,1:].groupby(['daytime', 'workday']).mean() # 取前 18 天数据消掉所有 NaN
    data = data.merge(right=typical_data, left_on=['daytime', 'workday'], right_index=True, suffixes=['', '_t'])
    for det_id in data.columns[1: NODE_NUM+1]:
        data.fillna({det_id: data[f'{det_id}_t']}, inplace=True)
        del data[f'{det_id}_t']
    data['dayofweek'] = dayofweek
    data.sort_values(by='time', inplace=True)
    del data['time'], data['workday']

    ## prompt-based inference
    neighbor_indices = get_knearest_nodes(A=np.load(f'{RAW_DATA_DICT}/adj_mat.npy'), k=2
    elem_iter = ((fwy_name + '-' + fwy_dir, det_name, det_city, det_county) for _, _, fwy_dir, det_name, det_county, fwy_name, det_city in station_info.itertuples())
    query_template = 'Considering a segment of the {} Freeway around {} in {}, {} County, California, USA, its speed observations from {} to {} in every five minutes are {}. ' +\
                     'The typical speed observations of its two nearest segments are {} and {}, respectively. Please predict its next twenty-four speed observations. ' + \
                     'Please analyze carefully then directly output the numerical results in Python list format without additional texts.'
    inference_results = prompt_based_ollama_inference(llm_name=ollama_model, data=data, hist_win=HIST_TIME_WINDOW, pred_win=PRED_TIME_WINDOW, neighbor_indices=neighbor_indices,
                                                      elem_iter=elem_iter, query_template=query_template, var_unit=VAR_UNIT, sample_ratio=4e-4, gpu_id=gpu_id, ollama_host=ollama_host)
    inference_results.to_csv(f'{RAW_DATA_DICT}/llm-description/description-qwen3-14b/direct-inference-qwen3-14b.csv.zip', compression='zip')



def run_bart(gpu_id:int, ollama_host:str, ollama_model:str):
    """ Pipeline for San Francisco BART ridership forecasting. Aggregates exit counts by destination and hour, handling missing hours by inserting zeros.
    """
    HIST_TIME_WINDOW = 5
    PRED_TIME_WINDOW = 7
    RAW_DATA_DICT = 'bart'
    VAR_UNIT = 'pax/h'

    ## get ridership data in Bay Area
    station_info = pd.read_excel(f'{RAW_DATA_DICT}/station-names.xls', index_col=0)
    station_info = station_info.loc[np.load(f'{RAW_DATA_DICT}/station-based-st-graph.npz', allow_pickle=True)['station_ids']]
    # aggregate station-based ridership
    data = pd.read_csv(f'{RAW_DATA_DICT}/2018-2025-raw/date-hour-soo-dest-2024.csv.gz', names=['Date', 'Hour', 'Origin', 'Destination', 'Exits'])
    data['Time'] = data['Date']  + ' ' + data['Hour'].map(lambda x: f'{x:02d}')
    data = data[['Destination', 'Time', 'Exits']]
    data = data.groupby(['Destination', 'Time']).sum().reset_index()
    # fill missing data
    start_time, end_time = data['Time'].iloc[0], data['Time'].iloc[-1]
    time_index = pd.date_range(start=f'{start_time}:00', end=f'{end_time}:00', freq='1h')
    observed_time_index = set(data['Time'])
    missing_data = DataFrame(data=[['12TH', t, 0] for t in time_index.strftime('%Y-%m-%d %H') if t not in observed_time_index], columns=data.columns)
    data = pd.concat([data, missing_data], ignore_index=True).sort_values(by=['Time', 'Destination'])
    data = data.pivot(index='Time', columns='Destination', values='Exits')
    data.fillna(0, inplace=True)
    data['daytime'], data['dayofweek'] = time_index.strftime('%H:%M'), time_index.dayofweek

    ## prompt-based inference
    neighbor_indices = get_knearest_nodes(A=np.load(f'{RAW_DATA_DICT}/station-based-st-graph.npz', allow_pickle=True)['A'], k=2)
    elem_iter = station_info['Station Name']
    query_template = 'Considering the {} Station of the San Francisco Bay Area Rapit Transit System, its ridership observations from {} to {} in every one-hour are {}. ' +\
                     'The typical ridership observations of its two nearest stations are {} and {}, respectively. Please predict its next seven reidership observations. ' + \
                     'Please analyze carefully then directly output the numerical results in Python list format without additional texts.'
    inference_results = prompt_based_ollama_inference(llm_name=ollama_model, data=data, hist_win=HIST_TIME_WINDOW, pred_win=PRED_TIME_WINDOW, neighbor_indices=neighbor_indices,
                                                      elem_iter=elem_iter, query_template=query_template, var_unit=VAR_UNIT, dayofweek_list=calendar.day_name, sample_ratio=1e-3,
                                                      gpu_id=gpu_id, ollama_host=ollama_host)
    inference_results.to_csv(f'{RAW_DATA_DICT}/description-qwen3-14b/direct-inference-qwen3-14b.csv.zip', compression='zip')

    

def run_urbanev(gpu_id:int, ollama_host:str, ollama_model:str):
    """ Pipeline for Shenzhen EV charging occupancy forecasting. Uses a Chinese-language prompt and calculates a spatial adjacency matrix 
    using a Gaussian kernel based on physical distances between charging zones.
    """
    HIST_TIME_WINDOW = 5
    PRED_TIME_WINDOW = 7
    RAW_DATA_DICT = 'UrbanEV'
    VAR_UNIT = '%'

    ## get electric vehicle charing occupancy data in Shenzhen City
    zones_points = pd.read_excel(f'{RAW_DATA_DICT}/zone-bus-metro-stations.xlsx', usecols=[2,12,14])
    data = pd.read_csv(f'{RAW_DATA_DICT}/occupancy.csv')
    time_index = pd.to_datetime(data['time'])
    del data['time']
    NODE_NUM = len(data.columns)
    data['daytime'], data['dayofweek'] = time_index.dt.strftime('%H:%M'), time_index.dt.dayofweek

    ## calculate adjacency matrix
    pre_A = pd.read_csv(f'{RAW_DATA_DICT}/adj.csv').values
    zone_dist = pd.read_csv(f'{RAW_DATA_DICT}/distance.csv').values
    zone_dist[(zone_dist > np.percentile(zone_dist, 5)) & (pre_A == 0)] = np.nan
    adj_mat = np.exp(-(zone_dist / np.nanstd(zone_dist))**2 * 0.3)
    adj_mat[np.isnan(adj_mat)] = 0
    adj_mat[(adj_mat < 0.05) & (adj_mat > 0)] = 0.
    neighbor_indices = get_knearest_nodes(A=adj_mat, k=2)

    ## prompt-based inference
    elem_iter = (sub_points['DISTRICT'].iloc[0] + '、'.join(sub_points['name']) for _, sub_points in zones_points.groupby('TAZID'))
    query_template = '考虑中国广东省深圳市{}附近区域，其{}至{}的每小时电动汽车充电站总占用率为 {}。同时段内其两个相邻最近区域的电动汽车充电站总占用率分别为 {} 和 {}。' +\
                     '请预测该区域接下来的七个观测结果。请认真思考后直接将预测结果输出为 Python 列表的形式并避免输入其它内容。'
    inference_results = prompt_based_ollama_inference(llm_name=ollama_model, data=data, hist_win=HIST_TIME_WINDOW, pred_win=PRED_TIME_WINDOW, neighbor_indices=neighbor_indices,
                                                      elem_iter=elem_iter, query_template=query_template, var_unit=VAR_UNIT,
                                                      dayofweek_list=['周一', '周二', '周三', '周四', '周五', '周六', '周日'], sample_ratio=1e-3, gpu_id=gpu_id, ollama_host=ollama_host)
    inference_results.to_csv(f'{RAW_DATA_DICT}/llm-description/description-qwen3-14b/direct-inference-qwen3-14b.csv.zip', compression='zip')



def run_subwaymta(gpu_id:int, ollama_host:str, ollama_model:str):
    """ Pipeline for New York City Subway ridership forecasting. Processes large-scale hourly ridership data, grouping by station complex and borough.
    """
    HIST_TIME_WINDOW = 5
    PRED_TIME_WINDOW = 7
    RAW_DATA_DICT = 'subway-mta'
    VAR_UNIT = 'pax/h'

    ## get ridership data in New York City
    hourly_ridership = pd.read_csv(f'{RAW_DATA_DICT}/MTA_Subway_Hourly_Ridership__2020-2024_20250317.zip', usecols=[0, 3, 4, 7, 9, 10])
    station_info = hourly_ridership.drop_duplicates(subset=['station_complex', 'borough'])
    station_info.set_index('station_complex', inplace=True)
    station_info = station_info.loc[np.load(f'{RAW_DATA_DICT}/station-based-st-graph.npz', allow_pickle=True)['station_ids']]
    # aggregate station-based ridership
    del hourly_ridership['borough']
    hourly_ridership['transit_timestamp'] = pd.to_datetime(hourly_ridership['transit_timestamp'], format="%m/%d/%Y %I:%M:%S %p")
    hourly_ridership = hourly_ridership[['transit_timestamp', 'station_complex', 'ridership']].groupby(['transit_timestamp', 'station_complex']).sum().reset_index()
    # fill missing data
    start_time, end_time = hourly_ridership['transit_timestamp'].iloc[0], hourly_ridership['transit_timestamp'].iloc[-1]
    time_index = pd.date_range(start=start_time, end=end_time, freq='1h')
    observed_time_index = set(hourly_ridership['transit_timestamp'])
    missing_data = DataFrame(data=[[t, '1 Av (L)', 0] for t in time_index if t not in observed_time_index], columns=hourly_ridership.columns)
    data = pd.concat([hourly_ridership, missing_data], ignore_index=True).sort_values(by=['transit_timestamp', 'station_complex'])
    data = data.pivot(index='transit_timestamp', columns='station_complex', values='ridership')
    data.fillna(0, inplace=True)
    data['daytime'], data['dayofweek'] = time_index.strftime('%H:%M'), time_index.dayofweek

    ## prompt-based inference
    neighbor_indices = get_knearest_nodes(A=np.load(f'{RAW_DATA_DICT}/station-based-st-graph.npz', allow_pickle=True)['A'], k=2)
    elem_iter = zip(station_info.index, station_info['borough'])
    query_template = 'Considering the {} Station in {} of the New York City Subway System, its ridership observations from {} to {} in every one-hour are {}. ' +\
                     'The typical ridership observations of its two nearest stations are {} and {}, respectively. Please predict its next seven reidership observations. ' + \
                     'Please analyze carefully then directly output the numerical results in Python list format without additional texts.'
    inference_results = prompt_based_ollama_inference(llm_name=ollama_model, data=data, hist_win=HIST_TIME_WINDOW, pred_win=PRED_TIME_WINDOW, neighbor_indices=neighbor_indices,
                                                      elem_iter=elem_iter, query_template=query_template, var_unit=VAR_UNIT, dayofweek_list=calendar.day_name, sample_ratio=8.5e-4,
                                                      gpu_id=gpu_id, ollama_host=ollama_host)
    inference_results.to_csv(f'{RAW_DATA_DICT}/description-qwen3-14b/direct-inference-qwen3-14b.csv.zip', compression='zip')



if __name__ == '__main__':
    processes = []
    for i, func in enumerate([run_metrla, run_bart, run_urbanev, run_subwaymta]):
        p = mp.Process(target=func, kwargs={'gpu_id':i, 'ollama_host':f'http://127.0.0.1:{11433+i}', 'ollama_model':'qwen3:14b'})
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

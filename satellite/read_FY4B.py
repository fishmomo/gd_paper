#### env == ai
"""
CLP产品说明
0:clear
1:water
2:super cooled
3:mixed
4:ice
5:uncertain
126:space
127:fillvalue
"""
import gzip
import pickle
from pathlib import Path
import re

import xarray as xr
import netCDF4
import numpy as np
import pandas as pd


def _get_field(data_dict, candidates, required=True):
    for key in candidates:
        if key in data_dict:
            return np.asarray(data_dict[key]), key
    if required:
        raise KeyError(f"未找到字段，候选键: {candidates}")
    return None, None


def load_station_meta(gd_station_fpath, country_station_fpath):
    gd_station_data = pd.read_csv(gd_station_fpath)
    gd_station_id = gd_station_data['station'].values.astype(str)


    country_station_data = pd.read_csv(country_station_fpath, encoding='gbk')
    country_station_id = country_station_data['station_id'].values.astype(str)

    indice_valid = np.isin(country_station_id, gd_station_id)
    return country_station_id, indice_valid


def process_file(file_path, country_station_id, indice_valid, precip_ncpath):
    try:
        with gzip.open(file_path, 'rb') as f:
            result_data = pickle.load(f)
    except (EOFError, gzip.BadGzipFile, pickle.UnpicklingError, OSError, ValueError) as e:
        print(f"[跳过] 文件读取失败: {Path(file_path).name} | {type(e).__name__}: {e}")
        return None

    #### 在 indice_valid 基础上筛选暖云类型云参数 COT, CTH, CER, CTT
    try:
        clp_data, clp_key = _get_field(result_data, ['CLP'])
        cot_data, cot_key = _get_field(result_data, ['COT'])
        cth_data, cth_key = _get_field(result_data, ['CTH'])
        cer_data, cer_key = _get_field(result_data, ['CER'])
        ctt_data, ctt_key = _get_field(result_data, ['CTT'])
    except KeyError as e:
        print(f"[跳过] 关键字段缺失: {Path(file_path).name} | {e}")
        return None

    # 暖云条件: CLP=water(1) 且云顶温度 > 273.15K
    warm_cloud_clp = np.isin(clp_data, [1])
    warm_cloud_ctt = ctt_data > 273.15
    warm_station_mask = indice_valid & warm_cloud_clp & warm_cloud_ctt 

    filename = Path(file_path).name
    time_matches = re.findall(r'\d{14}', filename)
    if not time_matches:
        print(f"[跳过] 文件名未找到14位时间戳: {filename}")
        return None
    YYMMDDhhmmss = time_matches[0]
    YY = YYMMDDhhmmss[:4]
    MM = YYMMDDhhmmss[4:6]
    DD = YYMMDDhhmmss[6:8]
    hh = YYMMDDhhmmss[8:10]
    file_time = f"{YYMMDDhhmmss[8:10]}:{YYMMDDhhmmss[10:12]}:{YYMMDDhhmmss[12:14]}"
    YYMMDDhh_precip = np.datetime64(f"{YY}-{MM}-{DD}T{int(hh):02}:00:00") + np.timedelta64(1, 'h')
    warm_cloud_df = pd.DataFrame({
        'file_name': filename,
        'time': file_time,
        'station_id': country_station_id[warm_station_mask],
        'COT': cot_data[warm_station_mask],
        'CTH': cth_data[warm_station_mask],
        'CER': cer_data[warm_station_mask],
        'CTT': ctt_data[warm_station_mask],
    })

    #### 在暖云参数基础上进一步筛选有降水的索引
    warm_valid_stationid = country_station_id[warm_station_mask]

    precip_data = xr.open_dataset(precip_ncpath)
    precip_stationid = precip_data.station.values
    precip_time = precip_data.time.values
    march_indice = np.where(precip_time == YYMMDDhh_precip)[0]
    if march_indice.size > 0:
        precip_arr = precip_data.precip_1h.values[march_indice,:].reshape([88,])
    else:
        return None
    precip_march = []
    for i, id_stat in enumerate(precip_stationid):
        if id_stat in warm_valid_stationid:
            precip_march.append(precip_arr[i])
    precip_mask = np.isin(country_station_id, precip_stationid)
    # precip_data, precip_key = _get_field(
    #     result_data,
    #     ['RR', 'rr', 'RAIN', 'rain', 'PRECIP', 'precip', 'QPE', 'Precipitation'],
    #     required=False
    # )

    # if precip_data is None:
    #     precip_mask = np.zeros_like(warm_station_mask, dtype=bool)
    # else:
    #     precip_mask = precip_data > 0

    precip_df = warm_cloud_df[precip_mask[warm_station_mask]].reset_index(drop=True)
    precip_df["precipitation"] = np.array(precip_march)
    precip_index = np.where(warm_station_mask & precip_mask)[0]
    print(f"[{Path(file_path).name}] 暖云站点数: {warm_station_mask.sum()} | 匹配降水站点数: {len(precip_index)}")
    print(
        f"字段映射 -> CLP:{clp_key}, COT:{cot_key}, CTH:{cth_key}, "
        f"CER:{cer_key}, CTT:{ctt_key}, 降水:{'precipitation'}"
    )
    return warm_cloud_df, precip_df, precip_index


def main():
    data_dir1 = Path(r"H:\YeWu\Zhou\guangzhou\FY-out-2025")
    data_dir2 = Path(r"H:\YeWu\Zhou\guangzhou\FY-out-2026")
    gd_station_fpath = Path(r"H:\YeWu\Zhou\guangzhou\points_in_gd_shape.csv")
    country_station_fpath = Path(r"H:\YeWu\Zhou\guangzhou\stationss.csv")
    precip_ncpath = Path(r"H:\YeWu\Zhou\guangzhou\rain_1h.nc")

    country_station_id, indice_valid = load_station_meta(gd_station_fpath, country_station_fpath)
    print(indice_valid.sum())

    file_list = sorted(list(data_dir1.glob("*.pkl.gz"))+list(data_dir2.glob("*.pkl.gz")))
    if not file_list:
        raise FileNotFoundError(f"目录下未找到 .pkl.gz 文件: {data_dir1}\n{data_dir2}")

    all_warm_df = []
    all_precip_df = []
    skip_count = 0

    for file_path in file_list:
        result = process_file(file_path, country_station_id, indice_valid, precip_ncpath)
        if result is None:
            skip_count += 1
            continue
        warm_cloud_df, precip_df, _ = result
        all_warm_df.append(warm_cloud_df)
        all_precip_df.append(precip_df)

    warm_cloud_all = pd.concat(all_warm_df, ignore_index=True) if all_warm_df else pd.DataFrame()
    precip_all = pd.concat(all_precip_df, ignore_index=True) if all_precip_df else pd.DataFrame()

    precip_all.to_csv(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427.csv")
    print(f"总文件数: {len(file_list)}")
    print(f"跳过文件数: {skip_count}")
    print(f"暖云样本总数: {len(warm_cloud_all)}")
    print(f"有降水样本总数: {len(precip_all)}")


if __name__ == '__main__':
    main()

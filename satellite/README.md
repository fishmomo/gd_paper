# satellite 代码索引

这个目录主要用于把 FY4B 卫星暖云产品、站点降水资料和广东站点信息匹配起来，生成暖云降水样本，并围绕 COT、CER、CTH、CTT 与降水强度做统计、聚类和可视化。

给后续对话快速定位用：如果要找“数据读取/样本生成”，优先看 `read_FY4B.py`、`read_station_precipitation.py`；如果要找“站点-小时聚合统计”，看 `summarize_fy4b_station_cloud_stats.py`；如果要找“分箱/概率/强度比例”，看 `analyze_*`、`compare_*`、`plot_*_ratios.py`、`plot_*_heatmaps.py`；如果要找“散点/KDE/3D/聚类”，看 `plot_fy4b_cloud_*` 和 `cluster_fy4b_precip_pair_kmeans.py`。

## 推荐处理流程

1. `read_station_precipitation.py`
   将站点小时降水 txt 转成 `rain_1h.nc`，供 FY4B 匹配降水使用。

2. `read_FY4B.py`
   读取 FY4B `.pkl.gz` 云产品，筛选广东站点暖云样本，并与下一小时降水匹配，输出暖云有降水样本 CSV。

3. `filter_fy4b_positive_precip.py`
   从 FY4B 输出表中筛选 `precipitation > 0` 的记录，并按文件、小时、站点排序。

4. `summarize_fy4b_station_cloud_stats.py`
   按 `date + hour + station_id` 聚合 COT、CTH、CER、CTT，生成每站每小时的 `max` 和 `mean` 云参数统计表。

5. 后续分析脚本
   基于聚合表继续做分箱比例、降水概率、强度分布、散点图、KDE、DBSCAN 聚类、月-小时频率和站点地图。

## 脚本说明

| 文件 | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `read_station_precipitation.py` | 读取站点小时降水 txt，清理重复表头，按 `time x station` 透视成 xarray Dataset。 | `data_20250201000000-20260201000000.txt` | `rain_1h.nc` |
| `read_FY4B.py` | 读取 2025/2026 FY4B `.pkl.gz`，提取 CLP、COT、CTH、CER、CTT；筛选暖云 `CLP=1` 且 `CTT > 273.15K`；匹配站点下一小时降水。 | `FY-out-2025/*.pkl.gz`、`FY-out-2026/*.pkl.gz`、站点 CSV、`rain_1h.nc` | `warmcloud_allprecip_0427.csv` |
| `read_FY4B copy.py` | `read_FY4B.py` 的旧/备份版本，只读取文件名匹配 `*5959_4000M_V0001.pkl.gz` 的 FY4B 文件。 | 同上，但 FY4B 文件过滤更窄 | `warmcloud_allprecip.csv` |
| `filter_fy4b_positive_precip.py` | 过滤 `precipitation > 0` 的暖云样本，并按文件、小时、站点排序。 | `warmcloud_allprecip_0427.csv` | `warmcloud_allprecip_0427_positive_by_station.csv` |
| `summarize_fy4b_station_cloud_stats.py` | 按日期、小时、站点聚合云参数，输出 `file_count`、平均降水、各云参数 `max/mean`；支持 `--precip-mode positive/zero`。无降水模式会按 `--min-file-count` 筛选，默认要求 `file_count >= 3`。 | positive 或 zero 降水样本 CSV | `*_date_hour_station_cloud_stats.csv` |
| `analyze_fy4b_precip_bins.py` | 对原始有降水暖云样本做 COT、CTH、CER、CTT 分箱，统计每个云参数分箱内的样本数和占比，并画组合条形图。 | `warmcloud_allprecip.csv` | `test_allprecip_filtered_positive_binned.csv`、`test_allprecip_bin_summary.csv`、`test_allprecip_bin_ratio.png` |
| `compare_fy4b_precip_no_precip_bins.py` | 比较有降水与无降水样本在 COT、CER、CTH、CTT 分箱中的比例差异。 | 降水/无降水站点聚合统计 CSV | `fy4b_precip_no_precip_bin_summary_*.csv`、`fy4b_precip_no_precip_bin_ratio_*.png` |
| `plot_fy4b_cloud_bins_precip_intensity_ratios.py` | 将云参数分箱后，统计不同降水强度等级在各云参数分箱内的比例，并画分组柱状图。 | `*_date_hour_station_cloud_stats.csv` | `fy4b_cloud_bin_rain_intensity_ratio_*.csv/png` |
| `plot_fy4b_cot_cer_joint_intensity_ratios.py` | 联合 COT 与 CER 分箱，统计不同降水强度等级比例。 | `*_date_hour_station_cloud_stats.csv` | `fy4b_cot_cer_joint_rain_intensity_ratio_*.csv` |
| `plot_fy4b_precip_probability_heatmaps.py` | 对云参数两两组合分箱，计算各分箱中的降水概率，并画热力图。 | 降水/无降水站点聚合统计 CSV | `fy4b_precip_probability_pair_heatmap_summary_*.csv`、`fy4b_precip_probability_pair_heatmaps_*.png` |
| `plot_warmcloud_precip_intensity_histogram.py` | 统计暖云降水强度等级频数和比例，并画降水强度直方图。 | `*_date_hour_station_cloud_stats.csv` | `warmcloud_precip_intensity_histogram_*.csv/png` |
| `plot_fy4b_cloud_pair_scatter.py` | 画 COT、CER、CTH、CTT 两两组合的有降水/抽样无降水散点图。 | 降水/无降水站点聚合统计 CSV | `fy4b_cloud_pair_scatter_*.png`、`fy4b_cloud_pair_scatter_no_precip_*.png` |
| `plot_fy4b_equal_no_precip_pair_scatter_kde.py` | 将无降水样本随机打乱后按降水样本数等量、无放回、最大次数切分；每次抽样只绘制无降水样本，输出 6 个双云参数组合在同一张 2x3 图中的散点图和 KDE 图。 | 降水/无降水站点聚合统计 CSV | `equal_no_precip_pair_combined_scatter_kde/` 下的逐次抽样组合散点图、组合 KDE 图和 manifest CSV |
| `compute_equal_no_precip_kde_similarity.py` | 复现无降水样本等量无放回抽样，保存每次抽样的二维 KDE 网格，并计算 22 次抽样之间的 JSD、Hellinger distance 和 Cosine similarity。 | 降水/无降水站点聚合统计 CSV | `equal_no_precip_kde_similarity/` 下的 KDE 网格 NPZ、指标矩阵 NPZ、两两指标 CSV、汇总 CSV 和热力图 |
| `plot_fy4b_cloud_pair_scatter_by_intensity.py` | 画云参数两两散点图，并按降水强度等级着色。 | 降水站点聚合统计 CSV | `fy4b_cloud_pair_scatter_by_intensity_*.png` |
| `plot_fy4b_cloud_pair_kde.py` | 画云参数两两组合的 KDE 等值线分布图，分别输出有降水和抽样无降水。 | 降水/无降水站点聚合统计 CSV | `fy4b_cloud_pair_kde_precip_*.png`、`fy4b_cloud_pair_kde_no_precip_*.png` |
| `plot_fy4b_cloud_pair_kde_all_no_precip_50contour.py` | 叠加有降水与全部无降水的 KDE，额外计算/绘制降水概率约 50% 的边界线。 | 降水/无降水站点聚合统计 CSV | `fy4b_cloud_pair_kde_all_no_precip_50contour_*.png` |
| `plot_fy4b_cloud_pair_kde_equal_sample_50contour.py` | 按有降水样本数 1:1 随机无放回抽取无降水样本，叠加有降水/无降水 2x3 双云参数 KDE，并绘制 50% 后验概率等值线；同时保存 KDE 网格和抽样样本，便于二次处理。 | 降水/无降水站点聚合统计 CSV | `kde_equal_no_precip_50contour/` 下的组合 KDE 图、KDE 网格 NPZ、抽样 CSV 和 manifest CSV |
| `plot_fy4b_cloud_triplet_scatter.py` | 画 COT、CER、CTH、CTT 三变量组合的 3D 散点图，对比有降水和抽样无降水。 | 降水/无降水站点聚合统计 CSV | `fy4b_cloud_3d_triplet_scatter_*.png` |
| `plot_fy4b_cloud_triplet_rotation.py` | 生成三变量 3D 散点图的旋转 GIF，并输出多个固定视角面板图。 | 降水/无降水站点聚合统计 CSV | `fy4b_*_rotation.gif`、`fy4b_*_angles.png` |
| `plot_fy4b_precip_pair_scatter_by_season.py` | 按季节绘制降水样本云参数两两散点图，并输出季节样本量统计。 | 降水站点聚合统计 CSV | `fy4b_precip_pair_scatter_by_season_*.csv/png` |
| `cluster_fy4b_precip_pair_kmeans.py` | 文件名里是 kmeans，但实际使用 DBSCAN：对降水样本云参数两两组合聚类，计算不同变量对聚类结果的重叠/相似度，并提取 top2 聚类样本。 | 降水站点聚合统计 CSV | `fy4b_precip_pair_dbscan_*.csv/png`、`*_top2_similarity_*.csv`、`*_top2_details_*.csv` |
| `summarize_dbscan_top2_month_hour_frequency.py` | 对 DBSCAN top2 聚类明细按变量对、聚类编号、月份、小时统计频次，输出长表和宽表。 | `fy4b_precip_pair_dbscan_top2_details_*.csv` | `*_month_hour_frequency_long.csv`、`*_month_hour_frequency_wide.csv` |
| `plot_dbscan_top2_monthly_ridgeline.py` | 对 DBSCAN top2 聚类明细按月份-小时频次画 ridgeline 图。 | `fy4b_precip_pair_dbscan_top2_details_*.csv` | `*_month_hour_counts.csv`、`*_hourly_ridgeline.png` |
| `plot_dbscan_top2_union_ridgeline.py` | 合并所有变量对 top2 聚类样本，输出并集样本、云参数统计、降水强度比例和月-小时 ridgeline 图。 | `fy4b_precip_pair_dbscan_top2_details_*.csv` | `*_top2_union_samples.csv`、`*_top2_union_cloud_stats.csv`、`*_top2_union_precip_intensity_ratios.csv`、`*_top2_union_hourly_ridgeline.png` |
| `plot_station_precip_intensity_maps.py` | 将站点降水强度频率、海拔分布、海拔-降水相关性和不同强度空间分布画成地图/统计图。 | 降水站点聚合统计 CSV、站点元数据 CSV | `station_precip_intensity_frequency_*.csv/png`、`station_altitude_distribution_map.png`、`station_altitude_precip_intensity_correlation_*.csv/png`、`station_altitude_precip_intensity_distribution_maps_*.png` |

## 常用参数约定

- `--input-csv`：单个输入 CSV。
- `--precip-csv`：有降水样本 CSV。
- `--no-precip-csv`：无降水样本 CSV。
- `--output-dir` / `--output-csv`：输出目录或输出文件。
- `--min-file-count`：同一日期、小时、站点至少需要多少个 FY4B 文件参与聚合，很多分析默认是 `3`。
- 在 `summarize_fy4b_station_cloud_stats.py --precip-mode zero` 中，`--min-file-count` 默认也是 `3`，用于要求无降水样本满足至少 3 个暖云时次。
- `--stat`：云参数使用 `mean` 还是 `max`，很多脚本会分别处理或允许指定。
- `--precip-mode positive/zero`：在 `summarize_fy4b_station_cloud_stats.py` 中选择正降水或零降水样本。

## 变量与单位

- `COT`：Cloud optical thickness，云光学厚度。
- `CER`：Cloud effective radius，云有效半径。
- `CTH`：Cloud top height，代码中常把米转换为千米绘图。
- `CTT`：Cloud top temperature，代码中常把 K 转换为摄氏度绘图。
- `precipitation`：站点小时降水量，单位通常按输入站点资料为 mm。
- `file_count`：同一日期、小时、站点参与聚合的 FY4B 文件数。

## 目录与结果文件

- `outputs/`：当前目录下已有的结果文件目录，主要是 CSV 和 PNG。
- `__pycache__/`：Python 缓存目录，不应纳入版本管理。
- 多数脚本默认输出到 `H:\YeWu\Zhou\guangzhou\output` 或 `H:\YeWu\Zhou\guangzhou`，运行前如果要把结果放到本仓库，建议显式传入 `--output-dir H:\YeWu\Zhou\code\satellite\outputs`。

## 给后续 Codex 对话的快速查找关键词

- “FY4B 读取、暖云筛选、降水匹配”：`read_FY4B.py`
- “站点降水 txt 转 nc”：`read_station_precipitation.py`
- “只保留有降水样本”：`filter_fy4b_positive_precip.py`
- “按站点小时聚合 mean/max”：`summarize_fy4b_station_cloud_stats.py`
- “云参数分箱比例”：`analyze_fy4b_precip_bins.py`、`compare_fy4b_precip_no_precip_bins.py`
- “降水强度比例”：`plot_fy4b_cloud_bins_precip_intensity_ratios.py`、`plot_fy4b_cot_cer_joint_intensity_ratios.py`
- “降水概率热力图”：`plot_fy4b_precip_probability_heatmaps.py`
- “云参数散点/KDE/3D”：`plot_fy4b_cloud_pair_scatter.py`、`plot_fy4b_cloud_pair_kde.py`、`plot_fy4b_cloud_triplet_scatter.py`
- “DBSCAN 聚类/top2/月小时”：`cluster_fy4b_precip_pair_kmeans.py`、`summarize_dbscan_top2_month_hour_frequency.py`、`plot_dbscan_top2_*`
- “站点地图、海拔、空间分布”：`plot_station_precip_intensity_maps.py`

import pandas as pd
import xarray as xr
import netCDF4

# ========= 1. 读取 txt =========
txt_file = r"H:\YeWu\Zhou\guangzhou\data_20250201000000-20260201000000.txt"   # 你的 txt 文件
nc_file = r"H:\YeWu\Zhou\guangzhou\rain_1h.nc"  # 输出 nc 文件

cols = [
    "station_id",     # 区站号(字符)
    "station_name",   # 站名
    "county_name",    # 区县名
    "year",
    "month",
    "day",
    "hour",
    "precip_1h"       # 过去1小时降水量
]

# ---------- 1) 先按行读取，剔除重复表头 ----------
valid_lines = []
with open(txt_file, "r", encoding="gbk") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue

        # 跳过重复出现的说明/表头行
        if "区站号(字符)" in s and "过去1小时降水量" in s:
            continue

        valid_lines.append(s)

# 把清洗后的文本交给 pandas
from io import StringIO
df = pd.read_csv(
    StringIO("\n".join(valid_lines)),
    sep=r"\s+",
    header=None,
    names=cols,
    dtype={"station_id": str}
)

# ---------- 2) 再做一次稳妥清洗 ----------
# 把数值列转成数值，无法转换的会变成 NaN
num_cols = ["year", "month", "day", "hour", "precip_1h"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 删除无效行
df = df.dropna(subset=["year", "month", "day", "hour", "precip_1h"]).copy()

# 整型化时间列
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)
df["day"] = df["day"].astype(int)
df["hour"] = df["hour"].astype(int)

# station_id 再保险过滤一次
df = df[df["station_id"] != "区站号(字符)"].copy()

# ---------- 3) 生成时间坐标 ----------
df["time"] = pd.to_datetime(
    dict(
        year=df["year"],
        month=df["month"],
        day=df["day"],
        hour=df["hour"]
    ),
    errors="coerce"
)

df = df.dropna(subset=["time"]).copy()

# ---------- 4) 站点元信息 ----------
station_meta = (
    df[["station_id", "station_name", "county_name"]]
    .drop_duplicates(subset=["station_id"])
    .set_index("station_id")
)

# ---------- 5) 转成 time × station ----------
pivot = df.pivot_table(
    index="time",
    columns="station_id",
    values="precip_1h",
    aggfunc="first"
).sort_index()

station_meta = station_meta.loc[pivot.columns]

# ---------- 6) 构建 xarray Dataset ----------
ds = xr.Dataset(
    data_vars={
        "precip_1h": (("time", "station"), pivot.values)
    },
    coords={
        "time": pivot.index.values,
        "station": pivot.columns.values,
        "station_name": ("station", station_meta["station_name"].values),
        "county_name": ("station", station_meta["county_name"].values),
    }
)

# ---------- 7) 属性 ----------
ds["precip_1h"].attrs = {
    "long_name": "past_1_hour_precipitation",
    "description": "过去1小时降水量",
    "units": "mm"
}
ds["time"].attrs = {"standard_name": "time"}
ds["station"].attrs = {"long_name": "station_id"}

ds.attrs = {
    "title": "Hourly precipitation dataset",
    "source": "Converted from txt file after removing repeated header lines",
    "Conventions": "CF-1.8"
}

# ---------- 8) 保存 ----------
encoding = {
    "precip_1h": {
        "zlib": True,
        "complevel": 4,
        "_FillValue": -9999.0
    }
}

ds.to_netcdf(nc_file, format="NETCDF4", encoding=encoding)

print(f"已保存为: {nc_file}")
print(ds)
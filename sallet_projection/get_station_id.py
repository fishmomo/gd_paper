# -*- coding: utf-8 -*-
"""
shp文件：H:\我的业务\CM\广东省\shapefile\广东省.shp
降水文件：D:\CPAS_个例数据\20250504个例\Ground\rain\rh\202505031500.001
"""

# -*- coding: utf-8 -*-
"""
功能：
1. 读取你提供格式的 txt 文件
2. 读取 shp 文件（不使用 geopandas）
3. 判断每个点是否在 shp 边界内（含边界）
4. 输出：
   - 所有 station 信息（增加 in_shape 字段）
   - 所有 station 的 CSV / JSON / XML / KML
   - 单独输出 shp 内的 station（CSV / KML）
5. KML 输出基于你提供的 Google Earth 模板生成，尽量保持模板样式一致

安装依赖：
pip install pandas shapely pyshp pyproj
"""

from pathlib import Path
from copy import deepcopy
import pandas as pd
import shapefile  # pyshp
from shapely.geometry import Point, shape
from shapely.ops import unary_union, transform
from pyproj import CRS, Transformer
import xml.etree.ElementTree as ET


KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
ATOM_NS = "http://www.w3.org/2005/Atom"

NS = {
    "kml": KML_NS,
    "gx": GX_NS,
    "atom": ATOM_NS
}

ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)
ET.register_namespace("kml", KML_NS)
ET.register_namespace("atom", ATOM_NS)


def parse_txt(txt_path: str) -> pd.DataFrame:
    """
    读取如下格式：
    第1行：说明行，例如 CPAS CJ
    第2行：表头，例如 Station_ID_C Lon Lat Alti PRE_1h
    第3行开始：数据
    """
    df = pd.read_csv(
        txt_path,
        sep=r"\s+",
        engine="python",
        skiprows=1
    )

    # 统一字段名
    df = df.rename(columns={
        "Station_ID_C": "station",
        "Lon": "lon",
        "Lat": "lat",
        "Alti": "Alti"
    })

    required_cols = ["station", "lon", "lat", "Alti"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"txt 解析失败，缺少字段: {missing}\n"
            f"当前识别到的列名为: {list(df.columns)}"
        )

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["Alti"] = pd.to_numeric(df["Alti"], errors="coerce")

    return df.copy()


def read_shp_as_union(shp_path: str):
    """
    读取 shp 中所有几何并合并为一个整体，后续统一判断点是否在面内。
    """
    sf = shapefile.Reader(shp_path)
    geoms = [shape(s.__geo_interface__) for s in sf.shapes()]

    if not geoms:
        raise ValueError("shp 文件中没有有效几何对象。")

    return unary_union(geoms)


def read_shp_crs(shp_path: str):
    """
    读取同名 .prj 文件中的坐标系。
    没有 .prj 时返回 None。
    """
    prj_path = Path(shp_path).with_suffix(".prj")
    if not prj_path.exists():
        return None

    wkt = prj_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not wkt:
        return None

    try:
        return CRS.from_wkt(wkt)
    except Exception:
        return None


def save_all_to_json(df: pd.DataFrame, json_path: Path):
    """
    输出全部站点为 JSON。
    """
    df.to_json(
        json_path,
        orient="records",
        force_ascii=False,
        indent=2
    )


def save_all_to_xml(df: pd.DataFrame, xml_path: Path):
    """
    输出全部站点为 XML。
    """
    root = ET.Element("stations")

    for _, row in df.iterrows():
        record_elem = ET.SubElement(root, "station_record")

        for col in df.columns:
            child = ET.SubElement(record_elem, str(col))
            value = row[col]

            if pd.isna(value):
                child.text = ""
            elif isinstance(value, bool):
                child.text = str(value).lower()
            else:
                child.text = str(value)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def format_value(value):
    """
    输出前做一点格式清洗，避免整数 station 被写成 50739.0 这种样子。
    """
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def load_template_info(template_kml_path: str):
    """
    从模板 KML 中提取：
    1. Document 根节点
    2. 第一个 Placemark 中的样式引用
    3. LookAt 默认参数
    4. Point 的 altitudeMode

    后面输出新 KML 时，会保留模板中的样式定义，
    并用这些默认参数构造新的 Placemark。
    """
    tree = ET.parse(template_kml_path)
    root = tree.getroot()

    document = root.find("kml:Document", NS)
    if document is None:
        raise ValueError("模板 KML 中未找到 Document 节点。")

    first_pm = document.find("kml:Placemark", NS)

    style_url_text = None
    lookat_defaults = {
        "heading": "0",
        "tilt": "0",
        "fovy": "35",
        "range": "5000",
        "altitudeMode": "relativeToGround",
        "altitude": "0"
    }
    point_altitude_mode = "relativeToGround"

    if first_pm is not None:
        style_url = first_pm.find("kml:styleUrl", NS)
        if style_url is not None and style_url.text:
            style_url_text = style_url.text.strip()

        lookat = first_pm.find("kml:LookAt", NS)
        if lookat is not None:
            heading = lookat.find("kml:heading", NS)
            tilt = lookat.find("kml:tilt", NS)
            gx_fovy = lookat.find("gx:fovy", NS)
            range_elem = lookat.find("kml:range", NS)
            altitude_mode = lookat.find("kml:altitudeMode", NS)
            altitude = lookat.find("kml:altitude", NS)

            if heading is not None and heading.text:
                lookat_defaults["heading"] = heading.text.strip()
            if tilt is not None and tilt.text:
                lookat_defaults["tilt"] = tilt.text.strip()
            if gx_fovy is not None and gx_fovy.text:
                lookat_defaults["fovy"] = gx_fovy.text.strip()
            if range_elem is not None and range_elem.text:
                lookat_defaults["range"] = range_elem.text.strip()
            if altitude_mode is not None and altitude_mode.text:
                lookat_defaults["altitudeMode"] = altitude_mode.text.strip()
            if altitude is not None and altitude.text:
                lookat_defaults["altitude"] = altitude.text.strip()

        point = first_pm.find("kml:Point", NS)
        if point is not None:
            p_alt_mode = point.find("kml:altitudeMode", NS)
            if p_alt_mode is not None and p_alt_mode.text:
                point_altitude_mode = p_alt_mode.text.strip()

    return style_url_text, lookat_defaults, point_altitude_mode


def remove_existing_placemarks(document_elem):
    """
    保留模板里的样式、StyleMap 等定义，只删除原有 Placemark。
    """
    for child in list(document_elem):
        if child.tag == f"{{{KML_NS}}}Placemark":
            document_elem.remove(child)


def build_extended_data(placemark, row):
    """
    在 KML 中附带属性，Google Earth 中也能保留字段信息。
    """
    extended = ET.SubElement(placemark, f"{{{KML_NS}}}ExtendedData")
    for field in ["station", "lon", "lat", "Alti", "in_shape"]:
        data_elem = ET.SubElement(extended, f"{{{KML_NS}}}Data", name=field)
        value_elem = ET.SubElement(data_elem, f"{{{KML_NS}}}value")
        value_elem.text = format_value(row.get(field, ""))


def add_text_element(parent, tag, text, namespace=KML_NS):
    elem = ET.SubElement(parent, f"{{{namespace}}}{tag}")
    elem.text = text
    return elem


def save_to_kml_by_template(df: pd.DataFrame, template_kml_path: str, output_kml_path: Path, document_name: str):
    """
    基于模板 KML 输出新文件：
    - 保留模板中的样式定义
    - 删除模板中的原始 Placemark
    - 用 station 数据重新生成 Placemark
    """
    tree = ET.parse(template_kml_path)
    root = tree.getroot()

    document = root.find("kml:Document", NS)
    if document is None:
        raise ValueError("模板 KML 中未找到 Document 节点。")

    style_url_text, lookat_defaults, point_altitude_mode = load_template_info(template_kml_path)

    # 修改文档名
    doc_name = document.find("kml:name", NS)
    if doc_name is None:
        doc_name = ET.Element(f"{{{KML_NS}}}name")
        document.insert(0, doc_name)
    doc_name.text = document_name

    # 删除模板中已有的 Placemark，仅保留样式等定义
    remove_existing_placemarks(document)

    # 批量写入新点
    for idx, row in df.iterrows():
        lon = row.get("lon")
        lat = row.get("lat")
        alti = row.get("Alti")

        if pd.isna(lon) or pd.isna(lat):
            continue

        placemark = ET.SubElement(document, f"{{{KML_NS}}}Placemark", id=f"station_{idx + 1}")

        name_elem = ET.SubElement(placemark, f"{{{KML_NS}}}name")
        name_elem.text = format_value(row.get("station", ""))

        # 模板里有 LookAt，这里按模板方式继续生成
        lookat = ET.SubElement(placemark, f"{{{KML_NS}}}LookAt")
        add_text_element(lookat, "longitude", format_value(lon))
        add_text_element(lookat, "latitude", format_value(lat))
        add_text_element(lookat, "altitude", lookat_defaults["altitude"])
        add_text_element(lookat, "heading", lookat_defaults["heading"])
        add_text_element(lookat, "tilt", lookat_defaults["tilt"])
        add_text_element(lookat, "fovy", lookat_defaults["fovy"], namespace=GX_NS)
        add_text_element(lookat, "range", lookat_defaults["range"])
        add_text_element(lookat, "altitudeMode", lookat_defaults["altitudeMode"])

        if style_url_text:
            style_url = ET.SubElement(placemark, f"{{{KML_NS}}}styleUrl")
            style_url.text = style_url_text

        # 模板虽然隐藏了 Balloon，但字段依然保留，便于后续查看或再加工
        build_extended_data(placemark, row)

        point = ET.SubElement(placemark, f"{{{KML_NS}}}Point")
        add_text_element(point, "altitudeMode", point_altitude_mode)

        coords = ET.SubElement(point, f"{{{KML_NS}}}coordinates")
        coords.text = f"{format_value(lon)},{format_value(lat)},{format_value(alti) if format_value(alti) != '' else '0'}"

    ET.indent(tree, space="\t", level=0)
    tree.write(output_kml_path, encoding="utf-8", xml_declaration=True)


def main():
    print("\n========== station 落区筛选（模板 KML 输出版）==========\n")

    shp_path = input("请输入 shp 文件路径（例如 D:/data/boundary.shp）：").strip().strip('"')
    txt_path = input("请输入 txt 文件路径（例如 D:/data/stations.txt）：").strip().strip('"')
    template_kml_path = input("请输入 KML 模板文件路径（例如 D:/data/test.kml）：").strip().strip('"')

    if not Path(shp_path).exists():
        raise FileNotFoundError(f"未找到 shp 文件：{shp_path}")

    if not Path(txt_path).exists():
        raise FileNotFoundError(f"未找到 txt 文件：{txt_path}")

    if not Path(template_kml_path).exists():
        raise FileNotFoundError(f"未找到 KML 模板文件：{template_kml_path}")

    print("正在读取 shp 文件...")
    shp_geom = read_shp_as_union(shp_path)

    print("正在读取 shp 坐标系信息...")
    shp_crs = read_shp_crs(shp_path)

    print("正在读取 txt 文件...")
    df_all = parse_txt(txt_path)

    # txt 的 lon/lat 默认按 WGS84 经纬度处理
    txt_crs = CRS.from_epsg(4326)

    transformer = None
    if shp_crs is not None and shp_crs != txt_crs:
        transformer = Transformer.from_crs(txt_crs, shp_crs, always_xy=True)
        print("检测到 shp 坐标系与经纬度不同，已自动进行坐标转换。")
    elif shp_crs is None:
        print("警告：未找到 .prj 文件，默认认为 shp 与 txt 使用同一坐标系。")

    print("正在判断点是否在 shp 边界内（含边界）...")

    in_shape_list = []

    for _, row in df_all.iterrows():
        lon = row["lon"]
        lat = row["lat"]

        if pd.isna(lon) or pd.isna(lat):
            in_shape_list.append(False)
            continue

        pt = Point(lon, lat)

        # 如果 shp 不是经纬度坐标系，则把点投影到 shp 坐标系后再判断
        if transformer is not None:
            pt = transform(transformer.transform, pt)

        # covers: 边界内 + 边界上都记为 True
        in_shape_list.append(shp_geom.covers(pt))

    # 全部站点
    df_all["in_shape"] = in_shape_list

    # shp 内站点
    df_in_shape = df_all[df_all["in_shape"]].copy()

    txt_file = Path('H:\\邢台观测站\\CWR_project\\guangzhou\sallet_projection\\')

    output_all_csv = txt_file.with_name("all_stations_with_flag.csv")
    output_all_json = txt_file.with_name("all_stations_with_flag.json")
    output_all_xml = txt_file.with_name("all_stations_with_flag.xml")
    output_all_kml = txt_file.with_name("all_stations_with_flag.kml")

    output_in_csv = txt_file.with_name("stations_in_shape.csv")
    output_in_kml = txt_file.with_name("stations_in_shape.kml")

    # 输出全部站点
    df_all.to_csv(output_all_csv, index=False, encoding="utf-8-sig")
    save_all_to_json(df_all, output_all_json)
    save_all_to_xml(df_all, output_all_xml)
    save_to_kml_by_template(
        df_all[["station", "lon", "lat", "Alti", "in_shape"]],
        template_kml_path,
        output_all_kml,
        "All Stations"
    )

    # 输出 shp 内站点
    df_in_shape[["station", "lon", "lat", "Alti"]].to_csv(
        output_in_csv,
        index=False,
        encoding="utf-8-sig"
    )
    save_to_kml_by_template(
        df_in_shape[["station", "lon", "lat", "Alti", "in_shape"]],
        template_kml_path,
        output_in_kml,
        "Stations In Shape"
    )

    print("\n========== 处理完成 ==========")
    print(f"总站点数：{len(df_all)}")
    print(f"shp 内站点数：{len(df_in_shape)}")
    print(f"全部站点 CSV：{output_all_csv}")
    print(f"全部站点 JSON：{output_all_json}")
    print(f"全部站点 XML：{output_all_xml}")
    print(f"全部站点 KML：{output_all_kml}")
    print(f"shp 内站点 CSV：{output_in_csv}")
    print(f"shp 内站点 KML：{output_in_kml}")

    print("\n全部站点预览：")
    print(df_all.head(20).to_string(index=False))

    print("\nshp 内站点预览：")
    if not df_in_shape.empty:
        print(df_in_shape[["station", "lon", "lat", "Alti"]].head(20).to_string(index=False))
    else:
        print("没有站点落在该 shp 边界内。")


if __name__ == "__main__":
    main()
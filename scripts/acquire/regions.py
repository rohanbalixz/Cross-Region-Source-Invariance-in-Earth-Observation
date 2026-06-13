"""Region and city bounding-box registry.

Single source of truth for the four test regions defined in
`docs/regions.md`. All acquisition scripts import REGIONS from here so that a
bbox change in one place propagates everywhere.

Bounding boxes are EPSG:4326 (lon_min, lat_min, lon_max, lat_max).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class City:
    name: str
    region: str
    bbox: Tuple[float, float, float, float]


CITIES: List[City] = [
    # ~1° squares (≈110×110 km at equator), centred on each metro core.
    # Scaled to ~11 cities/region (44 total) so cluster-bootstrap CIs over
    # cities are meaningful and cross-region generalization is defensible;
    # this exceeds published cross-city land-cover benchmarks (10-23 cities).

    # --- South Asia ---
    City("mumbai",    "south_asia", (72.400, 18.575, 73.400, 19.575)),
    City("delhi",     "south_asia", (76.625, 28.150, 77.625, 29.150)),
    City("dhaka",     "south_asia", (89.925, 23.300, 90.925, 24.300)),
    City("kolkata",   "south_asia", (87.860, 22.070, 88.860, 23.070)),
    City("bengaluru", "south_asia", (77.090, 12.470, 78.090, 13.470)),
    City("chennai",   "south_asia", (79.770, 12.580, 80.770, 13.580)),
    City("hyderabad", "south_asia", (77.970, 16.880, 78.970, 17.880)),
    City("karachi",   "south_asia", (66.510, 24.360, 67.510, 25.360)),
    City("lahore",    "south_asia", (73.840, 31.050, 74.840, 32.050)),
    City("ahmedabad", "south_asia", (72.070, 22.520, 73.070, 23.520)),
    City("pune",      "south_asia", (73.360, 18.020, 74.360, 19.020)),

    # --- Sub-Saharan Africa ---
    City("lagos",         "ssa", ( 2.950,  6.025,  3.950,  7.025)),
    City("nairobi",       "ssa", (36.350, -1.775, 37.350, -0.775)),
    City("kinshasa",      "ssa", (14.825, -4.875, 15.825, -3.875)),
    City("accra",         "ssa", (-0.690,  5.100,  0.310,  6.100)),
    City("abidjan",       "ssa", (-4.520,  4.850, -3.520,  5.850)),
    City("dar_es_salaam", "ssa", (38.780, -7.320, 39.780, -6.320)),
    City("addis_ababa",   "ssa", (38.240,  8.530, 39.240,  9.530)),
    City("kampala",       "ssa", (32.080, -0.150, 33.080,  0.850)),
    City("luanda",        "ssa", (12.730, -9.340, 13.730, -8.340)),
    City("johannesburg",  "ssa", (27.550, -26.700, 28.550, -25.700)),
    City("dakar",         "ssa", (-17.950, 14.220, -16.950, 15.220)),

    # --- East Asia ---
    City("pearl_river_delta", "east_asia", (113.250, 22.300, 114.250, 23.300)),
    City("shanghai",          "east_asia", (121.025, 30.725, 122.025, 31.725)),
    City("beijing",           "east_asia", (115.910, 39.400, 116.910, 40.400)),
    City("tokyo",             "east_asia", (139.190, 35.190, 140.190, 36.190)),
    City("seoul",             "east_asia", (126.480, 37.070, 127.480, 38.070)),
    City("osaka",             "east_asia", (135.000, 34.190, 136.000, 35.190)),
    City("chengdu",           "east_asia", (103.570, 30.070, 104.570, 31.070)),
    City("wuhan",             "east_asia", (113.800, 30.090, 114.800, 31.090)),
    City("chongqing",         "east_asia", (106.050, 29.060, 107.050, 30.060)),
    City("taipei",            "east_asia", (121.060, 24.530, 122.060, 25.530)),
    City("busan",             "east_asia", (128.580, 34.680, 129.580, 35.680)),

    # --- Andes / Andean Latin America ---
    City("bogota",     "andes", (-74.575,  4.175, -73.575,  5.175)),
    City("lima",       "andes", (-77.525,-12.550, -76.525,-11.550)),
    City("quito",      "andes", (-79.000, -0.675, -78.000,  0.325)),
    City("la_paz",     "andes", (-68.650,-17.000, -67.650,-16.000)),
    City("medellin",   "andes", (-76.060,  5.750, -75.060,  6.750)),
    City("arequipa",   "andes", (-72.040,-16.910, -71.040,-15.910)),
    City("santiago",   "andes", (-71.150,-33.950, -70.150,-32.950)),
    City("cusco",      "andes", (-72.470,-14.030, -71.470,-13.030)),
    City("cochabamba", "andes", (-66.660,-17.890, -65.660,-16.890)),
    City("cali",       "andes", (-77.020,  2.920, -76.020,  3.920)),
    City("caracas",    "andes", (-67.400,  9.990, -66.400, 10.990)),

    # =====================================================================
    # Region scale-up (4 -> 8 regions). 1deg bboxes centred on
    # each metro core. Chosen to SPAN the covariate space, with Oceania as a
    # deliberately well-represented CONTROL region (thesis predicts it should
    # transfer well, making the cross-region claim falsifiable).
    # =====================================================================

    # --- MENA / arid urbanism ---
    City("cairo",       "mena", (30.735, 29.545, 31.735, 30.545)),
    City("istanbul",    "mena", (28.480, 40.510, 29.480, 41.510)),
    City("tehran",      "mena", (50.890, 35.190, 51.890, 36.190)),
    City("casablanca",  "mena", (-8.090, 33.070, -7.090, 34.070)),
    City("riyadh",      "mena", (46.220, 24.210, 47.220, 25.210)),
    City("baghdad",     "mena", (43.860, 32.810, 44.860, 33.810)),
    City("amman",       "mena", (35.410, 31.450, 36.410, 32.450)),
    City("tunis",       "mena", (9.680, 36.310, 10.680, 37.310)),
    City("algiers",     "mena", (2.560, 36.250, 3.560, 37.250)),
    City("dubai",       "mena", (54.770, 24.700, 55.770, 25.700)),
    City("beirut",      "mena", (35.000, 33.390, 36.000, 34.390)),

    # --- Southeast Asia / tropical deltaic ---
    City("jakarta",     "sea", (106.350, -6.710, 107.350, -5.710)),
    City("manila",      "sea", (120.480, 14.100, 121.480, 15.100)),
    City("bangkok",     "sea", (100.000, 13.260, 101.000, 14.260)),
    City("ho_chi_minh", "sea", (106.160, 10.320, 107.160, 11.320)),
    City("kuala_lumpur","sea", (101.190, 2.640, 102.190, 3.640)),
    City("surabaya",    "sea", (112.250, -7.760, 113.250, -6.760)),
    City("hanoi",       "sea", (105.330, 20.530, 106.330, 21.530)),
    City("yangon",      "sea", (95.700, 16.370, 96.700, 17.370)),
    City("phnom_penh",  "sea", (104.420, 11.060, 105.420, 12.060)),
    City("singapore",   "sea", (103.320, 0.850, 104.320, 1.850)),
    City("bandung",     "sea", (107.110, -7.410, 108.110, -6.410)),

    # --- Eastern Europe / Central Asia / planned post-socialist ---
    City("moscow",        "eeca", (37.120, 55.250, 38.120, 56.250)),
    City("kyiv",          "eeca", (30.020, 49.950, 31.020, 50.950)),
    City("warsaw",        "eeca", (20.510, 51.730, 21.510, 52.730)),
    City("bucharest",     "eeca", (25.600, 43.930, 26.600, 44.930)),
    City("tashkent",      "eeca", (68.740, 40.800, 69.740, 41.800)),
    City("almaty",        "eeca", (76.450, 42.760, 77.450, 43.760)),
    City("baku",          "eeca", (49.370, 39.910, 50.370, 40.910)),
    City("minsk",         "eeca", (27.060, 53.400, 28.060, 54.400)),
    City("kharkiv",       "eeca", (35.730, 49.490, 36.730, 50.490)),
    City("novosibirsk",   "eeca", (82.420, 54.530, 83.420, 55.530)),
    City("yekaterinburg", "eeca", (60.110, 56.340, 61.110, 57.340)),

    # --- Oceania / well-represented CONTROL (predicted easy) ---
    City("sydney",       "oceania", (150.710, -34.370, 151.710, -33.370)),
    City("melbourne",    "oceania", (144.460, -38.310, 145.460, -37.310)),
    City("brisbane",     "oceania", (152.530, -27.970, 153.530, -26.970)),
    City("perth",        "oceania", (115.360, -32.450, 116.360, -31.450)),
    City("auckland",     "oceania", (174.260, -37.350, 175.260, -36.350)),
    City("adelaide",     "oceania", (138.100, -35.430, 139.100, -34.430)),
    City("gold_coast",   "oceania", (152.930, -28.520, 153.930, -27.520)),
    City("canberra",     "oceania", (148.630, -35.780, 149.630, -34.780)),
    City("wellington",   "oceania", (174.280, -41.790, 175.280, -40.790)),
    City("newcastle_au", "oceania", (151.280, -33.430, 152.280, -32.430)),
    City("christchurch", "oceania", (172.140, -44.030, 173.140, -43.030)),

    # =====================================================================
    # Region scale-up #2 (8 -> 12 regions) to tighten region-level
    # statistics. Western Europe and Canada add developed-region controls
    # (testing the developed-region-hardest pattern beyond Oceania);
    # lowland Latin America and Central America/Caribbean add tropical and
    # transitional settlement regimes distinct from the Andes.
    # =====================================================================

    # --- Western Europe (developed control) ---
    City("london",     "weur", (-0.630, 51.010, 0.370, 52.010)),
    City("paris",      "weur", (1.850, 48.360, 2.850, 49.360)),
    City("berlin",     "weur", (12.900, 52.020, 13.900, 53.020)),
    City("madrid",     "weur", (-4.200, 39.920, -3.200, 40.920)),
    City("rome",       "weur", (12.000, 41.400, 13.000, 42.400)),
    City("amsterdam",  "weur", (4.400, 51.870, 5.400, 52.870)),
    City("vienna",     "weur", (15.870, 47.710, 16.870, 48.710)),
    City("barcelona",  "weur", (1.670, 40.890, 2.670, 41.890)),
    City("milan",      "weur", (8.690, 44.960, 9.690, 45.960)),
    City("munich",     "weur", (11.080, 47.640, 12.080, 48.640)),
    City("lisbon",     "weur", (-9.640, 38.220, -8.640, 39.220)),

    # --- Lowland Latin America (non-Andean) ---
    City("sao_paulo",       "latam", (-47.130, -24.050, -46.130, -23.050)),
    City("rio_de_janeiro",  "latam", (-43.700, -23.410, -42.700, -22.410)),
    City("buenos_aires",    "latam", (-58.880, -35.100, -57.880, -34.100)),
    City("brasilia",        "latam", (-48.430, -16.280, -47.430, -15.280)),
    City("montevideo",      "latam", (-56.660, -35.400, -55.660, -34.400)),
    City("asuncion",        "latam", (-58.080, -25.780, -57.080, -24.780)),
    City("belo_horizonte",  "latam", (-44.430, -20.420, -43.430, -19.420)),
    City("curitiba",        "latam", (-49.770, -25.930, -48.770, -24.930)),
    City("porto_alegre",    "latam", (-51.730, -30.530, -50.730, -29.530)),
    City("salvador_br",     "latam", (-39.000, -13.470, -38.000, -12.470)),
    City("recife",          "latam", (-35.380, -8.550, -34.380, -7.550)),

    # --- Central America & Caribbean ---
    City("mexico_city",     "camcar", (-99.630, 18.930, -98.630, 19.930)),
    City("guatemala_city",  "camcar", (-91.010, 14.130, -90.010, 15.130)),
    City("havana",          "camcar", (-82.880, 22.610, -81.880, 23.610)),
    City("san_salvador",    "camcar", (-89.690, 13.190, -88.690, 14.190)),
    City("panama_city",     "camcar", (-80.020, 8.480, -79.020, 9.480)),
    City("tegucigalpa",     "camcar", (-87.700, 13.570, -86.700, 14.570)),
    City("managua",         "camcar", (-86.750, 11.630, -85.750, 12.630)),
    City("santo_domingo",   "camcar", (-70.430, 17.990, -69.430, 18.990)),
    City("san_jose_cr",     "camcar", (-84.590, 9.430, -83.590, 10.430)),
    City("kingston",        "camcar", (-77.290, 17.470, -76.290, 18.470)),
    City("port_au_prince",  "camcar", (-72.830, 18.090, -71.830, 19.090)),

    # --- Canada (cold developed control) ---
    City("toronto",      "canada", (-79.880, 43.150, -78.880, 44.150)),
    City("montreal",     "canada", (-74.070, 45.000, -73.070, 46.000)),
    City("vancouver",    "canada", (-123.620, 48.780, -122.620, 49.780)),
    City("calgary",      "canada", (-114.570, 50.550, -113.570, 51.550)),
    City("ottawa",       "canada", (-76.200, 44.920, -75.200, 45.920)),
    City("edmonton",     "canada", (-113.990, 53.050, -112.990, 54.050)),
    City("winnipeg",     "canada", (-97.640, 49.400, -96.640, 50.400)),
    City("quebec_city",  "canada", (-71.710, 46.310, -70.710, 47.310)),
    City("hamilton_ca",  "canada", (-80.370, 42.760, -79.370, 43.760)),
    City("halifax",      "canada", (-64.070, 44.150, -63.070, 45.150)),
    City("victoria",     "canada", (-123.870, 47.930, -122.870, 48.930)),

    # ===== n->20 extension (added 2026-06; bboxes ~1 deg on metro core; the GHSL
    # built-up output is the validity check -- ocean/mis-placed boxes return ~0) =====
    # --- Nordic (low-change, developed) ---
    City("stockholm",  "nordic", (17.570, 58.830, 18.570, 59.830)),
    City("oslo",       "nordic", (10.250, 59.410, 11.250, 60.410)),
    City("helsinki",   "nordic", (24.440, 59.670, 25.440, 60.670)),
    City("copenhagen", "nordic", (12.070, 55.180, 13.070, 56.180)),
    City("gothenburg", "nordic", (11.470, 57.210, 12.470, 58.210)),
    City("aarhus",     "nordic", ( 9.710, 55.660, 10.710, 56.660)),
    City("bergen",     "nordic", ( 4.820, 59.890,  5.820, 60.890)),
    City("stavanger",  "nordic", ( 5.230, 58.470,  6.230, 59.470)),
    City("tampere",    "nordic", (23.260, 61.000, 24.260, 62.000)),
    City("trondheim",  "nordic", ( 9.890, 62.930, 10.890, 63.930)),
    City("reykjavik",  "nordic", (-22.440, 63.650, -21.440, 64.650)),

    # --- Central Europe (low-mid change) ---
    City("prague",     "central_europe", (13.940, 49.580, 14.940, 50.580)),
    City("budapest",   "central_europe", (18.540, 47.000, 19.540, 48.000)),
    City("krakow",     "central_europe", (19.440, 49.560, 20.440, 50.560)),
    City("hamburg",    "central_europe", ( 9.490, 53.050, 10.490, 54.050)),
    City("frankfurt",  "central_europe", ( 8.180, 49.610,  9.180, 50.610)),
    City("zurich",     "central_europe", ( 8.040, 46.870,  9.040, 47.870)),
    City("brussels",   "central_europe", ( 3.850, 50.350,  4.850, 51.350)),
    City("cologne",    "central_europe", ( 6.460, 50.440,  7.460, 51.440)),
    City("stuttgart",  "central_europe", ( 8.680, 48.280,  9.680, 49.280)),
    City("wroclaw",    "central_europe", (16.540, 50.610, 17.540, 51.610)),
    City("bratislava", "central_europe", (16.610, 47.650, 17.610, 48.650)),

    # --- China interior / tier-2 (high-change) ---
    City("xian",      "china_interior", (108.450, 33.770, 109.450, 34.770)),
    City("zhengzhou", "china_interior", (113.120, 34.250, 114.120, 35.250)),
    City("hangzhou",  "china_interior", (119.650, 29.770, 120.650, 30.770)),
    City("nanjing",   "china_interior", (118.300, 31.560, 119.300, 32.560)),
    City("tianjin",   "china_interior", (116.700, 38.630, 117.700, 39.630)),
    City("shenyang",  "china_interior", (122.930, 41.300, 123.930, 42.300)),
    City("qingdao",   "china_interior", (119.880, 35.570, 120.880, 36.570)),
    City("changsha",  "china_interior", (112.440, 27.730, 113.440, 28.730)),
    City("jinan",     "china_interior", (116.620, 36.150, 117.620, 37.150)),
    City("harbin",    "china_interior", (126.030, 45.300, 127.030, 46.300)),
    City("kunming",   "china_interior", (102.220, 24.540, 103.220, 25.540)),

    # --- South Asia tier-2 (high-change) ---
    City("surat",       "south_asia_2", (72.330, 20.670, 73.330, 21.670)),
    City("jaipur",      "south_asia_2", (75.290, 26.410, 76.290, 27.410)),
    City("kanpur",      "south_asia_2", (79.830, 25.950, 80.830, 26.950)),
    City("lucknow",     "south_asia_2", (80.450, 26.350, 81.450, 27.350)),
    City("nagpur",      "south_asia_2", (78.590, 20.650, 79.590, 21.650)),
    City("patna",       "south_asia_2", (84.640, 25.090, 85.640, 26.090)),
    City("indore",      "south_asia_2", (75.360, 22.220, 76.360, 23.220)),
    City("faisalabad",  "south_asia_2", (72.580, 30.920, 73.580, 31.920)),
    City("chittagong",  "south_asia_2", (91.280, 21.860, 92.280, 22.860)),
    City("colombo",     "south_asia_2", (79.360,  6.430, 80.360,  7.430)),
    City("kathmandu",   "south_asia_2", (84.820, 27.220, 85.820, 28.220)),

    # --- Southern Africa (mid-high change) ---
    City("cape_town",     "southern_africa", (17.920, -34.420, 18.920, -33.420)),
    City("durban",        "southern_africa", (30.520, -30.360, 31.520, -29.360)),
    City("pretoria",      "southern_africa", (27.690, -26.250, 28.690, -25.250)),
    City("harare",        "southern_africa", (30.550, -18.330, 31.550, -17.330)),
    City("lusaka",        "southern_africa", (27.780, -15.920, 28.780, -14.920)),
    City("maputo",        "southern_africa", (32.070, -26.470, 33.070, -25.470)),
    City("gaborone",      "southern_africa", (25.410, -25.150, 26.410, -24.150)),
    City("windhoek",      "southern_africa", (16.580, -23.060, 17.580, -22.060)),
    City("bulawayo",      "southern_africa", (28.080, -20.650, 29.080, -19.650)),
    City("lubumbashi",    "southern_africa", (26.980, -12.160, 27.980, -11.160)),
    City("antananarivo",  "southern_africa", (47.010, -19.380, 48.010, -18.380)),

    # --- Mediterranean basin (mid change) ---
    City("athens",        "mediterranean", (23.230, 37.480, 24.230, 38.480)),
    City("tel_aviv",      "mediterranean", (34.280, 31.580, 35.280, 32.580)),
    City("izmir",         "mediterranean", (26.640, 37.920, 27.640, 38.920)),
    City("ankara",        "mediterranean", (32.370, 39.430, 33.370, 40.430)),
    City("alexandria",    "mediterranean", (29.420, 30.700, 30.420, 31.700)),
    City("rabat",         "mediterranean", (-7.340, 33.520, -6.340, 34.520)),
    City("tripoli_ly",    "mediterranean", (12.690, 32.390, 13.690, 33.390)),
    City("valencia_es",   "mediterranean", (-0.880, 38.970,  0.120, 39.970)),
    City("jeddah",        "mediterranean", (38.690, 20.990, 39.690, 21.990)),
    City("aleppo",        "mediterranean", (36.660, 35.700, 37.660, 36.700)),
    City("thessaloniki",  "mediterranean", (22.440, 40.140, 23.440, 41.140)),

    # --- Brazil north/interior (mid-high change) ---
    City("fortaleza",    "brazil_north", (-39.020, -4.230, -38.020, -3.230)),
    City("manaus",       "brazil_north", (-60.520, -3.620, -59.520, -2.620)),
    City("belem",        "brazil_north", (-48.990, -1.960, -47.990, -0.960)),
    City("goiania",      "brazil_north", (-49.750, -17.190, -48.750, -16.190)),
    City("campinas",     "brazil_north", (-47.560, -23.410, -46.560, -22.410)),
    City("sao_luis",     "brazil_north", (-44.800, -3.030, -43.800, -2.030)),
    City("natal",        "brazil_north", (-35.710, -6.290, -34.710, -5.290)),
    City("teresina",     "brazil_north", (-43.300, -5.590, -42.300, -4.590)),
    City("cuiaba",       "brazil_north", (-56.600, -16.100, -55.600, -15.100)),
    City("maceio",       "brazil_north", (-36.240, -10.170, -35.240, -9.170)),
    City("joao_pessoa",  "brazil_north", (-35.360, -7.620, -34.360, -6.620)),

    # --- Japan/Korea tier-2 (low-change, developed) ---
    City("nagoya",    "japan_korea_2", (136.410, 34.680, 137.410, 35.680)),
    City("fukuoka",   "japan_korea_2", (129.900, 33.090, 130.900, 34.090)),
    City("sapporo",   "japan_korea_2", (140.850, 42.570, 141.850, 43.570)),
    City("sendai",    "japan_korea_2", (140.370, 37.770, 141.370, 38.770)),
    City("hiroshima", "japan_korea_2", (131.960, 33.890, 132.960, 34.890)),
    City("daegu",     "japan_korea_2", (128.100, 35.370, 129.100, 36.370)),
    City("gwangju",   "japan_korea_2", (126.350, 34.660, 127.350, 35.660)),
    City("daejeon",   "japan_korea_2", (126.880, 35.850, 127.880, 36.850)),
    City("kaohsiung", "japan_korea_2", (119.800, 22.130, 120.800, 23.130)),
    City("niigata",   "japan_korea_2", (138.540, 37.420, 139.540, 38.420)),
    City("kanazawa",  "japan_korea_2", (136.130, 36.090, 137.130, 37.090)),
]


REGIONS: Dict[str, List[City]] = {}
for c in CITIES:
    REGIONS.setdefault(c.region, []).append(c)


def city_by_name(name: str) -> City:
    for c in CITIES:
        if c.name == name:
            return c
    raise KeyError(f"unknown city {name!r}")


if __name__ == "__main__":
    for region, cities in REGIONS.items():
        print(f"{region}: {[c.name for c in cities]}")

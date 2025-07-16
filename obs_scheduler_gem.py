"""
リストされたその夜に観測すべき天域（天体）の観測順を決めるプログラム
ここまで望遠鏡指向解析に使う天域を対象としてきましたが、ここでは少し一般化して、リストされた複数の天体を１夜の間に観測する順番を決める問題として考えたい。
・天体も天域も与えられる座標は赤経、赤緯とする
・個々の天体は、その観測の重要性から重み付けがなされ、１から５の範囲で天体リストに付与される。重みが小さい場合は、その夜には観測されない場合も許容される。
・各天体には、その時刻（いつ？）において、西の地平線に没するまでの時間が計算され付与する
・各天体は、少しでも大気条件の良い高高度での観測が望まれる。
・各天体の観測時刻の間隔dは、望遠鏡移動時間（移動距離sに比例）a*s、センターリング時間（固定とする）b、露出時間（実際には天体と観測の種類に依存するが、ここでは固定）cなど多要素があるが,ここでは、
d = a*s + b + c 
で表すとする。仮の係数として a=5 deg/sec, b=15 sec, c=20 sec を採用する
・ 北半球における一般的な観測の天球上の経路としては、地平線に没するまでの時間を考慮して、南西に低い天体から始めて、同じ時角帯に沿って高赤緯へ移り、Uターンして次の時角帯にそって南下し、南端でUターンし、、というパターンが推奨されるであろう。時角帯の幅をいくらにするのが良いかは、観測すべき天体数にも依存するであろう。
修正１
必要な情報を次のような"obs_list.txt"で与え、
天体名がHR番号の場合はBSC5.dbを利用して座標を得る、
修正２
・観測経路を赤経-赤緯グラフで示すobservation_path.pngを出力。
・構成の外部化: a, b, c係数、DEFAULT_WEIGHT、MIN_ALTITUDE_DEG、BSC_DB_FILE、PLOT_OUTPUT_FILEといった設定値をobs_schedule.cfgというJSON形式の設定ファイルから読み込むように変更。
・コマンドライン引数: argparseモジュールを使用して、観測リストファイル（obs_list.txt）のパスや観測日付をコマンドライン引数で指定できる
・タイムゾーンの取り扱い: 観測日付のタイムゾーンを明示的にAsia/Tokyoとして扱い、astropyで正確にUTCに変換してから処理する
修正3
・観測地情報をobs_list.txtファイルからobs_schedule.cfgファイルへ移動させた
修正４
・設定情報をobs_schedule.cfgファイル(JASON)からobs_schedule.yml(YAML)に変更

"""
import math
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import namedtuple
import astropy
import sys
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from astropy.time import Time
import re # Regular expression module for parsing star names
#import json # For reading config file
import yaml
import argparse # For command-line arguments
import pytz # For timezone handling
import numpy as np # plot_observation_path で追加されたモジュール

# Define namedtuple for ObserverSite and Star
ObserverSite = namedtuple("ObserverSite", ["lat", "lon", "alt"])
Star = namedtuple("Star", ["name", "ra", "dec", "weight", "minutes_left", "skycoord"])
# 定数（仮）
MIN_ALTITUDE_DEG = 10.0  # 実際は cfgファイルから読まれる

def get_coordinates_from_db(hr_number, bsc_db_path):
    """
    BSC5.db から HR 番号に基づいて RA と Dec を取得する。
    RA は時、Dec は度で返される。
    """
    conn = None
    try:
        conn = sqlite3.connect(bsc_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ra, decl FROM bsc5 WHERE hr = ?", (int(hr_number),))
        row = cursor.fetchone()
        if row:
            return row[0], row[1] # RA in hours, Dec in degrees
        else:
            return None, None
    except Exception as e:
        print(f"Error accessing BSC5.db for HR {hr_number} at {bsc_db_path}: {e}")
        return None, None
    finally:
        if conn:
            conn.close()

def parse_obs_list(file_path):
    """obs_list.txt の読み込みとパース"""
    obs_datetime_obj = None 
    scheduling_strategy = "original" # Default strategy
    star_list = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                elif line.startswith("obs_datetime"):
                    # ここを修正: 最初のコロンでのみ分割する
                    raw_datetime_str = line.split(":", 1)[1].strip() 
                    # print(f"DEBUG: Processing obs_datetime string: '{raw_datetime_str}' (Length: {len(raw_datetime_str)})") # デバッグ用print文は削除またはコメントアウト
                    
                    obs_datetime_obj = datetime.strptime(raw_datetime_str, "%Y-%m-%dT%H:%M:%S")
                    # print(obs_datetime_obj) # デバッグ用print文は削除またはコメントアウト
                elif line.startswith("scheduling_strategy"):
                    strategy_val = line.split(":")[1].strip().lower()
                    if strategy_val in ["original", "time_to_set_priority"]:
                        scheduling_strategy = strategy_val
                    else:
                        print(f"Warning: Unknown scheduling strategy '{strategy_val}'. Using 'original'.")
                        scheduling_strategy = "original"
                elif line.startswith('"'):
                    parts = [x.strip().strip('"') for x in line.split(",")]
                    name = parts[0]
                    ra_str = parts[1].strip() if parts[1].strip() else None
                    dec_str = parts[2].strip() if parts[2].strip() else None
                    weight = int(parts[3]) if len(parts) > 3 and parts[3].strip() else None
                    star_list.append((name, ra_str, dec_str, weight))
    except FileNotFoundError:
        print(f"Error: obs_list.txt file not found at {file_path}")
        return None, None, [], "original"
    except Exception as e:
        print(f"Error parsing obs_list.txt: {e}")
        return None, None, [], "original"

    return obs_datetime_obj, star_list, scheduling_strategy

def estimate_altitude(skycoord, obs_time, location):
    altaz_frame = AltAz(obstime=obs_time, location=location)
    altaz = skycoord.transform_to(altaz_frame)
    return altaz.alt.deg

def estimate_minutes_until_altitude(skycoord, obs_time, location, min_altitude_deg):
    """
    指定高度(min_altitude_deg)に到達するまでの時間（分）を高速・近似的に計算。
    高度・緯度・赤緯から時角を計算し、LSTとの差からΔtを求める。
    """
    ra = skycoord.ra.hour  # [h]
    dec = skycoord.dec.deg
    lat = location.lat.deg

    sin_h = math.sin(math.radians(min_altitude_deg))
    sin_phi = math.sin(math.radians(lat))
    sin_dec = math.sin(math.radians(dec))
    cos_phi = math.cos(math.radians(lat))
    cos_dec = math.cos(math.radians(dec))

    try:
        cosH = (sin_h - sin_phi * sin_dec) / (cos_phi * cos_dec)
        cosH = max(-1.0, min(1.0, cosH))
        H = math.degrees(math.acos(cosH)) / 15.0  # 時角 [h]
    except ZeroDivisionError:
        return float("inf")

    lst_now = obs_time.sidereal_time('apparent', longitude=location.lon).hour
    lst_altitude = ra + H

    delta_h = (lst_altitude - lst_now) % 24  # [h]
    return delta_h * 60  # [min]

def angular_distance(skycoord1, skycoord2):
    """astropy を使用して2つの天体間の角度距離を計算"""
    return skycoord1.separation(skycoord2).deg

def schedule_observations(stars, initial_astropy_time, location, scheduling_strategy):
#    a, b, c = coeffs['a_coeff'], coeffs['b_coeff'], coeffs['c_coeff']
    scheduled = []
    a, b, c = A_COEFF, B_COEFF, C_COEFF
    min_altitude_deg = MIN_ALTITUDE_DEG
    # Initial pointing: Assume it starts at the zenith of the observer at the initial_astropy_time.
    current_skycoord = SkyCoord(ra=initial_astropy_time.sidereal_time('apparent', longitude=location.lon),
                                dec=location.lat,
                                frame='icrs')

    current_elapsed_time_seconds = 0 # Elapsed time in seconds

    remaining = stars[:]
    
    # Initial calculation of minutes_left for all stars at the starting time
    for i in range(len(remaining)):
        minutes = estimate_minutes_until_altitude(remaining[i].skycoord, initial_astropy_time, location, min_altitude_deg)
        remaining[i] = remaining[i]._replace(minutes_left=minutes)

    while remaining:
        best = None
        best_score = -1
        
        current_astropy_time = initial_astropy_time + current_elapsed_time_seconds * u.second

        for star in remaining:
            alt = estimate_altitude(star.skycoord, current_astropy_time, location)
            minutes_left_at_current_time = estimate_minutes_until_altitude(star.skycoord, current_astropy_time, location, min_altitude_deg)

            # Filter out stars below min_altitude_deg or already set
            if alt <= min_altitude_deg or minutes_left_at_current_time < 0:
                continue
            
            dist = angular_distance(current_skycoord, star.skycoord)
            cost = a * dist + b + c
            
            # --- Scoring Logic based on scheduling_strategy ---
            value = 0 # Initialize value

            if scheduling_strategy == "original":
                # Original logic: high altitude priority with setting penalty
                value = star.weight * alt 
                if minutes_left_at_current_time < 60 and minutes_left_at_current_time > 0:
                    value *= (minutes_left_at_current_time / 60.0)**2 
                elif minutes_left_at_current_time == float('inf'):
                    pass # Circumpolar stars have no penalty for setting soon
            
            elif scheduling_strategy == "time_to_set_priority":
                # Prioritize stars about to set, high altitude itself doesn't add value
                if minutes_left_at_current_time == float('inf'):
                    value = star.weight * 0.01 # Circumpolar stars: very low value as they never set
                elif minutes_left_at_current_time > 0:
                    effective_minutes_left = min(minutes_left_at_current_time, 240) # Cap at 4 hours (240 minutes)
                    value = star.weight * (1.0 / (effective_minutes_left + 0.1)) # Add epsilon to avoid div by zero
            
            else:
                # Fallback for unknown strategy, use original behavior (shouldn't happen with validation)
                print(f"Warning: Unknown scheduling strategy '{scheduling_strategy}'. Using 'original' behavior.")
                value = star.weight * alt 
                if minutes_left_at_current_time < 60 and minutes_left_at_current_time > 0:
                    value *= (minutes_left_at_current_time / 60.0)**2 
                elif minutes_left_at_current_time == float('inf'):
                    pass

            # --- End of Scoring Logic ---

            score = value / cost
            
            if score > best_score:
                best_score = score
                best = (star, dist, cost, alt, value, score)
        
        if best:
            star, dist, cost, alt, value, score = best
            
            # Get LST for display at the observation time
            lst_for_display = current_astropy_time.sidereal_time('apparent', longitude=location.lon).hour
            
            # Include RA and Dec in the scheduled list for plotting
            scheduled.append((star.name, star.ra, star.dec, round(alt,1), round(value,1), round(cost,1), round(score,3), round(lst_for_display, 2), round(current_elapsed_time_seconds / 60.0, 1)))
            
            # Update current_skycoord to the observed star's coordinates for the next iteration
            current_skycoord = star.skycoord 
            current_elapsed_time_seconds += cost 
            remaining.remove(star)
        else:
            break # No more observable stars
    return scheduled


def plot_observation_path(scheduled_data, initial_stars, initial_astropy_time, location, min_altitude_deg, output_file):
    """
    観測経路を天頂中心の高度方位グラフにプロットします。
    観測開始時の各天体の位置を背景に示し、観測経路をその上に描画します。

    Args:
        scheduled_data (list): schedule_observationsから返されるスケジュールされた観測のリスト。
                                各要素は (name, ra, dec, ...) のタプルであると期待されます。
        initial_stars (list): スケジュール前の、すべてのStarオブジェクトのリスト。
                                観測開始時の天球上の天体位置を示すために使用されます。
        initial_astropy_time (astropy.time.Time): 観測開始時刻のastropy Timeオブジェクト。
        location (astropy.coordinates.EarthLocation): 観測地点のEarthLocationオブジェクト。
        min_altitude_deg (float): 観測可能な最低高度（度）。地平線の定義に使用されます。
        output_file (str): プロットを保存するファイル名。
    """
    plt.figure(figsize=(10, 10))
    ax = plt.gca() # 現在の軸を取得

    # 天頂中心の円形プロットのための設定
    # 高度90度（天頂）が中心(r=0)、高度0度（地平線）が外周(r=R_horizon)に対応します。
    # R_horizonはプロットの最大半径（90度を基準に設定）
    R_horizon = 90 # 高度をそのまま半径にマッピング（90度が中心、0度が外周）
    
    # 地平線 (0度) の円を描画
    horizon_circle = plt.Circle((0, 0), R_horizon, color='gray', fill=False, linestyle='-', linewidth=1.5, label='Horizon (0° Alt)')
    ax.add_patch(horizon_circle)

    # 高度円を描画 (薄い色で)
    for alt_deg in [30, 60]:
        # 高度が高いほど中心に近い
        r_circle = R_horizon * (90 - alt_deg) / 90
        circle = plt.Circle((0, 0), r_circle, color='lightgray', fill=False, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.add_patch(circle)
        # 高度ラベルを追加
        ax.text(r_circle + 2, 0, f'{alt_deg}°', va='center', ha='left', color='gray', fontsize=8)

    # 方位線を描画 (薄い色で)
    azimuth_labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
    for az_deg in range(0, 360, 45): # 45度ごとに線とラベル
        # 北（0度）が上方向（Y軸正方向）に来るように方位角を変換
        theta_rad = np.deg2rad(90 - az_deg) # 北=0, 東=90, 南=180, 西=270
        ax.plot([0, R_horizon * np.cos(theta_rad)], [0, R_horizon * np.sin(theta_rad)], 
                color='lightgray', linestyle='--', linewidth=0.5, alpha=0.6)
        
        # 方位ラベルを円の外側に追加
        label_r = R_horizon + 5 # ラベルの距離
        ax.text(label_r * np.cos(theta_rad), label_r * np.sin(theta_rad), 
                azimuth_labels.get(az_deg, f'{az_deg}°'), 
                ha='center', va='center', color='dimgray', fontsize=9)

    # 天球座標（RA/Dec）を高度方位座標（Alt/Az）に変換し、さらにプロット用の(x,y)座標に変換するヘルパー関数
    def transform_skycoord_to_xy(skycoord, obs_time, obs_location):
        altaz_frame = AltAz(obstime=obs_time, location=obs_location)
        altaz_coords = skycoord.transform_to(altaz_frame)
        
        alt_deg = altaz_coords.alt.deg
        az_deg = altaz_coords.az.deg

        # 高度を半径に変換（90度=中心、0度=外周）
        r = R_horizon * (90 - alt_deg) / 90
        # 方位を角度に変換（北=上、東=右）
        theta_rad = np.deg2rad(90 - az_deg)

        x = r * np.cos(theta_rad)
        y = r * np.sin(theta_rad)
        return x, y, alt_deg # 高度も返すことで、後でフィルタリング可能

    # 観測開始時のすべての天体（initial_stars）を背景にプロット
    initial_x_coords = []
    initial_y_coords = []
    for star in initial_stars:
        x, y, alt_deg = transform_skycoord_to_xy(star.skycoord, initial_astropy_time, location)
        # 地平線より少し下にある天体（例：-10度まで）もプロットに含める
        if alt_deg > -10: 
            initial_x_coords.append(x)
            initial_y_coords.append(y)
            # 各天体の名前ラベルは、点が多いとグラフが読みにくくなるため、ここでは省略
            # 必要であれば ax.text(x, y, star.name, fontsize=7, color='gray', alpha=0.7, ha='center', va='bottom') を追加
    
    ax.plot(initial_x_coords, initial_y_coords, 'o', color='skyblue', markersize=4, alpha=0.5, label='All Stars (Initial Position)')

    # 観測経路をプロット
    path_x_coords = []
    path_y_coords = []
    for i, row in enumerate(scheduled_data):
        star_name = row[0]
        star_ra_h = row[1]
        star_dec_deg = row[2]
        
        # スケジュールされた各天体のSkyCoordを再構築
        star_skycoord = SkyCoord(ra=star_ra_h*u.hourangle, dec=star_dec_deg*u.deg, frame='icrs')
        
        x, y, _ = transform_skycoord_to_xy(star_skycoord, initial_astropy_time, location)
        path_x_coords.append(x)
        path_y_coords.append(y)
        
        # 観測経路上の各点に天体名ラベルを追加
        ax.text(x, y, star_name, fontsize=8, ha='center', va='bottom', color='darkblue')

    if path_x_coords:
        ax.plot(path_x_coords, path_y_coords, 'o-', color='blue', linewidth=2, markersize=7, label='Observation Path')
        
        # 観測経路の開始点と終了点を強調表示
        ax.plot(path_x_coords[0], path_y_coords[0], 'o', color='green', markersize=9, zorder=5, label='Path Start')
        ax.plot(path_x_coords[-1], path_y_coords[-1], 'o', color='red', markersize=9, zorder=5, label='Path End')
        
    ax.set_aspect('equal', adjustable='box') # アスペクト比を1:1に設定し、円が歪まないようにする
    
    # プロット範囲を調整（円とラベルがすべて収まるように少し広げる）
    plot_limit = R_horizon * 1.3 # 例: 円の半径の1.3倍
    ax.set_xlim([-plot_limit, plot_limit]) 
    ax.set_ylim([-plot_limit, plot_limit])
    
    ax.set_xticks([]) # X軸の目盛りを非表示
    ax.set_yticks([]) # Y軸の目盛りを非表示
    ax.set_frame_on(False) # プロットの枠線を非表示

    # グラフタイトル
    import pytz # タイトル表示用に再インポート
    local_tz = pytz.timezone("Asia/Tokyo") # 天頂時刻表示のためにタイムゾーンを再定義
    title_time_local = initial_astropy_time.to_datetime(local_tz).strftime('%Y-%m-%d %H:%M JST')
    plt.title(f"Observation Path on Celestial Sphere\n(Zenith-centered at {title_time_local})")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.9)) # 凡例を外側に配置
    plt.tight_layout() # レイアウトを自動調整
    plt.savefig(output_file)
    plt.close()
    print(f"観測経路のプロットを '{output_file}' に保存しました。")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="天体観測のスケジューリングと経路プロット。")
    parser.add_argument('-f', '--obs-list-file', default="obs_list.txt", help='観測リストファイルへのパス (デフォルト: obs_list.txt)')
    parser.add_argument('-d', '--obs-date', help='観測日付 (YYYY-MM-DD)。obs_list.txtの値を上書きします。')
    args = parser.parse_args()

    CONFIG_FILE = "obs_schedule.yml"
    try:
        with open(CONFIG_FILE, 'r') as f:
            config_data = yaml.safe_load(f)       # YAML読み込み
    except FileNotFoundError:
        sys.exit(f"Error: '{CONFIG_FILE}' not found.")
    except yaml.YAMLError as e:
        sys.exit(f"Error parsing YAML: {e}")

    obs_cfg = config_data["observer_site"]
    site = ObserverSite(
        lat=obs_cfg["latitude"],
        lon=obs_cfg["longitude"],
        alt=obs_cfg["altitude"]
    )
    location = EarthLocation(
        lat=site.lat * u.deg,
        lon=site.lon * u.deg,
        height=site.alt * u.m
    )
    obs_set = config_data.get("observation_settings", {})
    A_COEFF = obs_set.get("a_coeff", 5)
    B_COEFF = obs_set.get("b_coeff", 15)
    C_COEFF = obs_set.get("c_coeff", 20)
    DEFAULT_WEIGHT = obs_set.get("default_weight", 3)
    MIN_ALTITUDE_DEG = obs_set.get("min_altitude_deg", 10)
    PLOT_OUTPUT_FILE = obs_set.get("plot_output_file", "observation_path.png")
    db_set    = config_data.get("database_settings", {})
    BSC_DB_FILE = db_set.get("bsc_db_path", "/home/o2/Documents/etc/BSC5/BSC5.db")
    # 観測日時と天体リストのパース
    obs_datetime, raw_stars, scheduling_strategy = parse_obs_list(args.obs_list_file)

    # EarthLocation と Time の生成
    location = EarthLocation(lat=site.lat*u.deg,
                             lon=site.lon*u.deg,
                             height=site.alt*u.m)

    tz = pytz.timezone("Asia/Tokyo")
    local_dt = tz.localize(obs_datetime)
    initial_astropy_time = Time(local_dt, format='datetime', scale='utc')

    stars = []
    for name, ra_str, dec_str, weight in raw_stars:
        star_skycoord = None
        if ra_str is None or dec_str is None:
            if name.startswith("HR"):
                try:
                    hr = int(name[2:])
                    ra_deg, dec_deg = get_coordinates_from_db(hr, BSC_DB_FILE) # BSC_DB_FILEを渡す
                    if ra_deg is not None and dec_deg is not None:
                        star_skycoord = SkyCoord(ra=ra_deg*u.hourangle, dec=dec_deg*u.deg, frame='icrs')
                except ValueError:
                    print(f"Warning: Invalid HR number format for {name}. Skipping.")
                    continue
            if star_skycoord is None:
                print(f"Warning: Could not get coordinates for {name}. Skipping.")
                continue
        else:
            try:
                star_skycoord = SkyCoord(ra_str + " " + dec_str, unit=(u.hourangle, u.deg), frame='icrs')
            except Exception as e:
                print(f"Error parsing coordinates for {name} ({ra_str}, {dec_str}): {e}. Skipping.")
                continue
        minutes_left =  estimate_minutes_until_altitude(star_skycoord, initial_astropy_time, location, MIN_ALTITUDE_DEG)
#        minutes_left = estimate_minutes_until_set(star_skycoord, initial_astropy_time, location, MIN_ALTITUDE_DEG)
        stars.append(Star(name=name, ra=star_skycoord.ra.hour, dec=star_skycoord.dec.deg, weight=weight, minutes_left=minutes_left, skycoord=star_skycoord))

    scheduled = schedule_observations(stars, initial_astropy_time, location, scheduling_strategy)

    print(f"{'Star':<12} {'Alt[°]':>8} {'Value':>8} {'Cost[s]':>8} {'Score':<8} {'LST[h]':<8} {'Elapsed[min]':<12}")
    print("-" * 75)
    for row in scheduled:
        print(f"{row[0]:<12} {row[1]:8.1f} {row[2]:>8} {row[3]:>8} {row[4]:<8.3f} {row[5]:<8.2f} {row[6]:<12.1f}")
    print("-" * 75)

    # ... (後続の星リストの処理、スケジューリング、プロットの呼び出しは変更なし)
# 観測経路をプロット
    plot_observation_path(scheduled, stars, initial_astropy_time, location, MIN_ALTITUDE_DEG, PLOT_OUTPUT_FILE)


"""
#obs_list.txt
#観測地　経度は東経をプラスとする
#longitude(deg): 133.5867
#latitude(deg): 34.304
#alt(metor): 7
#観測予定夜(地方標準時)
obs_datetime: 2025-07-12T20:00:00
#スケジューリング戦略 (original: 高高度優先, time_to_set_priority: 地平没までの時間優先)
scheduling_strategy: time_to_set_priority
# 天体リストRA[h], Dec[°], 重み
# 天体名がHR番号の場合は座標は省略可（BSC5.dbから求める）
object:
"S_-60_0.0h", "00 00 00.000", "-60 00 00.0", 1
"S_-60_4.0h", "04 00 00.000", "-60 00 00.0", 1
"S_-60_8.0h", "08 00 00.000", "-60 00 00.0", 1
"S_-60_12.0h", "12 00 00.000", "-60 00 00.0", 1
"S_-60_16.0h", "16 00 00.000", "-60 00 00.0", 1
"S_-60_20.0h", "20 00 00.000", "-60 00 00.0", 1



#obs_schejule.cfg　#YAMLにしたので不要に
{
  "observer_site": {
    "latitude": 34.304,
    "longitude": 133.5867,
    "altitude": 7
  },
  "observation_settings": {
        "a_coeff": 15,
        "b_coeff": 5,
        "c_coeff": 10,
        "default_weight": 1,
        "min_altitude_deg": 10,
        "plot_output_file": "observation_path.png"
  },
  "database_settings": {
        "bsc_db_path": "/home/o2/Documents/etc/BSC5/BSC5.db"
  }}

#obs_schejule.yml
#観測地情報、角度は度の小数、距離はメートル
observer_site:
  latitude: 34.304
  longitude: 133.5867
  altitude: 7

observation_settings:
#　望遠鏡移動コスト計算のための係数 cost = a*d + b + c
#  dは移動距離（度）、aは1度あたりの所要時間（秒）、bは観測時間、cはデッドタイム
  a_coeff: 15
  b_coeff: 5
  c_coeff: 10
  default_weight: 1
#　観測許容高度の下限（度）
  min_altitude_deg: 10
  plot_output_file: observation_path.png

database_settings:
  bsc_db_path: /home/o2/Documents/etc/BSC5/BSC5.db

"""

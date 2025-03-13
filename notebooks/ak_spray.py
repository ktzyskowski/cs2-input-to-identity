# import numpy as np
# import pandas as pd
# from demoparser2 import DemoParser
# from matplotlib import pyplot as plt
# from scipy.interpolate import make_interp_spline
#
#
# def distance_between_consecutive_points(pitch, yaw):
#     dx = np.diff(yaw)
#     dy = np.diff(pitch)
#     distances = np.sqrt(dx ** 2 + dy ** 2)
#     return distances
#
#
# def plot_points(pitch, yaw):
#     t = np.linspace(0, 1, len(yaw))
#     x_spline = make_interp_spline(t, yaw, k=3)
#     y_spline = make_interp_spline(t, pitch, k=3)
#     t_fine = np.linspace(0, 1, 1000)
#     x_fine = x_spline(t_fine)
#     y_fine = y_spline(t_fine)
#
#     ax = plt.gca()
#     ax.set_xlim([-15, 15])
#     ax.set_ylim([-12, 1])
#     plt.scatter(yaw, pitch, c=np.arange(len(yaw)), cmap="coolwarm_r")
#     plt.plot(x_fine, y_fine, c="black", linewidth=0.3)
#     plt.show()
#
#
# def get_weapon_fire_df(parser: DemoParser, weapon: str = "weapon_ak47"):
#     """Given a demo parser, extract a DataFrame containing the weapon fire events.
#
#     Extra processing is done to aggregate each weapon fire event into a collection of sprays. Each spray is given a
#     unique numeric identifier in a new column.
#
#     The weapon fire events are filtered by weapon type. The default is the AK47.
#
#     :param parser: the demo parser.
#     :param weapon: the weapon to extract events for.
#     :return: a DataFrame containing the weapon fire events.
#     """
#     # slow calls at the start
#     weapon_fire_df = parser.parse_event("weapon_fire")
#     tick_df = parser.parse_ticks(["FIRE", "active_weapon_ammo"], ticks=weapon_fire_df.tick)
#
#     # filter by specific weapon and sort by player ID and tick
#     weapon_fire_df = weapon_fire_df[weapon_fire_df.weapon == weapon]
#     weapon_fire_df = weapon_fire_df.sort_values(by=["user_steamid", "tick"])
#
#     # collect spray IDs
#     spray_ids = []
#     current_spray_id = 0
#     for _, weapon_fire_row in weapon_fire_df.iterrows():
#         tick, steamid = weapon_fire_row.tick, int(weapon_fire_row.user_steamid)
#         tick_row = tick_df[(tick_df.tick == tick) & (tick_df.steamid == steamid)]
#         if not tick_row.iloc[0].FIRE:
#             # if FIRE is not down, this weapon fire is start of a new spray
#             current_spray_id = current_spray_id + 1
#         spray_ids.append(current_spray_id)
#     weapon_fire_df["spray_id"] = spray_ids
#
#     weapon_fire_df = weapon_fire_df.reset_index(drop=True)
#     return weapon_fire_df
#
#
# def iterate_sprays(
#         weapon_fire_df: pd.DataFrame,
#         player_hurt_df: pd.DataFrame,
#         min_shots: int = 3,
# ):
#     for spray_id, spray_df in weapon_fire_df.groupby("spray_id"):
#         # do not include single fire taps
#         if spray_df.shape[0] < min_shots:
#             continue
#
#         # collect information on who is hurt during spray
#         # spray_hurt_df = player_hurt_df[player_hurt_df.attacker_steamid == spray_df.iloc[0].user_steamid]
#         # spray_hurt_df = spray_hurt_df[spray_hurt_df.tick.isin(spray_df.tick.values)]
#         # if spray_hurt_df.shape[0] == 0:
#         #     # at least 1 bullet should be on target
#         #     continue
#
#         # if len(spray_hurt_df.user_steamid.unique()) > 1:
#         #     # only consider sprays on a single target
#         #     continue
#
#         yield spray_df
#
#
# def main():
#     # parser = DemoParser("/Users/ktz/msai/msthesis/res/003733085075743440950_0998745574.dem")
#     parser = DemoParser("/Users/ktz/msai/msthesis/res/chinggis-warriors-vs-the-huns-inferno.dem")
#
#     # collect all weapon sprays
#     weapon_fire_df = get_weapon_fire_df(parser)
#     print(weapon_fire_df)
#
#     # collect all player hurt events
#     player_hurt_df = parser.parse_event("player_hurt")
#
#     # collect mouse information for each tick
#     mouse_df = parser.parse_ticks(["yaw", "pitch"])
#
#     min_shots = 5  # minimum number of shots contained in a spray
#     max_deviation = 3  # max distance between each point in a spray (in degrees)
#     for spray_df in iterate_sprays(weapon_fire_df, player_hurt_df, min_shots=min_shots):
#         spray_mouse_df = mouse_df[mouse_df.tick.isin(spray_df.tick.values)]
#         spray_mouse_df = spray_mouse_df[spray_mouse_df.steamid == int(spray_df.iloc[0].user_steamid)]
#
#         # shift sprays so they begin at origin
#         spray_mouse_df.pitch -= spray_mouse_df.iloc[0].pitch
#         spray_mouse_df.yaw -= spray_mouse_df.iloc[0].yaw
#
#         yaw, pitch = spray_mouse_df.yaw, -spray_mouse_df.pitch
#         if np.any(distance_between_consecutive_points(pitch, yaw) > max_deviation):
#             # eliminate samples that deviate too much from spray
#             continue
#
#         plot_points(pitch, yaw)
#
#
# if __name__ == "__main__":
#     # pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#
#     main()

"""
1. Data Processing & Engineering

here is the data processing and engineering part after collecting data from https://cricsheet.org/matches/
Data received are in the json format and transformed the data with needed features in to CSV format.

"""
import pandas as pd
import os
from dotenv import load_dotenv
import logging
import json

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# get constants from environment variables
DIRECTORY_PATH = os.getenv("DIRECTORY_PATH")  # directory where json data files are stored.

# to store all records related to match
match_data_list = []
# to store all records related to player
player_data_list = []

# Iterate over all subdirectories and JSON files
for root, _, files in os.walk(DIRECTORY_PATH):
    for file in files:
        if file.endswith(".json"):  # Process only JSON files
            json_path = os.path.join(root, file)
            # Read the JSON file as a dictionary
            try:
                with open(json_path, "r", encoding="utf-8") as json_file:
                    data_dict = json.load(json_file)
                    logger.info(f"length of innings list , {len(data_dict['innings'])}")
                    info = data_dict.get("info", {})
                    innings = data_dict.get("innings", {})
                    match_data = {
                        "balls_per_over": info.get("balls_per_over", "N/A"),
                        "city": info.get("city", "N/A"),
                        "venue": info.get("venue", "N/A"),
                        "match_dates": info.get("dates", [None]),
                        "event_name": info.get("event", {}).get("name", "N/A"),
                        "gender": info.get("gender", "N/A"),
                        "match_type": info.get("match_type", "N/A"),
                        "match_referees": info.get("officials", {}).get("match_referees", [None]),
                        "reserve_umpires": info.get("officials", {}).get("reserve_umpires", [None]),
                        "tv_umpires": info.get("officials", {}).get("tv_umpires", [None]),
                        "umpires": info.get("officials", {}).get("umpires", [None]),
                        "winner": info.get("outcome", {}).get("winner", "N/A"),
                        "winner_runs": info.get("outcome", {}).get("runs", "N/A"),
                        "total_overs": info.get("overs", "N/A"),
                        "player_of_match": info.get("player_of_match", "N/A"),
                        "season": info.get("season", "N/A"),
                        "team_type": info.get("team_type", "N/A"),
                        "teams": info.get("teams", [None]),
                    }
                    # Extract players for each team
                    players = info.get("players", {})

                    # Transforming dictionary to list of tuples
                    players_list = [(player, team) for team, players in players.items() for player in players]
                    match_data["players"] = players_list
                    # process innings data

                    innings_earned = {"player": "",
                                      "innings": {"taotal_runs_achieved_by_player": [], "sixes_by_player": []}}
                    # Creating dictionary for total runs and sixes
                    batting_stats = {}
                    try:
                        for team in innings:
                            team_of_player = team['team']
                            for over in team['overs']:
                                for delivery in over['deliveries']:
                                    batter = delivery['batter']
                                    runs = delivery['runs']['batter']

                                    if batter not in batting_stats:
                                        batting_stats[batter] = {"total_runs_achieved_by_player": 0,
                                                                 "sixes_by_player": 0,
                                                                 "team": team_of_player}

                                    batting_stats[batter]["total_runs_achieved_by_player"] += runs
                                    if runs == 6:
                                        batting_stats[batter]["sixes_by_player"] += 1
                    except Exception as e:
                        logger.error(f"Error processing innings data in {file}: {e}")
                        continue
                    player_status = {}
                    for player, stats in batting_stats.items():
                        player_status["player"] = player,
                        player_status["runs_achieved"] = stats["total_runs_achieved_by_player"],
                        player_status["sixes_achieved"] = stats["sixes_by_player"],
                        player_status["team"] = stats["team"],
                        player_status["event_city"] = match_data["city"],
                        player_status["event_venue"] = match_data["venue"],
                        player_status["event_dates"] = match_data["match_dates"],
                        player_status["event_name"] = match_data["event_name"],
                        player_status["match_type"] = match_data["match_type"],
                        player_status["season"] = match_data["season"],
                        player_status["team_type"] = match_data["team_type"],
                        try:
                            if len(match_data["teams"]) == 2:
                                opposite_team = [team for team in match_data["teams"] if team != stats["team"]][0]
                            else:
                                opposite_team = "N/A"
                        except Exception as e:
                            logger.warning(f"Error fetching opposite team for {player} in {file}: {e}")
                            opposite_team = "N/A"
                        # print("opposite_team",opposite_team)
                        player_status["opposite_team"] = opposite_team,
                    match_data_list.append(match_data)
                    player_data_list.append(player_status)
                    # print(data_list)
                    logger.info(f'Processed...,{file}')
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error in {file}: {e}")
            except FileNotFoundError as e:
                logger.error(f"File not found: {json_path}")
            except Exception as e:
                logger.error(f"Unexpected error processing {file}: {e}")

# Concatenate all DataFrames
try:
    # Ensure match_data_list and player_data_list are not empty
    if match_data_list:
        # df_match_data = pd.concat(match_data_list, ignore_index=True)
        df_match_data = pd.DataFrame(match_data_list)
    else:
        df_match_data = pd.DataFrame()
        logging.warning("match_data_list is empty. Creating an empty DataFrame.")
    if player_data_list:
        # df_player_data = pd.concat(player_data_list, ignore_index=True)
        df_player_data = pd.DataFrame(player_data_list)
    else:
        df_player_data = pd.DataFrame()
        logging.warning("player_data_list is empty. Creating an empty DataFrame.")
    # df = pd.DataFrame(data["info"])
    try:
        df_match_data.to_csv(os.path.join(DIRECTORY_PATH, f"match_data.csv"), encoding='utf-8', index=False)
        df_player_data.to_csv(os.path.join(DIRECTORY_PATH, f"player_data.csv"), encoding='utf-8', index=False)
    except Exception as e:
        logging.error(f"Error saving CSV files: {e}")

except Exception as e:
    logging.error(f"Error processing DataFrames: {e}")
# logger.info(f'Processed...,{filename}')
logger.info("json files converted to csv")

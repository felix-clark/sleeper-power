#!/usr/bin/env python

import os
import time
from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import combinations

# TODO: Consider cvxpy for a simpler interface; possibly use mip backend
import mip
import pandas as pd
import requests
from plotnine import aes, geom_point, geom_text, ggplot, labs
from plotnine.geoms import annotate
from plotnine.scales import scale_color_cmap, xlim, ylim
from plotnine.themes import theme_bw, theme_set


def get_url(*args) -> str:
    sleeper_base_url = "https://api.sleeper.app/v1"
    """Compose a URL for a sleeper API request"""
    elements = [sleeper_base_url] + [str(arg) for arg in args]
    url: str = "/".join(elements)
    return url


def get_req(*args) -> dict:
    """Return a sleeper API request as a JSON"""
    url: str = get_url(*args)
    req = requests.get(url, timeout=10)
    return req.json()


def get_nfl() -> dict:
    """Get the current state of the NFL"""
    return get_req("state", "nfl")


def get_rosters(league_id) -> dict:
    """Get the rosters in a league"""
    return get_req("league", league_id, "rosters")


def get_user(user_id) -> dict:
    """Get a user based on their ID (can also be their username)"""
    return get_req("user", user_id)


def get_users(league_id) -> dict:
    """Get all users in a league"""
    url = get_url("league", league_id, "users")
    req = requests.get(url, timeout=10)
    users = req.json()
    return users


def get_league(league_id: int) -> dict:
    """Get a league based on the ID"""
    return get_req("league", league_id)


def get_players(cache: str = "players.parq") -> pd.DataFrame:
    """Get database of players, caching locally."""
    if os.path.isfile(cache):
        mod_time = os.path.getmtime(cache)
        current_time = time.time()
        since_mod = current_time - mod_time
        # Update the database at most once per day
        if since_mod < 60 * 60 * 24:
            d_play = pd.read_parquet(cache)
            return d_play
    print("Updating player cache")
    players = get_req("players", "nfl")
    d_play = pd.DataFrame(players).transpose()
    d_play.to_parquet(cache)
    return d_play


def get_matchups(league_id: int, week: int) -> dict:
    """Get all matchups for a league in a given week"""
    return get_req("league", league_id, "matchups", week)


def allowed_slots(roster: set, player_pos: list) -> list:
    """Return a list of allowed positions for the player"""
    std_pos = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
    flex_pos = {
        'FLEX': set(['RB', 'WR', 'TE']),
        'WRRB_FLEX': set(['RB', 'WR']),
        'REC_FLEX': set(['WR', 'TE']),
    }
    if roster - set(std_pos) - set(flex_pos.keys()):
        raise NotImplementedError(f"Position in {roster} not supported")
    player_set = set(player_pos)
    allowed = roster & std_pos & player_set
    for fpos, fset in flex_pos.items():
        if fset & player_set:
            allowed.add(fpos)
    # return in same order as roster
    return [p for p in roster if p in allowed]


def max_points(league_roster, db_player, matchup) -> float:
    """Compute the maximum points that could have been acquired from a perfect lineup."""
    assert abs(sum(matchup["starters_points"]) - matchup["points"]) < 0.01
    starting_roster = Counter([p for p in league_roster if p != "BN"])
    # points scored for all players, including bench
    players_points = matchup["players_points"]
    matchup_players = db_player.loc[matchup["players"]]
    player_pos = matchup_players["fantasy_positions"]
    player_allowed = player_pos.apply(lambda ps: allowed_slots(starting_roster.keys(), ps))
    mod = mip.Model(sense=mip.MAXIMIZE)
    mod.verbose = 0
    # List of player ID, slots, and model variable
    vars = []
    for pl_id, pl_slots in player_allowed.items():
        pl_vars = []
        for slot in pl_slots:
            v = mod.add_var(name=f"{pl_id}_{slot}", var_type=mip.BINARY)
            pl_vars.append((pl_id, slot, v))
        # Add constraint that each player is only used once at most
        mod += mip.xsum(v for _, _, v in pl_vars) <= 1, pl_id
        vars.extend(pl_vars)
    for slot, n_players in starting_roster.items():
        mod += mip.xsum(v for _, pslot, v in vars if pslot == slot) <= n_players, slot
        # print(slot, n_players)
    mod.objective = mip.xsum(players_points[pid] * v for pid, _, v in vars)
    status = mod.optimize()
    assert status == mip.OptimizationStatus.OPTIMAL, "FEASIBLE might also be allowed"
    # for v in mod.vars:
    #     pl_id = v.name.split("_")[0]
    #     pl_name = matchup_players.loc[pl_id, "full_name"]
    #     print(f"{pl_name} {v.name}: {v.x}")
    best: float = mod.objective_value
    return best


def main():
    """Run the main testing function"""
    parser = ArgumentParser(description="Generate plots for a Sleeper fantasy football league")
    parser.add_argument("league_id", type=int, help="Sleeper league ID")
    args = parser.parse_args()
    league_id = args.league_id

    db_player = get_players()

    nfl = get_nfl()
    # print(nfl)
    current_week = nfl["week"]
    league = get_league(league_id)
    league_name = league["name"]
    league_roster = league["roster_positions"]
    users = get_users(league_id)
    user_to_name: dict = {u["user_id"]: u["display_name"] for u in users}
    rosters = get_rosters(league_id)
    roster_to_id: dict = {r["roster_id"]: r["owner_id"] for r in rosters}
    n_rosters = len(rosters)
    # print(users)
    # print(rosters[0])

    win_rates = {}
    points_for = {}
    points_against = {}
    for roster in rosters:
        try:
            user = user_to_name[roster["owner_id"]]
        except KeyError as e:
            print(e)
            user = "N/A"
        # record = roster["metadata"]["record"]
        settings = roster["settings"]
        wins = settings["wins"]
        ties = settings["ties"]
        losses = settings["losses"]
        win_rates[user] = (wins + 0.5 * ties) / (wins + losses + ties)
        points_for[user] = settings["fpts"] + 0.01 * settings["fpts_decimal"]
        points_against[user] = (
            settings["fpts_against"] + 0.01 * settings["fpts_against_decimal"]
        )

    # exit(1)
    # print(league)
    scores_by_user = defaultdict(list)
    best_scores_by_user = defaultdict(list)
    scores_by_week = []
    best_scores_by_week = []
    for week in range(1, current_week):
        print(f"week {week}")
        matchups = get_matchups(league_id, week)
        week_results = {}
        best_week_results = {}
        for matchup in matchups:
            roster_id = matchup["roster_id"]
            try:
                username = user_to_name[roster_to_id[roster_id]]
            except KeyError as e:
                print(f"Unknown user from roster ID {roster_id}")
                print(e)
                username = "N/A"
            points = matchup["points"]
            best_points = max_points(league_roster, db_player, matchup)
            scores_by_user[username].append(points)
            best_scores_by_user[username].append(best_points)
            week_results[username] = points
            best_week_results[username] = best_points
        scores_by_week.append(week_results)
        best_scores_by_week.append(best_week_results)
    # print(scores_by_user)
    # print(scores_by_week)
    # print(best_scores_by_user)

    # matchup-independent wins
    miws = defaultdict(int)
    # best-ball matchup-ind wins
    bb_miws = defaultdict(int)

    for weekly_scores in scores_by_week:
        usernames = weekly_scores.keys()
        for user_a, user_b in combinations(usernames, 2):
            score_a = weekly_scores[user_a]
            score_b = weekly_scores[user_b]
            if score_a == score_b:
                print(f"A tie between {user_a} and {user_b} with a score of {score_a}")
                miws[user_a] += 0.5
                miws[user_b] += 0.5
            elif score_a < score_b:
                miws[user_b] += 1
            elif score_a > score_b:
                miws[user_a] += 1
            else:
                assert False

    for weekly_scores in best_scores_by_week:
        usernames = weekly_scores.keys()
        for user_a, user_b in combinations(usernames, 2):
            score_a = weekly_scores[user_a]
            score_b = weekly_scores[user_b]
            if score_a == score_b:
                print(f"A best-ball tie between {user_a} and {user_b} with a score of {score_a}")
                bb_miws[user_a] += 0.5
                bb_miws[user_b] += 0.5
            elif score_a < score_b:
                bb_miws[user_b] += 1
            elif score_a > score_b:
                bb_miws[user_a] += 1
            else:
                assert False

    max_miws = (n_rosters - 1) * (current_week - 1)
    # matchup-independent win rates
    miwrs = {u: w / max_miws for u, w in miws.items()}
    bb_miwrs = {u: w / max_miws for u, w in bb_miws.items()}
    # print(miwrs)
    assert sum(miws.values()) == n_rosters * max_miws // 2
    rank_list = sorted(miws, key=lambda n: miws[n], reverse=True)
    for user in rank_list:
        miwr = miwrs[user]
        print(f"{user}:\t{miwr:.3f}")

    data = pd.DataFrame(
        {
            "win_rate": win_rates,
            # matchup-independent win rate
            "miwr": miwrs,
            # lineup- and matchup-independent win rate ("best ball" miwr)
            "bb_miwr": bb_miwrs,
            "points_for": points_for,
            "points_against": points_against,
            "bb_points_for": {n: sum(p) for n, p in best_scores_by_user.items()},
        }
    )
    # data.index.name = "name"
    data["manager"] = data.index
    data["luck"] = data["win_rate"] - data["miwr"]
    print(data)

    theme_set(theme_bw())

    # id_line_xy = [0., 1.]
    # id_data = pd.DataFrame({"x": id_line_xy, "y": id_line_xy})
    # id_line = geom_path(
    #     data=id_data,
    #     mapping=aes(x="x", y="y"),
    #     inherit_aes=False,
    #     linetype="dashed",
    #     color="grey",
    # )

    wr_plot = (
        ggplot(data=data, mapping=aes(x="luck", y="miwr"))
        + geom_point(mapping=aes(color="points_for"), size=3)
        # Make some dummy points so the text adjustment has room to move
        + geom_point(
            data=pd.DataFrame({"x": [-0.5, 0.5], "y": [0.0, 1.0]}),
            alpha=0.0,
            color="white",
            inherit_aes=False,
            mapping=aes(x="x", y="y"),
        )
        + geom_text(
            mapping=aes(label="manager"),
            adjust_text={
                "expand_points": (1.5, 1.5),
                "arrowprops": {
                    "arrowstyle": "->",
                },
            },
        )
        + xlim(-0.5, 0.5)
        + ylim(0.0, 1.0)
        + scale_color_cmap("cool")
        + labs(title=f"{league_name} performance", x="Luck", y="Power", color="Points for")
        + annotate("text", x=0., y=.95, label="good", fontstyle="italic", color="grey")
        + annotate("text", x=0., y=.05, label="bad", fontstyle="italic", color="grey")
        + annotate("text", x=0.45, y=0.5, label="lucky", fontstyle="italic", color="grey")
        + annotate("text", x=-0.45, y=0.5, label="unlucky", fontstyle="italic", color="grey")
        + annotate("text", x=0.45, y=.95, label="winning", fontstyle="italic", color="grey")
        + annotate("text", x=-0.45, y=.05, label="losing", fontstyle="italic", color="grey")
    )
    wr_plot.save("luck.png")

    eff_plot = (
        # ggplot(data=data, mapping=aes(x="bb_miwr", y="miwr"))
        ggplot(data=data, mapping=aes(x="miwr - bb_miwr", y="bb_miwr"))
        + geom_point(mapping=aes(color="bb_points_for"), size=3)
        # + id_line
        # Make some dummy points so the text adjustment has room to move
        + geom_point(
            data=pd.DataFrame({"x": [-.25, 0.25], "y": [0.0, 1.0]}),
            alpha=0.0,
            color="white",
            inherit_aes=False,
            mapping=aes(x="x", y="y"),
        )
        + geom_text(
            mapping=aes(label="manager"),
            adjust_text={
                "expand_points": (1.5, 1.5),
                "arrowprops": {
                    "arrowstyle": "->",
                },
            },
        )
        + xlim(-0.25, 0.25)
        + ylim(0.0, 1.0)
        + scale_color_cmap("cool")
        + labs(
            title=f"{league_name} depth",
            x="Lineup boost",
            y="Roster power",
            color="Max points for",
        )
        + annotate("text", x=-.2, y=.5, label="deep", fontstyle="italic", color="grey")
        + annotate("text", x=.2, y=.5, label="shallow", fontstyle="italic", color="grey")
        + annotate("text", x=0., y=.05, label="bad", fontstyle="italic", color="grey")
        + annotate("text", x=0., y=.95, label="good", fontstyle="italic", color="grey")
    )
    eff_plot.save("eff.png")


if __name__ == "__main__":
    main()

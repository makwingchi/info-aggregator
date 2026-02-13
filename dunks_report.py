#!/usr/bin/env python3
"""
Fetch Dunks & Threes games data for yesterday and summarize momentum
and player highlights using play-by-play and box score stats plus an LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from loguru import logger

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None


BASE_URL = "https://dunksandthrees.com"
DEFAULT_TZ = "America/New_York"
ENV_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
)


def fetch_url(url: str, timeout: int = 20) -> str:
    # Fetch remote HTML/JSON with a consistent user agent for the site.
    logger.debug(f"Fetching URL: {url} (timeout={timeout})")
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; dunks-report/1.0)",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def find_matching_bracket(text: str, start_idx: int, open_ch: str, close_ch: str) -> int:
    # Walk the string to find the matching closing bracket for a JS blob.
    depth = 0
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"No matching bracket for {open_ch} at {start_idx}")


def extract_array(text: str, marker: str) -> str:
    # Extract a bracketed array that starts right after a marker token.
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Marker not found: {marker}")
    # The marker ends right before the opening bracket.
    start = idx + len(marker) - 1
    if text[start] != "[":
        raise ValueError(f"Expected '[' after marker {marker}")
    end = find_matching_bracket(text, start, "[", "]")
    return text[start : end + 1]


def extract_object(text: str, marker: str) -> str:
    # Extract a braced object that starts right after a marker token.
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Marker not found: {marker}")
    # The marker ends right before the opening brace.
    start = idx + len(marker) - 1
    if text[start] != "{":
        raise ValueError(f"Expected '{{' after marker {marker}")
    end = find_matching_bracket(text, start, "{", "}")
    return text[start : end + 1]


_KEY_RE = re.compile(r'([,{])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:')


def js_to_json(js_text: str) -> str:
    # Normalize JS literals and keys so the blob can be parsed as JSON.
    text = js_text
    # Replace JS-only literals with JSON-friendly values.
    text = text.replace("void 0", "null")
    text = text.replace("NaN", "null")
    text = text.replace("undefined", "null")
    text = text.replace("-Infinity", "null")
    text = text.replace("Infinity", "null")
    # Quote unquoted keys from the embedded data.
    text = _KEY_RE.sub(r'\1"\2":', text)
    # Handle numbers like -.5 or .5 that are valid in JS but not JSON.
    text = re.sub(r'([:\[,])\s*-\.(\d+)', r"\1 -0.\2", text)
    text = re.sub(r'([:\[,])\s*\.(\d+)', r"\1 0.\2", text)
    return text


def parse_js_array(js_array_text: str) -> List[Dict[str, Any]]:
    # Parse a JS array literal into Python objects.
    json_text = js_to_json(js_array_text)
    return json.loads(json_text)


def parse_js_object(js_object_text: str) -> Dict[str, Any]:
    # Parse a JS object literal into Python objects.
    json_text = js_to_json(js_object_text)
    return json.loads(json_text)


def coerce_int(value: Any) -> Optional[int]:
    # Convert numeric stats to ints while preserving missing values.
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_shooting_stats(row: Dict[str, Any]) -> Dict[str, Optional[int]]:
    # Compute total FG from 2PT and 3PT splits.
    fg2m = coerce_int(row.get("fg2m"))
    fg2a = coerce_int(row.get("fg2a"))
    fg3m = coerce_int(row.get("fg3m"))
    fg3a = coerce_int(row.get("fg3a"))
    ftm = coerce_int(row.get("ftm"))
    fta = coerce_int(row.get("fta"))
    fgm = fg2m + fg3m if fg2m is not None and fg3m is not None else None
    fga = fg2a + fg3a if fg2a is not None and fg3a is not None else None
    return {
        "fgm": fgm,
        "fga": fga,
        "fg3m": fg3m,
        "fg3a": fg3a,
        "ftm": ftm,
        "fta": fta,
    }


def format_ratio(made: Optional[int], att: Optional[int]) -> str:
    # Format made/attempts with a fallback for missing stats.
    if made is None or att is None:
        return "n/a"
    return f"{made}/{att}"


def format_plus_minus(value: Any) -> str:
    # Format plus-minus with a sign for readability.
    pm = coerce_int(value)
    if pm is None:
        return "?"
    return f"{pm:+d}"


def extract_latest_games(html: str, date_str: Optional[str]) -> Tuple[str, List[str]]:
    # Find the latest game date in the HTML and extract its game IDs.
    date_matches = list(re.finditer(r'game_dt:"(\d{4}-\d{2}-\d{2})"', html))
    if not date_matches:
        raise ValueError("No game_dt values found in games page.")

    dates = [m.group(1) for m in date_matches]
    latest = max(dates) if date_str is None else date_str
    if date_str and date_str not in dates:
        raise ValueError(f"Date not found in games page: {date_str}")

    # Find the block containing games for this date.
    target_idx = html.find(f'game_dt:"{latest}"')
    games_marker = "games:["
    games_idx = html.find(games_marker, target_idx)
    if games_idx == -1:
        raise ValueError(f"games:[ not found after date {latest}")
    start = games_idx + len(games_marker) - 1
    end = find_matching_bracket(html, start, "[", "]")
    games_blob = html[start : end + 1]
    game_ids = re.findall(r"game_id:(\d+)", games_blob)
    if not game_ids:
        raise ValueError(f"No game_id values found for date {latest}")
    return latest, game_ids


def format_clock(period: int, clock: str) -> str:
    # Format a clock time with its period.
    return f"Q{period} {clock}"


def calc_quarter_scores(pbp: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int]]:
    # Convert cumulative play-by-play scores into per-quarter splits.
    per_period_scores: Dict[int, Tuple[int, int]] = {}
    for ev in pbp:
        period = ev.get("period")
        home = ev.get("home_score")
        away = ev.get("away_score")
        if period is None or home is None or away is None:
            continue
        per_period_scores[period] = (home, away)
    # Convert cumulative to period scores.
    result: Dict[int, Tuple[int, int]] = {}
    prev_home = 0
    prev_away = 0
    for period in sorted(per_period_scores.keys()):
        home, away = per_period_scores[period]
        result[period] = (home - prev_home, away - prev_away)
        prev_home, prev_away = home, away
    return result


def calc_lead_changes(pbp: List[Dict[str, Any]]) -> int:
    # Count lead changes based on score margin sign changes.
    last_sign = 0
    changes = 0
    for ev in pbp:
        margin = ev.get("home_score_margin")
        if margin is None:
            continue
        sign = 1 if margin > 0 else -1 if margin < 0 else 0
        if last_sign != 0 and sign != 0 and sign != last_sign:
            changes += 1
        if sign != 0:
            last_sign = sign
    return changes


def calc_biggest_run(pbp: List[Dict[str, Any]]) -> Tuple[int, Optional[int]]:
    # Identify the largest continuous scoring run by a single team.
    current_team = None
    current_pts = 0
    best_pts = 0
    best_team = None
    for ev in pbp:
        pts = ev.get("pts")
        if not pts:
            continue
        team_id = ev.get("off_team_id")
        if team_id is None:
            continue
        if team_id != current_team:
            current_team = team_id
            current_pts = 0
        current_pts += pts
        if current_pts > best_pts:
            best_pts = current_pts
            best_team = team_id
    return best_pts, best_team


def calc_biggest_winprob_swing(pbp: List[Dict[str, Any]]) -> Tuple[float, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    # Find the biggest win probability swing between consecutive events.
    last = None
    best_delta = 0.0
    best_pair = (None, None)
    for ev in pbp:
        wp = ev.get("p_win_prob")
        if wp is None:
            continue
        if last is not None:
            delta = wp - last.get("p_win_prob", 0.0)
            if abs(delta) > abs(best_delta):
                best_delta = delta
                best_pair = (last, ev)
        last = ev
    return best_delta, best_pair[0], best_pair[1]


def team_name_from_id(game_info: Dict[str, Any], team_id: int) -> str:
    # Resolve a team name from the game metadata.
    if team_id == game_info.get("home_team_id"):
        return game_info.get("home_full_name", "Home")
    if team_id == game_info.get("away_team_id"):
        return game_info.get("away_full_name", "Away")
    return "Unknown"


def summarize_game_basic(game_info: Dict[str, Any], box: List[Dict[str, Any]], pbp: List[Dict[str, Any]]) -> str:
    # Build a deterministic summary when LLM output is disabled.
    home = game_info["home_full_name"]
    away = game_info["away_full_name"]

    quarter_scores = calc_quarter_scores(pbp)
    lead_changes = calc_lead_changes(pbp)
    run_pts, run_team = calc_biggest_run(pbp)
    swing_delta, swing_start, swing_end = calc_biggest_winprob_swing(pbp)

    # Build highlights from box score.
    players_by_team: Dict[int, List[Dict[str, Any]]] = {}
    for row in box:
        if row.get("mp", 0) and row.get("team_id") is not None:
            players_by_team.setdefault(row["team_id"], []).append(row)

    def top_n(team_id: int, key: str, n: int = 2) -> List[Dict[str, Any]]:
        # Pull the top N performers for a given stat.
        return sorted(players_by_team.get(team_id, []), key=lambda r: r.get(key, 0), reverse=True)[:n]

    def fmt_player(row: Dict[str, Any]) -> str:
        # Format a detailed stat line for a highlight player.
        shooting = extract_shooting_stats(row)
        stat_parts = [
            f'{row.get("pts", 0)} pts',
            f'{row.get("reb", 0)} reb',
            f'{row.get("ast", 0)} ast',
            f'TS {row.get("tspct", "?")}',
            f'FG {format_ratio(shooting["fgm"], shooting["fga"])}',
            f'3P {format_ratio(shooting["fg3m"], shooting["fg3a"])}',
            f'FT {format_ratio(shooting["ftm"], shooting["fta"])}',
            f'{row.get("blk", 0)} blk',
            f'{row.get("stl", 0)} stl',
            f'{row.get("tov", 0)} tov',
            f'+/- {format_plus_minus(row.get("plus_minus"))}',
        ]
        return f'{row.get("player_name","?")} ({", ".join(stat_parts)})'

    home_id = game_info["home_team_id"]
    away_id = game_info["away_team_id"]
    home_top_pts = top_n(home_id, "pts", 1)
    away_top_pts = top_n(away_id, "pts", 1)
    home_top_ast = top_n(home_id, "ast", 1)
    away_top_ast = top_n(away_id, "ast", 1)
    home_top_reb = top_n(home_id, "reb", 1)
    away_top_reb = top_n(away_id, "reb", 1)

    swing_team = None
    if swing_delta != 0 and swing_start and swing_end:
        swing_team = home if swing_delta > 0 else away
        swing_time = format_clock(swing_end.get("period", "?"), swing_end.get("game_clock", "?"))
        swing_text = f"largest win-prob swing: {swing_team} {swing_delta:+.2f} at {swing_time}"
    else:
        swing_text = "largest win-prob swing: n/a"

    run_text = "largest run: n/a"
    if run_team and run_pts:
        run_text = f"largest run: {team_name_from_id(game_info, run_team)} {run_pts}-0"

    quarter_parts = []
    for period in sorted(quarter_scores.keys()):
        hs, as_ = quarter_scores[period]
        quarter_parts.append(f"Q{period} {away} {as_}-{hs} {home}")

    lines = []
    lines.append("- Momentum")
    lines.append(f"  - {swing_text}")
    lines.append(f"  - {run_text}; lead changes: {lead_changes}")
    if quarter_parts:
        for quarter_line in quarter_parts:
            lines.append(f"  - {quarter_line}")
    else:
        lines.append("  - Quarter splits: n/a")
    lines.append("- Player highlights")
    if home_top_pts:
        p = home_top_pts[0]
        lines.append(f"  - {home}: {fmt_player(p)}")
    if away_top_pts:
        p = away_top_pts[0]
        lines.append(f"  - {away}: {fmt_player(p)}")
    if home_top_ast:
        p = home_top_ast[0]
        lines.append(f"  - {home} playmaking: {fmt_player(p)}")
    if away_top_ast:
        p = away_top_ast[0]
        lines.append(f"  - {away} playmaking: {fmt_player(p)}")
    if home_top_reb:
        p = home_top_reb[0]
        lines.append(f"  - {home} glass: {fmt_player(p)}")
    if away_top_reb:
        p = away_top_reb[0]
        lines.append(f"  - {away} glass: {fmt_player(p)}")
    return "\n".join(lines)


def get_yesterday_date(tz_name: str) -> str:
    # Get yesterday's date in the requested timezone.
    if ZoneInfo is not None:
        now = datetime.now(ZoneInfo(tz_name))
    else:
        now = datetime.now()
    return (now.date() - timedelta(days=1)).isoformat()


def build_azure_client() -> Tuple[Any, str]:
    # Build the Azure OpenAI client after validating env configuration.
    missing = [key for key in ENV_KEYS if not os.getenv(key)]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing Azure OpenAI config in .env or environment: {missing_text}")
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install the openai package (pip install openai).") from exc
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    return client, os.environ["AZURE_OPENAI_DEPLOYMENT"]


def scoring_plays(pbp: List[Dict[str, Any]], away: str, home: str, limit: int) -> List[Dict[str, Any]]:
    # Collect key scoring plays from play-by-play.
    plays: List[Dict[str, Any]] = []
    for ev in pbp:
        pts = ev.get("pts")
        if not pts:
            continue
        desc = ev.get("dsc") or ev.get("description")
        if not desc:
            continue
        period = ev.get("period")
        clock = ev.get("game_clock")
        away_score = ev.get("away_score")
        home_score = ev.get("home_score")
        plays.append(
            {
                "time": format_clock(period, clock) if period and clock else None,
                "score": f"{away} {away_score} - {home} {home_score}"
                if away_score is not None and home_score is not None
                else None,
                "description": desc,
            }
        )
    if limit and len(plays) > limit:
        head = plays[: max(1, limit // 2)]
        tail = plays[-(limit - len(head)) :]
        plays = head + tail
    return plays


def winprob_swings(
    pbp: List[Dict[str, Any]], away: str, home: str, limit: int = 5
) -> List[Dict[str, Any]]:
    # Rank the largest win probability changes for context.
    swings: List[Tuple[float, float, Dict[str, Any]]] = []
    last = None
    for ev in pbp:
        wp = ev.get("p_win_prob")
        if wp is None:
            continue
        if last is not None and last.get("p_win_prob") is not None:
            delta = wp - last["p_win_prob"]
            swings.append((abs(delta), delta, ev))
        last = ev
    swings.sort(reverse=True, key=lambda item: item[0])
    result: List[Dict[str, Any]] = []
    for _, delta, ev in swings[:limit]:
        period = ev.get("period")
        clock = ev.get("game_clock")
        away_score = ev.get("away_score")
        home_score = ev.get("home_score")
        result.append(
            {
                "time": format_clock(period, clock) if period and clock else None,
                "delta": round(delta, 3),
                "score": f"{away} {away_score} - {home} {home_score}"
                if away_score is not None and home_score is not None
                else None,
                "description": ev.get("dsc") or ev.get("description"),
            }
        )
    return result


def build_llm_payload(
    game_info: Dict[str, Any],
    box: List[Dict[str, Any]],
    pbp: List[Dict[str, Any]],
    max_pbp: int,
) -> Dict[str, Any]:
    # Build a structured payload for the LLM to summarize.
    home = game_info["home_full_name"]
    away = game_info["away_full_name"]
    home_score = game_info["home_score"]
    away_score = game_info["away_score"]

    quarter_scores = calc_quarter_scores(pbp)
    lead_changes = calc_lead_changes(pbp)
    run_pts, run_team = calc_biggest_run(pbp)
    swing_delta, swing_start, swing_end = calc_biggest_winprob_swing(pbp)

    players_by_team: Dict[int, List[Dict[str, Any]]] = {}
    for row in box:
        if row.get("mp", 0) and row.get("team_id") is not None:
            players_by_team.setdefault(row["team_id"], []).append(row)

    def top_n(team_id: int, key: str, n: int = 5) -> List[Dict[str, Any]]:
        # Select top contributors for the requested stat.
        return sorted(players_by_team.get(team_id, []), key=lambda r: r.get(key) if r.get(key) is not None else float('-inf'), reverse=True)[:n]

    def pack_player(row: Dict[str, Any]) -> Dict[str, Any]:
        # Include full stat lines for highlight players.
        shooting = extract_shooting_stats(row)
        return {
            "player": row.get("player_name"),
            "pts": row.get("pts"),
            "reb": row.get("reb"),
            "ast": row.get("ast"),
            "tspct": row.get("tspct"),
            "fgm": shooting["fgm"],
            "fga": shooting["fga"],
            "fg3m": shooting["fg3m"],
            "fg3a": shooting["fg3a"],
            "ftm": shooting["ftm"],
            "fta": shooting["fta"],
            "blk": row.get("blk"),
            "stl": row.get("stl"),
            "tov": row.get("tov"),
            "plus_minus": row.get("plus_minus"),
        }

    home_id = game_info["home_team_id"]
    away_id = game_info["away_team_id"]

    quarter_lines = []
    for period in sorted(quarter_scores.keys()):
        hs, as_ = quarter_scores[period]
        quarter_lines.append(f"Q{period} {away} {as_}-{hs} {home}")

    swing_text = None
    if swing_delta != 0 and swing_start and swing_end:
        swing_team = home if swing_delta > 0 else away
        swing_time = format_clock(swing_end.get("period", "?"), swing_end.get("game_clock", "?"))
        swing_text = f"{swing_team} {swing_delta:+.2f} at {swing_time}"

    return {
        "matchup": f"{away} at {home}",
        "final_score": f"{away} {away_score}, {home} {home_score}",
        "momentum_signals": {
            "quarter_splits": quarter_lines,
            "lead_changes": lead_changes,
            "largest_run": f"{team_name_from_id(game_info, run_team)} {run_pts}-0" if run_team and run_pts else None,
            "largest_winprob_swing": swing_text,
            "winprob_swings": winprob_swings(pbp, away, home, limit=5),
        },
        "top_players": {
            home: [pack_player(row) for row in top_n(home_id, "p_epm")],
            away: [pack_player(row) for row in top_n(away_id, "p_epm")],
        },
        "scoring_plays": scoring_plays(pbp, away, home, max_pbp),
    }


def summarize_game_llm(
    client: Any,
    deployment: str,
    game_info: Dict[str, Any],
    box: List[Dict[str, Any]],
    pbp: List[Dict[str, Any]],
    max_pbp: int,
) -> str:
    # Use the LLM to generate a readable summary from structured stats.
    logger.debug(f"Building LLM payload for {game_info.get('away_full_name')} at {game_info.get('home_full_name')}")
    payload = build_llm_payload(game_info, box, pbp, max_pbp)
    system = (
        "You are an expert NBA analyst. Use only the provided data to summarize momentum and player highlights."
    )
    instructions = (
        "Write a comprehensive recap that mentions the final score and both teams. Output markdown with two sections:\n"
        "- Momentum: up to 10 sentences that include quarter-by-quarter scoring details from quarter_splits, plus swings "
        "and runs\n"
        "- Player highlights: up to 10 sentences that include points, rebounds, assists, TS, FG x/y, 3P x/y, FT x/y, "
        "blocks, steals, turnovers, and plus-minus for key players\n"
        "Think in English and respond in Simplified Chinese."
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": instructions + "\n\nData:\n" + json.dumps(payload, ensure_ascii=True, indent=2)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def parse_game_page(html: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Extract embedded game data objects from the HTML payload.
    game_info_obj = extract_object(html, "game_info:{")
    box_arr = extract_array(html, "box:[")
    pbp_arr = extract_array(html, "pbp:[")
    game_info = parse_js_object(game_info_obj)
    box = parse_js_array(box_arr)
    pbp = parse_js_array(pbp_arr)
    logger.debug(f"Parsed game page: box={len(box)}, pbp={len(pbp)}")
    return game_info, box, pbp


def format_game_header(game_info: Dict[str, Any]) -> List[str]:
    # Build header lines with matchup and final score.
    home = game_info["home_full_name"]
    away = game_info["away_full_name"]
    home_score = game_info["home_score"]
    away_score = game_info["away_score"]
    return [
        f"## {away} at {home} ({away_score}-{home_score})",
        f"Matchup: {away} vs {home}",
    ]


def main() -> int:
    # Orchestrate fetching, summarization, and output.
    parser = argparse.ArgumentParser(description="Summarize latest Dunks & Threes games.")
    parser.add_argument("--date", help="YYYY-MM-DD date override")
    parser.add_argument("--games-html", help="Path to cached /games HTML")
    parser.add_argument("--game-html-dir", help="Directory of cached game HTML files")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--tz", default=DEFAULT_TZ, help="Timezone for yesterday date calculation")
    parser.add_argument("--max-pbp", type=int, default=80, help="Max play-by-play scoring events to include")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM summarization")
    parser.add_argument("--output", help="Write report to a file instead of stdout")
    args = parser.parse_args()

    logger.info("Starting report generation")

    env_loaded = load_dotenv(args.env_file, override=False)
    if env_loaded:
        logger.info(f"Loaded environment from {args.env_file}")
    else:
        logger.info(f"No .env loaded from {args.env_file}")

    if args.games_html:
        logger.info(f"Reading cached games HTML from {args.games_html}")
        with open(args.games_html, "r", encoding="utf-8", errors="replace") as f:
            games_html = f.read()
    else:
        logger.info(f"Fetching games list from {args.base_url}")
        games_html = fetch_url(f"{args.base_url}/games")

    target_date = args.date or get_yesterday_date(args.tz)
    date_str, game_ids = extract_latest_games(games_html, target_date)
    logger.info(f"Found {len(game_ids)} games for {date_str}")

    out_lines = [f"# NBA Games Report ({date_str})", ""]

    client = None
    deployment = None
    if not args.no_llm:
        try:
            logger.info("Initializing Azure OpenAI client")
            client, deployment = build_azure_client()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    else:
        logger.info("Skipping LLM summarization (--no-llm)")

    for game_id in game_ids:
        logger.info(f"Processing game {game_id}")
        if args.game_html_dir:
            path = f"{args.game_html_dir.rstrip('/')}/game_{game_id}.html"
            logger.info(f"Reading cached game HTML from {path}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                game_html = f.read()
        else:
            game_html = fetch_url(f"{args.base_url}/games/{game_id}")

        game_info, box, pbp = parse_game_page(game_html)
        out_lines.extend(format_game_header(game_info))
        out_lines.append("")

        logger.info(f"game_info = \n{game_info}")
        logger.info(f"box = \n{box[:5]}")
        logger.info(f"pbp = \n{pbp[:5]}")

        if client and deployment:
            out_lines.append(summarize_game_llm(client, deployment, game_info, box, pbp, args.max_pbp))
        else:
            out_lines.append(summarize_game_basic(game_info, box, pbp))
        out_lines.append("")

    report = "\n".join(out_lines).rstrip()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(report)
        logger.info(f"Wrote report to {args.output}")
    else:
        print(report)
        logger.info("Report written to stdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

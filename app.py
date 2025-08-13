from fastapi import FastAPI, Header, HTTPException
from typing import Dict, Optional
import math
import os

app = FastAPI(title="Betting Brain API", version="1.0")

# Ruta raíz para comprobar que está vivo
@app.get("/")
def root():
    return {"status": "ok", "service": "betting-brain-api"}

# 🔐 Seguridad: API KEY (configúrala en Render como variable de entorno API_KEY)
API_KEY = os.getenv("API_KEY", "SUPER_SECRETA_123")

def check_auth(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# 📊 Ratings ELO de ejemplo
ELO = {
    # Fútbol
    "Real Madrid": 1845, "Barcelona": 1835, "Osasuna": 1650, "Valencia": 1705,
    "PSG": 1850, "Chelsea": 1820,
    # NBA
    "Lakers": 1825, "Warriors": 1805, "Celtics": 1880, "Heat": 1810,
}
HOME_ADV_FOOT = 60   # ventaja de local fútbol
HOME_ADV_NBA  = 80   # ventaja de local NBA

# 🗓️ FIXTURES de ejemplo
FIXTURES = [
    {"match_id":"F001","league":"LaLiga","sport":"football","home":"Real Madrid","away":"Barcelona","kickoff_utc":"2025-08-14T19:00:00Z"},
    {"match_id":"N101","league":"NBA","sport":"basketball","home":"Lakers","away":"Warriors","kickoff_utc":"2025-08-14T20:00:00Z"},
    {"match_id":"F002","league":"Supercopa","sport":"football","home":"PSG","away":"Chelsea","kickoff_utc":"2025-08-13T20:00:00Z"},
]

# 💸 ODDS de ejemplo (cámbialas por las reales cuando quieras)
ODDS = {
    ("F001","1X2"): {
        "best_book": "BookA",
        "selections": [
            {"selection":"Home","decimal_odds":1.95},
            {"selection":"Draw","decimal_odds":3.35},
            {"selection":"Away","decimal_odds":4.00},
        ]
    },
    ("N101","moneyline"): {
        "best_book":"BookA",
        "selections":[
            {"selection":"Home","decimal_odds":1.80},
            {"selection":"Away","decimal_odds":2.10},
        ]
    },
    ("F002","1X2"): {
        "best_book": "BookB",
        "selections": [
            {"selection":"Home","decimal_odds":2.10},  # PSG
            {"selection":"Draw","decimal_odds":3.40},
            {"selection":"Away","decimal_odds":3.25},  # Chelsea
        ]
    },
}

# =============================
# FUNCIONES DEL MODELO
# =============================

def elo_winprob(home: str, away: str, sport: str) -> float:
    elo_h = ELO.get(home, 1700)
    elo_a = ELO.get(away, 1700)
    adv   = HOME_ADV_FOOT if sport == "football" else HOME_ADV_NBA
    diff  = (elo_h + adv) - elo_a
    # logística clásica ELO
    p_home = 1.0 / (1.0 + math.pow(10.0, -diff/400.0))
    return p_home

def split_draw_for_football(p_home: float, p_away_est: float) -> Dict[str, float]:
    # Empate según equilibrio (entre 0.15 y 0.30 aprox.)
    base_draw = 0.22
    closeness = 1.0 - abs(p_home - p_away_est)  # 0..1
    p_draw = min(max(base_draw + 0.08 * (closeness - 0.5), 0.15), 0.30)
    # Re-normaliza
    scale = (1.0 - p_draw) / (p_home + p_away_est)
    return {
        "home": p_home * scale,
        "draw": p_draw,
        "away": p_away_est * scale
    }

def implied_prob(odds: float) -> float:
    return 1.0 / odds

def kelly_fraction(p: float, odds: float, k: float = 0.25) -> float:
    b = odds - 1.0
    f_star = ((b * p) - (1.0 - p)) / b if b > 0 else 0.0
    return max(f_star, 0.0) * k

# =============================
# ENDPOINTS
# =============================

@app.get("/fixtures")
def get_fixtures(date: str = "", league: str = ""):
    out = FIXTURES
    if league:
        out = [m for m in out if m["league"].lower() == league.lower()]
    return out

@app.get("/odds")
def get_odds(match_id: str, market: str):
    key = (match_id, market)
    if key not in ODDS:
        raise HTTPException(status_code=404, detail="Odds not found for that match/market")
    return ODDS[key]

@app.get("/predict")
def predict(
    match_id: str,
    market: str,
    bank: float = 5000.0,
    edge_min: float = 0.0,  # 👈 ahora puedes filtrar si quieres (por defecto 0.0 = mostrar todo)
    authorization: Optional[str] = Header(default=None)
):
    check_auth(authorization)

    match = next((m for m in FIXTURES if m["match_id"] == match_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    odds_info = ODDS.get((match_id, market))
    if not odds_info:
        raise HTTPException(status_code=404, detail="Odds not found for that market")

    sport = match["sport"]
    home  = match["home"]
    away  = match["away"]

    picks = []
    if sport == "football" and market == "1X2":
        p_home_raw = elo_winprob(home, away, sport)
        p_away_raw = 1.0 - p_home_raw
        probs = split_draw_for_football(p_home_raw, p_away_raw)  # dict con home/draw/away

        for s in odds_info["selections"]:
            sel = s["selection"]
            odds = s["decimal_odds"]
            p_model = probs["home"] if sel == "Home" else probs["draw"] if sel == "Draw" else probs["away"]
            p_implied = implied_prob(odds)
            edge = p_model - p_implied
            stake = bank * kelly_fraction(p_model, odds, 0.25)
            min_value_odds = 1.0 / p_model
            picks.append({
                "selection": sel,
                "prob_model": round(p_model, 4),
                "prob_implied": round(p_implied, 4),
                "edge": round(edge, 4),
                "stake_recommended": round(stake, 2),
                "min_value_odds": round(min_value_odds, 3),
                "reasons": [
                    f"ELO {home} vs {away} con ventaja local",
                    "Empate asignado según equilibrio ELO",
                    "Kelly fraccional 0.25×"
                ]
            })

    elif sport == "basketball" and market == "moneyline":
        p_home = elo_winprob(home, away, sport)
        p_away = 1.0 - p_home
        for s in odds_info["selections"]:
            sel = s["selection"]
            odds = s["decimal_odds"]
            p_model = p_home if sel == "Home" else p_away
            p_implied = implied_prob(odds)
            edge = p_model - p_implied
            stake = bank * kelly_fraction(p_model, odds, 0.25)
            min_value_odds = 1.0 / p_model
            picks.append({
                "selection": sel,
                "prob_model": round(p_model, 4),
                "prob_implied": round(p_implied, 4),
                "edge": round(edge, 4),
                "stake_recommended": round(stake, 2),
                "min_value_odds": round(min_value_odds, 3),
                "reasons": [
                    f"ELO {home} vs {away} con ventaja local",
                    "Modelo ML por ELO",
                    "Kelly fraccional 0.25×"
                ]
            })
    else:
        raise HTTPException(status_code=400, detail="Market/sport not implemented in demo")

    # 🔎 Ahora el filtro es configurable: por defecto edge_min=0.0 (muestra todo lo ≥ 0)
    picks = [p for p in picks if p["edge"] >= edge_min]
    # Ordena por edge descendente para leer mejor
    picks.sort(key=lambda x: x["edge"], reverse=True)

    return {"picks": picks}

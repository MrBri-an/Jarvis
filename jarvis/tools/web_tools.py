"""Web tools: DuckDuckGo search (no key) and Open-Meteo weather (no key)."""
from __future__ import annotations

import html
import re

import requests

from .registry import ToolRegistry, ToolResult

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) JARVIS/0.1"}


def register_web_tools(reg: ToolRegistry) -> None:
    reg.tool(
        "web_search",
        "Search the web and return top results (title + snippet). Use for facts, news, lookups.",
        {"query": {"type": "string"}},
        ["query"],
    )(_web_search)

    reg.tool(
        "get_weather",
        "Current weather + today's forecast for a city.",
        {"location": {"type": "string", "description": "City name, e.g. 'London' or 'Austin'"}},
        ["location"],
    )(_weather)


def _web_search(query: str) -> ToolResult:
    try:
        r = requests.get(
            "https://html.duckduckgo.com/html/", params={"q": query},
            headers=_UA, timeout=10,
        )
        r.raise_for_status()
        results = re.findall(
            r'class="result__a"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
            r.text, re.S,
        )
        if not results:
            raise ValueError("no results parsed")
        out = []
        for title, snippet in results[:5]:
            t = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
            s = html.unescape(re.sub(r"<[^>]+>", "", snippet)).strip()
            out.append(f"- {t}: {s}")
        return ToolResult("done", "search results for "
                          f"'{query}':\n" + "\n".join(out))
    except Exception:
        import webbrowser

        webbrowser.open(f"https://duckduckgo.com/?q={requests.utils.quote(query)}")
        return ToolResult("unverified",
                          "couldn't fetch search results; opened the search in the browser instead",
                          speak="I couldn't search directly, so I opened the results in your browser")


def _weather(location: str) -> ToolResult:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1}, headers=_UA, timeout=10,
    ).json()
    if not geo.get("results"):
        return ToolResult("failed", f"unknown location '{location}'")
    g = geo["results"][0]
    w = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": g["latitude"], "longitude": g["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "forecast_days": 2,
        }, headers=_UA, timeout=10,
    ).json()
    cur = w["current"]
    day = w["daily"]
    codes = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
             45: "fog", 51: "light drizzle", 61: "light rain", 63: "rain",
             65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow",
             80: "rain showers", 95: "thunderstorm"}
    cond = codes.get(cur["weather_code"], "mixed")
    text = (
        f"Weather in {g['name']}, {g.get('country', '')}: {cur['temperature_2m']}°F, "
        f"{cond}, feels like {cur['apparent_temperature']}°F, wind {cur['wind_speed_10m']} mph, "
        f"humidity {cur['relative_humidity_2m']}%. "
        f"Today: {day['temperature_2m_min'][0]}–{day['temperature_2m_max'][0]}°F, "
        f"{day['precipitation_probability_max'][0]}% chance of precipitation."
    )
    return ToolResult("done", text)

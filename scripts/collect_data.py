#!/usr/bin/env python3
"""
Collect beatmaps and replays from osu! official site (osu.ppy.sh).

Authentication:
  - API calls (search, scores):  OAuth2 client credentials (no user login)
  - Downloads (.osz, .osr):      osu_session cookie (from browser)

Endpoints used:
  - API: GET /api/v2/beatmapsets/search         (search beatmapsets)
  - API: GET /api/v2/beatmaps/{id}/scores       (list scores)
  - Web: GET /beatmapsets/{id}/download          (download .osz, cookie auth)
  - Web: GET /scores/{id}/download               (download .osr, cookie auth)

Prerequisites:
  1. Register an OAuth application at https://osu.ppy.sh/home/account/edit
     -> "New OAuth Application" (any name, any callback URL).
     Note down Client ID and Client Secret.

  2. Get your osu_session cookie:
     - Open https://osu.ppy.sh in your browser and log in
     - Press F12 -> Application -> Cookies -> osu.ppy.sh
     - Copy the value of the ``osu_session`` cookie

   3. Put credentials in your config YAML (pass with -c)::

       osu_api:
         client_id: 12345
         client_secret: "your_secret_here"
         osu_session: "eyJpdiI6...%3D"    # from browser F12

Usage::

    python scripts/collect_data.py --count 50
    python scripts/collect_data.py --stars 3.0 7.0 --count 100 --data raw_data
    python scripts/collect_data.py --count 20 --dry-run
    python scripts/collect_data.py --replays-only --data raw_data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml

try:
    import requests
except ImportError:
    print("ERROR: requests is required: pip install requests")
    sys.exit(1)


# ── Browser-like headers for web endpoints ───────────────────────────

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0"
)


# ── osu! client ─────────────────────────────────────────────────────

class OsuClient:
    """osu! client: OAuth for API, session cookie for web downloads.

    Endpoints:
      - API (OAuth Bearer):
          GET /api/v2/beatmapsets/search
          GET /api/v2/beatmaps/{id}/scores
      - Web (osu_session cookie):
          GET /beatmapsets/{id}/download   -> .osz
          GET /scores/{id}/download        -> .osr
    """

    API_BASE = "https://osu.ppy.sh/api/v2"
    WEB_BASE = "https://osu.ppy.sh"
    TOKEN_URL = "https://osu.ppy.sh/oauth/token"

    def __init__(
        self,
        client_id: int,
        client_secret: str,
        osu_session: str,
        requests_per_minute: int = 60,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.osu_session = osu_session

        # Separate sessions for API and web (different auth mechanisms)
        self._api_session = requests.Session()
        self._web_session = requests.Session()

        # Configure web session with browser-like headers + cookie
        self._web_session.headers.update({
            "User-Agent": _BROWSER_UA,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,image/apng,*/*;"
                "q=0.8,application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        })
        self._web_session.cookies.set(
            "osu_session", osu_session,
            domain=".osu.ppy.sh", path="/",
        )

        self._token: Optional[str] = None
        self._token_expires: float = 0.0
        self._min_interval = 60.0 / requests_per_minute
        self._last_request: float = 0.0

    # ── OAuth (for API calls only) ────────────────────────────────

    def _ensure_api_token(self) -> None:
        """Get or refresh OAuth client credentials token for API calls."""
        if self._token and time.time() < self._token_expires - 60:
            return

        resp = self._api_session.post(self.TOKEN_URL, json={
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
            "scope": "public",
        })
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 86400)

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    # ── API requests (OAuth Bearer) ───────────────────────────────

    def _api_get(self, path: str, params: Optional[Dict] = None) -> requests.Response:
        """Authenticated GET to /api/v2/..."""
        self._ensure_api_token()
        self._throttle()
        resp = self._api_session.get(
            f"{self.API_BASE}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30,
        )
        return resp

    # ── Web requests (session cookie) ─────────────────────────────

    def _web_get(
        self,
        path: str,
        referer: str = "",
        timeout: int = 120,
    ) -> requests.Response:
        """GET to osu.ppy.sh/... with session cookie + browser headers."""
        self._throttle()
        headers: Dict[str, str] = {}
        if referer:
            headers["Referer"] = referer
        resp = self._web_session.get(
            f"{self.WEB_BASE}{path}",
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        return resp

    def _web_xhr(
        self,
        path: str,
        referer: str = "",
        timeout: int = 30,
    ) -> requests.Response:
        """XHR-style GET (JSON) to osu.ppy.sh with session cookie.

        Used for endpoints like /beatmaps/{id}/scores that expect
        x-requested-with and return JSON.
        """
        self._throttle()
        # XSRF token: Laravel reads X-XSRF-TOKEN header, which is the
        # URL-decoded value of the XSRF-TOKEN cookie set by the server.
        import urllib.parse
        xsrf_cookie = self._web_session.cookies.get("XSRF-TOKEN", domain=".osu.ppy.sh")
        xsrf = urllib.parse.unquote(xsrf_cookie) if xsrf_cookie else ""

        headers: Dict[str, str] = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
        }
        if xsrf:
            headers["X-XSRF-TOKEN"] = xsrf
        if referer:
            headers["Referer"] = referer

        resp = self._web_session.get(
            f"{self.WEB_BASE}{path}",
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        return resp

    # ── Beatmapset search (API) ───────────────────────────────────

    def search_beatmapsets(
        self,
        star_min: float = 4.0,
        star_max: float = 6.0,
        status: str = "ranked",
        mode: int = 0,
        cursor_string: str = "",
        sort: str = "plays_desc",
    ) -> Tuple[List[Dict], str]:
        """Search for beatmapsets.  Returns (beatmapsets, next_cursor)."""
        params: Dict[str, Any] = {
            "m": mode,
            "s": status,
            "sort": sort,
        }
        if star_min > 0 or star_max < 99:
            q_parts = []
            if star_min > 0:
                q_parts.append(f"stars>={star_min:.1f}")
            if star_max < 99:
                q_parts.append(f"stars<{star_max:.1f}")
            params["q"] = " ".join(q_parts)
        if cursor_string:
            params["cursor_string"] = cursor_string

        resp = self._api_get("/beatmapsets/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("beatmapsets", []), data.get("cursor_string", "") or ""

    # ── Beatmap scores (Web XHR, NM filter) ─────────────────────────

    def get_beatmap_scores(
        self,
        beatmap_id: int,
        beatmapset_id: int = 0,
    ) -> List[Dict]:
        """Get NoMod scores for a beatmap via web endpoint.

        Uses GET /beatmaps/{id}/scores?mode=osu&mods[]=NM&type=global
        which is the same endpoint the osu! website uses.
        """
        referer = (
            f"{self.WEB_BASE}/beatmapsets/{beatmapset_id}"
            if beatmapset_id else ""
        )
        path = f"/beatmaps/{beatmap_id}/scores?mode=osu&mods%5B%5D=NM&type=global"
        resp = self._web_xhr(path, referer=referer)
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        return data.get("scores", [])

    # ── .osz download (Web, cookie) ───────────────────────────────

    def download_osz(self, beatmapset_id: int) -> Optional[bytes]:
        """Download .osz via GET /beatmapsets/{id}/download (cookie auth)."""
        referer = f"{self.WEB_BASE}/beatmapsets/{beatmapset_id}"
        resp = self._web_get(
            f"/beatmapsets/{beatmapset_id}/download",
            referer=referer,
            timeout=120,
        )
        if resp.status_code == 200 and len(resp.content) > 100:
            if resp.content[:2] == b"PK":
                return resp.content
        return None

    # ── .osr download (Web, cookie) ───────────────────────────────

    def download_replay(self, score_id: int, referer_set_id: int = 0) -> Optional[bytes]:
        """Download .osr via GET /scores/{id}/download (cookie auth)."""
        referer = f"{self.WEB_BASE}/beatmapsets/{referer_set_id}" if referer_set_id else ""
        resp = self._web_get(
            f"/scores/{score_id}/download",
            referer=referer,
            timeout=60,
        )
        if resp.status_code == 200 and len(resp.content) > 100:
            return resp.content
        return None

    # ── Session validity check ────────────────────────────────────

    def check_session(self, verbose: bool = False) -> bool:
        """Verify osu_session cookie is valid by fetching /home.

        Returns True if session is valid (user is logged in).
        """
        self._throttle()
        # Follow all redirects; if session is invalid osu! lands on login page
        resp = self._web_session.get(
            f"{self.WEB_BASE}/home",
            timeout=15,
            allow_redirects=True,
        )
        final_url = resp.url
        if verbose:
            print()
            cookie_val = self.osu_session
            print(f"    cookie length: {len(cookie_val)}")
            print(f"    cookie start:  {cookie_val[:30]}...")
            print(f"    cookie end:    ...{cookie_val[-20:]}")
            print(f"    status:        {resp.status_code}")
            print(f"    final url:     {final_url}")

        # If we ended up on a login page, session is invalid
        if resp.status_code == 200 and "/login" not in final_url:
            return True
        return False


# ── Main collection logic ───────────────────────────────────────────

def collect(
    client: OsuClient,
    data_dir: Path,
    target_beatmapsets: int = 50,
    star_min: float = 4.0,
    star_max: float = 6.0,
    max_replays_per_beatmap: int = 3,
    dry_run: bool = False,
    replays_only: bool = False,
) -> Dict[str, int]:
    """
    Main collection loop.

    1. Search ranked beatmapsets in the given star range (API)
    2. Download .osz from osu.ppy.sh (web, cookie)
    3. For each std beatmap, find top NoMod scores with replays (API)
    4. Download replays from osu.ppy.sh (web, cookie)
    """
    beatmaps_dir = data_dir / "beatmaps"
    replays_dir = data_dir / "replays"
    beatmaps_dir.mkdir(parents=True, exist_ok=True)
    replays_dir.mkdir(parents=True, exist_ok=True)

    existing_osz = {p.stem for p in beatmaps_dir.glob("*.osz")}
    existing_osr = {p.stem for p in replays_dir.glob("*.osr")}

    stats = {
        "beatmapsets_found": 0,
        "beatmapsets_downloaded": 0,
        "beatmapsets_skipped": 0,
        "replays_downloaded": 0,
        "replays_failed": 0,
        "replays_skipped": 0,
    }

    cursor = ""
    collected = 0

    print(f"\nSearching for ranked osu!std beatmapsets ({star_min:.1f}-{star_max:.1f}*)...")
    print(f"Target: {target_beatmapsets} beatmapsets\n")

    while collected < target_beatmapsets:
        try:
            beatmapsets, cursor = client.search_beatmapsets(
                star_min=star_min,
                star_max=star_max,
                cursor_string=cursor,
            )
        except requests.HTTPError as e:
            print(f"  API error during search: {e}")
            break

        if not beatmapsets:
            print("  No more beatmapsets found.")
            break

        for bset in beatmapsets:
            if collected >= target_beatmapsets:
                break

            set_id = bset["id"]
            artist = bset.get("artist", "Unknown")
            title = bset.get("title", "Unknown")
            label = f"{set_id} {artist} - {title}"
            stats["beatmapsets_found"] += 1

            # Filter: only sets with osu!std beatmaps in star range
            std_maps = [
                bm for bm in bset.get("beatmaps", [])
                if bm.get("mode") == "osu"
                and star_min <= bm.get("difficulty_rating", 0) <= star_max
            ]
            if not std_maps:
                continue

            print(f"  [{collected+1}/{target_beatmapsets}] {label}")
            print(f"    {len(std_maps)} std beatmap(s) in range")

            # ── Download .osz ─────────────────────────────────────
            osz_name = f"{set_id} {artist} - {title}"
            osz_name = "".join(
                c if c.isalnum() or c in " -_()" else "_"
                for c in osz_name
            )

            if not replays_only:
                osz_path = beatmaps_dir / f"{osz_name}.osz"
                if osz_path.stem in existing_osz or osz_path.exists():
                    print(f"    .osz already exists, skipping")
                    stats["beatmapsets_skipped"] += 1
                elif dry_run:
                    print(f"    [DRY RUN] Would download .osz")
                else:
                    print(f"    Downloading .osz...", end=" ", flush=True)
                    osz_data = client.download_osz(set_id)
                    if osz_data:
                        osz_path.write_bytes(osz_data)
                        print(f"OK ({len(osz_data) / 1024 / 1024:.1f} MB)")
                        stats["beatmapsets_downloaded"] += 1
                        existing_osz.add(osz_path.stem)
                    else:
                        print("FAILED (session expired?)")
                        continue

            # ── Download replays for each std beatmap ─────────────
            for bm in std_maps:
                bm_id = bm["id"]
                star = bm.get("difficulty_rating", 0)
                version = bm.get("version", "?")
                print(f"    [{version}] ({star:.2f}*) id={bm_id}")

                try:
                    # Web endpoint already filters NoMod via mods[]=NM
                    scores = client.get_beatmap_scores(bm_id, beatmapset_id=set_id)
                except Exception as e:
                    print(f"      Score fetch failed: {e}")
                    continue

                # Filter: must have a downloadable replay
                scores = [s for s in scores if s.get("has_replay")]

                if not scores:
                    print(f"      No NoMod scores with replays")
                    continue

                scores = scores[:max_replays_per_beatmap]
                print(f"      {len(scores)} NoMod replay(s) available")

                for score in scores:
                    score_id = score["id"]
                    user = score.get("user", {}).get("username", "unknown")
                    osr_name = f"{bm_id}_{score_id}_{user}"
                    osr_name = "".join(
                        c if c.isalnum() or c in " -_()" else "_"
                        for c in osr_name
                    )

                    if osr_name in existing_osr:
                        print(f"        {user}: already exists")
                        stats["replays_skipped"] += 1
                        continue

                    if dry_run:
                        print(f"        {user}: [DRY RUN] would download")
                        continue

                    print(f"        {user}: downloading...", end=" ", flush=True)
                    osr_data = client.download_replay(score_id, referer_set_id=set_id)

                    if osr_data:
                        osr_path = replays_dir / f"{osr_name}.osr"
                        osr_path.write_bytes(osr_data)
                        print(f"OK ({len(osr_data)} bytes)")
                        stats["replays_downloaded"] += 1
                        existing_osr.add(osr_name)
                    else:
                        print("FAILED")
                        stats["replays_failed"] += 1

            collected += 1

        if not cursor:
            print("  Reached end of search results.")
            break

    return stats


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect beatmaps and replays from osu.ppy.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/collect_data.py --count 50
  python scripts/collect_data.py --stars 3.0 7.0 --count 100
  python scripts/collect_data.py --dry-run --count 20
  python scripts/collect_data.py --replays-only

How to get osu_session cookie:
  1. Open https://osu.ppy.sh in browser, log in
  2. F12 -> Application -> Cookies -> osu.ppy.sh
  3. Copy the value of "osu_session"
  4. Paste into your config YAML under osu_api.osu_session
        """,
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to config YAML (for osu_api credentials and data paths)",
    )
    parser.add_argument(
        "--data", "-d", default=None,
        help="Path to raw_data directory (overrides config)",
    )
    parser.add_argument(
        "--count", "-n", type=int, default=50,
        help="Number of beatmapsets to collect (default: 50)",
    )
    parser.add_argument(
        "--stars", nargs=2, type=float, default=None,
        metavar=("MIN", "MAX"),
        help="Star rating range (default: 4.0 6.0)",
    )
    parser.add_argument(
        "--max-replays", type=int, default=3,
        help="Max replays per beatmap difficulty (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--replays-only", action="store_true",
        help="Skip .osz download; only download replays",
    )
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--client-secret", type=str, default=None)
    parser.add_argument(
        "--session", type=str, default=None,
        help="osu_session cookie value (overrides config)",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────
    config: Dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: config not found: {config_path}")
            sys.exit(1)
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    api_cfg = config.get("osu_api", {})

    client_id = args.client_id or api_cfg.get("client_id")
    client_secret = args.client_secret or api_cfg.get("client_secret")
    osu_session = args.session or api_cfg.get("osu_session", "")

    # Clean up cookie value: strip whitespace, handle URL-encoding
    if osu_session:
        osu_session = str(osu_session).strip()
        # If user pasted URL-encoded value (e.g. from curl), decode it
        if "%3D" in osu_session or "%2B" in osu_session or "%2F" in osu_session:
            import urllib.parse
            osu_session = urllib.parse.unquote(osu_session)

    if not client_id or not client_secret:
        print("ERROR: osu! API credentials not configured.")
        print()
        print("Add to your config YAML:")
        print("  osu_api:")
        print("    client_id: 12345")
        print('    client_secret: "your_secret"')
        print('    osu_session: "eyJpdiI6..."')
        print()
        print("Then pass: --config path/to/config.yaml")
        print()
        print("Or pass via CLI:")
        print("  --client-id 12345 --client-secret xxx --session eyJ...")
        print()
        print("Register at: https://osu.ppy.sh/home/account/edit")
        sys.exit(1)

    if not osu_session:
        print("ERROR: osu_session cookie not configured.")
        print()
        print("How to get it:")
        print("  1. Open https://osu.ppy.sh in browser, log in")
        print("  2. F12 -> Application -> Cookies -> osu.ppy.sh")
        print('  3. Copy the value of "osu_session"')
        print("  4. Add to your config YAML:")
        print('       osu_session: "eyJpdiI6..."')
        print("  Or pass via CLI:")
        print("       --session eyJpdiI6...")
        sys.exit(1)

    # ── Resolve paths ─────────────────────────────────────────────
    data_dir = Path(args.data) if args.data else Path(
        config.get("data", {}).get("raw_data_dir", "raw_data")
    )
    star_min, star_max = (args.stars if args.stars else [4.0, 6.0])

    # ── Build client ──────────────────────────────────────────────
    client = OsuClient(
        client_id=int(client_id),
        client_secret=str(client_secret),
        osu_session=str(osu_session),
    )

    # ── Validate session ──────────────────────────────────────────
    print("=" * 60)
    print("AutOsu Data Collector")
    print("=" * 60)
    print(f"  Data directory:    {data_dir}")
    print(f"  Star range:        {star_min:.1f} - {star_max:.1f}")
    print(f"  Target sets:       {args.count}")
    print(f"  Max replays/diff:  {args.max_replays}")
    print(f"  Dry run:           {args.dry_run}")
    print(f"  Replays only:      {args.replays_only}")

    print(f"  Checking session...", end="", flush=True)
    if client.check_session(verbose=True):
        print("  -> OK (logged in)")
    else:
        print("  -> FAILED")
        print()
        print("  osu_session cookie is invalid or expired.")
        print("  Possible causes:")
        print("    1. Value was truncated by YAML — wrap in double quotes")
        print("    2. Cookie expired — get a fresh one from F12")
        print("    3. Cookie was URL-encoded — paste the raw value")
        print()
        print('  In your config YAML, make sure it looks like:')
        print('    osu_session: "eyJpdiI6...In0="')
        print()
        print("  Or pass directly via CLI:")
        print('    --session "eyJpdiI6...In0="')
        sys.exit(1)

    print("=" * 60)

    # ── Run ───────────────────────────────────────────────────────
    try:
        stats = collect(
            client=client,
            data_dir=data_dir,
            target_beatmapsets=args.count,
            star_min=star_min,
            star_max=star_max,
            max_replays_per_beatmap=args.max_replays,
            dry_run=args.dry_run,
            replays_only=args.replays_only,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        stats = {"note": "interrupted"}

    # ── Report ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Collection complete!")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if not args.dry_run and stats.get("beatmapsets_downloaded", 0) > 0:
        print()
        print("Next steps:")
        print(f"  1. python scripts/prepare_data.py --data {data_dir} --verbose")
        print(f"  2. python scripts/generate_dataset.py --data {data_dir}")


if __name__ == "__main__":
    main()

"""Static map model, built once at startup: a walkability grid, a sight-blocking grid, and a
breadth-first flow field toward one goal that is filled in a few thousand cells per decision.

Everything is flat arrays indexed `j * w + i`. The outer ring of cells is always blocked, so
neighbor lookups never need bounds checks.
"""

import json
import math
import os
from array import array

CELL = 2.0  # meters per grid cell
UNSEEN = 65535
STEEP_CM = 130  # a 1 m terrain edge rising more than this cannot be walked (50 deg is 119 cm)
SOLID_ABOVE = 1.5  # a prism top this far above the ground is a wall, not a step
SIGHT_ABOVE = 9.0  # a prism top this far above the ground hides a mech


class Nav:
    def __init__(self, map_dir: str) -> None:
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            world = json.load(f)
        terrain = world["terrain"]
        self.tnx, self.tny = terrain["nx"], terrain["ny"]
        self.min_x, self.min_y = terrain["minX"], terrain["minY"]
        self.heights = array("h")
        with open(os.path.join(map_dir, "v2", terrain["file"]), "rb") as f:
            self.heights.frombytes(f.read())
        if len(self.heights) != self.tnx * self.tny or terrain["step"] != 1.0:
            raise ValueError("terrain grid does not match world.json")

        self.w = self.tnx // 2
        self.h = self.tny // 2
        n = self.w * self.h
        raw = bytearray(n)  # the mech's center cannot be inside these
        self.tall = bytearray(n)  # these block sight
        self._mark_steep(raw)
        for prism in world["prisms"]:
            self._fill_prism(prism, raw)
        self._mark_border(raw, world["bounds"])
        self.raw = raw
        self.blocked = self._inflate(raw)  # one cell of clearance for the mech's radius
        self.beside = self._mark_beside()  # walkable cells with a building within 8 m

        self.dist = array("H", [UNSEEN]) * n
        self.goal: tuple[float, float] | None = None
        self._frontier: list[int] = []
        self._next: list[int] = []
        self._pos = 0
        self._level = 0

    # ------------------------------------------------------------------ grid construction
    def _mark_steep(self, raw: bytearray) -> None:
        tnx, heights, w, h = self.tnx, self.heights, self.w, self.h
        for tj in range(self.tny - 1):
            row = heights[tj * tnx : (tj + 1) * tnx]
            up = heights[(tj + 1) * tnx : (tj + 2) * tnx]
            j = min(h - 1, tj // 2)
            for ti, (a, b, c) in enumerate(zip(row, row[1:], up, strict=False)):
                if abs(a - b) > STEEP_CM or abs(a - c) > STEEP_CM:
                    raw[j * w + min(w - 1, ti // 2)] = 1

    def _fill_prism(self, prism: dict, raw: bytearray) -> None:
        """Even-odd scanline fill over the ring and its holes together, sampled at cell centers."""
        edges = []
        for ring in [prism["ring"], *prism.get("holes", [])]:
            pts = list(zip(ring[0::2], ring[1::2], strict=True))
            for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True):
                if y0 != y1:
                    edges.append((x0, y0, x1, y1) if y0 < y1 else (x1, y1, x0, y0))
        if not edges:
            return
        base, top = prism["base"], prism["top"]
        w, tall = self.w, self.tall
        j0 = max(1, int((min(e[1] for e in edges) - self.min_y) / CELL))
        j1 = min(self.h - 2, int((max(e[3] for e in edges) - self.min_y) / CELL))
        for j in range(j0, j1 + 1):
            y = self.min_y + (j + 0.5) * CELL
            xs = sorted(
                x0 + (y - y0) * (x1 - x0) / (y1 - y0) for x0, y0, x1, y1 in edges if y0 <= y < y1
            )
            for xa, xb in zip(xs[0::2], xs[1::2], strict=False):
                i0 = max(1, math.ceil((xa - self.min_x) / CELL - 0.5))
                i1 = min(w - 2, math.floor((xb - self.min_x) / CELL - 0.5))
                for i in range(i0, i1 + 1):
                    ground = self.heights[(2 * j + 1) * self.tnx + 2 * i + 1] / 100
                    if top - ground > SOLID_ABOVE:
                        raw[j * w + i] = 1
                        if top - ground >= SIGHT_ABOVE and base - ground < SIGHT_ABOVE / 2:
                            tall[j * w + i] = 1

    def _mark_border(self, raw: bytearray, bounds: dict) -> None:
        w, h = self.w, self.h
        i0 = int((bounds["minX"] - self.min_x) / CELL) + 1
        i1 = int((bounds["maxX"] - self.min_x) / CELL) - 1
        j0 = int((bounds["minY"] - self.min_y) / CELL) + 1
        j1 = int((bounds["maxY"] - self.min_y) / CELL) - 1
        for j in range(h):
            for i in range(w):
                if not (i0 <= i <= i1 and j0 <= j <= j1):
                    raw[j * w + i] = 1

    def _inflate(self, raw: bytearray) -> bytearray:
        w, n = self.w, len(raw)
        out = bytearray(raw)
        for c in range(w + 1, n - w - 1):
            if raw[c]:
                for d in (-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1):
                    out[c + d] = 1
        for c in list(range(w)) + list(range(n - w, n)):
            out[c] = 1
        for c in range(0, n, w):
            out[c] = out[c + w - 1] = 1
        return out

    def _mark_beside(self) -> bytearray:
        """Cells a mech can stand in that have a sight-blocking building within 8 m."""
        w, h, tall, blocked = self.w, self.h, self.tall, self.blocked
        beside = bytearray(len(tall))
        rad = 4
        for c, is_tall in enumerate(tall):
            if not is_tall:
                continue
            j, i = divmod(c, w)
            j0, j1 = max(1, j - rad), min(h - 2, j + rad)
            i0, i1 = max(1, i - rad), min(w - 2, i + rad)
            for jj in range(j0, j1 + 1):
                row = jj * w
                for ii in range(i0, i1 + 1):
                    cell = row + ii
                    if not blocked[cell]:
                        beside[cell] = 1
        return beside

    # ------------------------------------------------------------------ lookups
    def cell(self, x: float, y: float) -> int:
        i = min(self.w - 1, max(0, int((x - self.min_x) / CELL)))
        j = min(self.h - 1, max(0, int((y - self.min_y) / CELL)))
        return j * self.w + i

    def center(self, c: int) -> tuple[float, float]:
        return (
            self.min_x + (c % self.w + 0.5) * CELL,
            self.min_y + (c // self.w + 0.5) * CELL,
        )

    def ground(self, x: float, y: float) -> float:
        i = min(self.tnx - 1, max(0, round(x - self.min_x)))
        j = min(self.tny - 1, max(0, round(y - self.min_y)))
        return self.heights[j * self.tnx + i] / 100

    def _ray(self, grid: bytearray, x0: float, y0: float, x1: float, y1: float, escape: bool):
        """True if a 1 m-sampled segment touches `grid`. With `escape`, a blocked start is forgiven
        for the first few meters: a mech sliding along a wall stands in the clearance band."""
        dx, dy = x1 - x0, y1 - y0
        steps = int(math.hypot(dx, dy)) + 1
        dx /= steps
        dy /= steps
        w, h, min_x, min_y = self.w, self.h, self.min_x, self.min_y
        leaving = escape
        for k in range(steps + 1):
            i = int((x0 + dx * k - min_x) / CELL)
            j = int((y0 + dy * k - min_y) / CELL)
            if not (0 <= i < w and 0 <= j < h):
                return True
            if grid[j * w + i]:
                if not (leaving and k <= 5):
                    return True
            else:
                leaving = False
        return False

    def clear(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Can the mech walk the straight segment?"""
        return not self._ray(self.blocked, x0, y0, x1, y1, True)

    def hidden(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Does a tall building stand between the two points?"""
        return self._ray(self.tall, x0, y0, x1, y1, False)

    def by_cover(self, x: float, y: float) -> bool:
        """Is (x, y) a walkable cell with a building within 8 m?"""
        return bool(self.beside[self.cell(x, y)])

    # ------------------------------------------------------------------ flow field
    def set_goal(self, x: float, y: float) -> None:
        """Restart the flow field toward (x, y). `work` fills it in over the following decisions."""
        if self.goal == (x, y):
            return
        self.goal = (x, y)
        self.dist = array("H", [UNSEEN]) * len(self.blocked)
        start = self._nearest_free(self.cell(x, y))
        self._frontier, self._next, self._pos, self._level = [], [], 0, 0
        if start is not None:
            self.dist[start] = 0
            self._frontier = [start]

    def _nearest_free(self, c: int) -> int | None:
        w, h, blocked = self.w, self.h, self.blocked
        ci, cj = c % w, c // w
        for r in range(0, 40):
            for j in range(max(1, cj - r), min(h - 1, cj + r + 1)):
                for i in range(max(1, ci - r), min(w - 1, ci + r + 1)):
                    if max(abs(i - ci), abs(j - cj)) == r and not blocked[j * w + i]:
                        return j * w + i
        return None

    def work(self, budget: int) -> None:
        """Expand up to `budget` cells of the breadth-first search from the goal."""
        dist, blocked, w = self.dist, self.blocked, self.w
        while budget > 0 and self._frontier:
            frontier, nxt, level = self._frontier, self._next, self._level + 1
            end = min(len(frontier), self._pos + budget)
            budget -= end - self._pos
            for c in frontier[self._pos : end]:
                for nb in (c - 1, c + 1, c - w, c + w):
                    if dist[nb] == UNSEEN and not blocked[nb]:
                        dist[nb] = level
                        nxt.append(nb)
            self._pos = end
            if end == len(frontier):
                self._frontier, self._next, self._pos, self._level = nxt, [], 0, level

    def waypoint(self, x: float, y: float, look: int = 14) -> tuple[float, float] | None:
        """A point to walk straight at that follows the flow field, or None while the field has
        not reached this position."""
        dist, w = self.dist, self.w
        c = self.cell(x, y)
        if dist[c] == UNSEEN:  # standing in the clearance band: start from the best neighbor
            ring = [c + dj * w + di for dj in (-2, -1, 0, 1, 2) for di in (-2, -1, 0, 1, 2)]
            ring = [r for r in ring if 0 <= r < len(dist) and dist[r] != UNSEEN]
            if not ring:
                return None
            c = min(ring, key=dist.__getitem__)
        best = self.center(c)
        for _ in range(look):
            if dist[c] == 0:
                break
            c = min(
                (c - 1, c + 1, c - w, c + w, c - w - 1, c - w + 1, c + w - 1, c + w + 1),
                key=dist.__getitem__,
            )
            px, py = self.center(c)
            if not self.clear(x, y, px, py):
                break
            best = (px, py)
        return best

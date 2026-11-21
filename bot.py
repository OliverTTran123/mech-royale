"""Rule-based mech pilot. Placement is scored, so every rule serves staying alive:

- zone: be inside the next circle before the current one starts to shrink;
- threats: back off behind cover when outnumbered or losing, and let health regenerate;
- loot: collect cubes, a laser or lightning, and supply drops while nobody is near;
- targets: shoot the mech that is cheapest to kill, leading the shot.

Legs and torso are independent, so movement and aim are decided separately each decision.
"""

import math
import random

import mechfight as mf
from nav import Nav

TWO_PI = 2 * math.pi
FIELD_CELLS_PER_TICK = 4000  # flow-field work per decision, about 2 ms
ENGAGE = 170.0  # mechs closer than this are threats and targets
WANTED = ("laser", "lightning")  # a mortar is skipped: it invites laser cook-off
UPGRADES = [  # most wanted first: staying alive, then damage
    "armor", "plating", "nanites", "gun_damage", "gun_rate", "servos", "laser_damage",
    "lightning_efficiency", "gyros", "gun_mag", "laser_charge", "gun_velocity", "lightning_range",
    "lightning_chain", "reactor", "jets", "capacitor", "sensors", "laser_pierce",
]  # fmt: skip
LOUD = {"gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet"}


def wrap(angle: float) -> float:
    return (angle + math.pi) % TWO_PI - math.pi


class Bot:
    name = "oval-survivor"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.rng = random.Random(hello["seed"])
        self.reach = cfg["loot"]["pickup_radius"]
        self.full_hp = cfg["mech"]["hp"]
        self.half_height = {
            "mech": cfg["mech"]["height"] / 2,
            "npc": cfg["npc"]["height"] / 2,
            "crate": cfg["crates"]["height"] / 2,
        }
        self.half_width = {
            "mech": cfg["mech"]["radius"],
            "npc": cfg["npc"]["radius"],
            "crate": cfg["crates"]["radius"],
        }
        self.nav: Nav | None = None
        try:
            self.nav = Nav(hello["map"]["dir"])
        except (OSError, ValueError, KeyError) as error:
            mf.log(f"no map, steering blind: {error!r}")

        self.attackers: dict[int, float] = {}  # mech id -> last time it damaged me
        self.hurt_t = self.npc_hurt_t = self.heard_t = self.last_threat_t = -99.0
        self.hurt_bearing = self.heard_bearing = 0.0
        self.laser_off_until = 0.0
        self.retreat = False
        self.flee_yaw: float | None = None
        self.flee_at = -99.0
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.side = 1  # which way feelers turn first
        self.slow = 0  # decisions spent pushing without moving
        self.jetting = False
        self.jet_start = 0.0
        self.was_moving = False
        self.loot_id = -1
        self.loot_since = 0.0
        self.skip: set[int] = set()  # loot given up on

    # ------------------------------------------------------------------ one decision
    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        t = obs["t"]
        x, y, z = me["pos"]
        zone = obs["zone"]
        nav = self.nav
        if nav:
            nav.set_goal(zone["next_center"][0], zone["next_center"][1])
            nav.work(FIELD_CELLS_PER_TICK)
        self._events(obs["events"], t)

        mechs, npcs = [], []
        for m in obs["mechs"]:
            m["_d"] = math.hypot(m["pos"][0] - x, m["pos"][1] - y)
            (mechs if m["kind"] == "mech" else npcs).append(m)
        threats = sorted(
            (m for m in mechs if m["_d"] < ENGAGE or self.attackers.get(m["id"], -99) > t - 4),
            key=lambda m: m["_d"],
        )
        hp = me["hp"]
        frac = hp / me["max_hp"]
        if threats:
            self.last_threat_t = t
            nearest_hp = threats[0].get("hp", self.full_hp)
            if len(threats) >= 2 or frac < 0.4 or hp < 0.6 * nearest_hp:
                self.retreat = True
            elif frac >= 0.7 and hp >= 0.8 * nearest_hp:
                self.retreat = False
        elif t - self.last_threat_t > 3:
            self.retreat = False
        target = min(threats, key=lambda m: self._kill_cost(m, t)) if threats else None

        # --- legs
        nc, nr = zone["next_center"], zone["next_radius"]
        safe_r = nr - min(20.0, 0.3 * nr) if nr > 0 else max(2.0, 0.4 * zone["radius"])
        d_out = math.hypot(x - nc[0], y - nc[1]) - safe_r
        walk = me["stats"]["walk_speed"]
        urgent = d_out > 0 and (
            not me["in_zone"] or t + d_out / walk * 1.3 + 8 >= zone["shrink_start"]
        )
        calm = not threats
        healthy = frac >= 0.8
        loot = self._loot(obs, me, npcs, t, healthy) if calm and not urgent else None
        dodge = self._shell_dodge(obs["projectiles"], x, y, z)
        yaw: float | None = None
        boost = False
        if dodge is not None:
            yaw, boost = dodge, True
        elif urgent:
            yaw = self._toward(me, nc[0], nc[1], True)
            boost = not me["in_zone"] or t >= zone["shrink_start"]
        elif self.retreat and threats:
            yaw = self._flee(me, threats, zone, t)
            boost = threats[0]["_d"] < 120
        elif target is not None:
            yaw = self._strafe(me, target, zone, t)
        elif loot is not None:
            if math.hypot(loot[0] - x, loot[1] - y) > loot[2]:
                yaw = self._toward(me, loot[0], loot[1], False)
        elif d_out > 0:
            yaw = self._toward(me, nc[0], nc[1], True)

        # --- pushing against something: jump and jet over it
        speed = math.hypot(me["vel"][0], me["vel"][1])
        self.slow = self.slow + 1 if self.was_moving and speed < 2.0 else 0
        self.was_moving = yaw is not None
        if self.jetting:
            cleared = speed > 4.0 and t - self.jet_start > 0.5
            if cleared or me["energy"] < 4 or t - self.jet_start > 7 or yaw is None:
                self.jetting = False
        elif self.slow >= 5 and me["energy"] >= 40 and yaw is not None:
            self.jetting, self.jet_start, self.slow = True, t, 0
            self.side = -self.side
        jump = self.jetting and me["grounded"]
        if self.jetting:
            boost = True
        if boost and me["energy"] < 2:
            boost = False

        # --- torso
        farm = None if target or not calm else self._farm_target(obs, me, npcs, t, healthy)
        aim, fire, firing = self._gunnery(me, target, farm, t)
        if aim is None:
            aim = [self._look(obs["sounds"], me, t, yaw), 0.0]

        move = [0, 0] if yaw is None else [round(math.cos(yaw), 3), round(math.sin(yaw), 3)]
        action: mf.Action = {
            "move": move,
            "boost": boost,
            "jump": jump,
            "aim": [round(wrap(aim[0]), 4), round(aim[1], 4)],
            "fire": fire,
        }
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        if gun and not gun["reload"]:
            idle = target is None and not firing and gun["ammo"] < gun["stats"]["mag"]
            if gun["ammo"] == 0 or idle:  # empty, or topping up between fights
                action["reload"] = True
        if obs["offer"]:
            action["pick"] = self._pick(obs["offer"])
        take = self._take(obs["pickups"], me)
        if take is not None:
            action["take"] = take
        return action

    # ------------------------------------------------------------------ bookkeeping
    def _events(self, events: list, t: float) -> None:
        for e in events:
            kind = e["type"]
            if kind == "damaged":
                if e["weapon"] == "npc":
                    self.npc_hurt_t = t
                elif e.get("by") is not None:
                    self.attackers[e["by"]] = t
                    self.hurt_t = t
                    if e.get("bearing") is not None:
                        self.hurt_bearing = e["bearing"]
            elif kind == "effect" and e["effect"] == "short":
                self.laser_off_until = t + self.cfg["weapons"]["lightning"]["short_s"]

    def _pick(self, offer: dict) -> int:
        def rank(choice: dict) -> int:
            return UPGRADES.index(choice["id"]) if choice["id"] in UPGRADES else len(UPGRADES)

        choices = offer["choices"]
        return min(range(len(choices)), key=lambda i: rank(choices[i]))

    def _take(self, pickups: list, me: dict) -> int | None:
        if None not in me["weapons"]:
            return None
        x, y, z = me["pos"]
        held = {w["kind"] for w in me["weapons"] if w}
        for p in pickups:
            px, py, pz = p["pos"]
            near = math.hypot(px - x, py - y) < self.reach - 0.3 and abs(pz - z) < 3.7
            if p["kind"] in WANTED and p["kind"] not in held and near:
                return p["id"]
        return None

    def _kill_cost(self, m: dict, t: float) -> float:
        """Lower is a better target: little health, close, and already shooting at me."""
        cost = m.get("hp", self.full_hp) * (1 + m["_d"] / 100)
        return cost * 0.6 if self.attackers.get(m["id"], -99) > t - 4 else cost

    # ------------------------------------------------------------------ legs
    def _toward(self, me: dict, gx: float, gy: float, far: bool) -> float:
        """Heading to walk toward (gx, gy): the flow field for the zone, feelers for close goals."""
        x, y, z = me["pos"]
        direct = math.atan2(gy - y, gx - x)
        nav = self.nav
        if nav is None or math.hypot(gx - x, gy - y) < 12:
            return direct
        if nav.raw[nav.cell(x, y)] and z - nav.ground(x, y) > 2.5:
            return direct  # on a roof: walk off it
        if far:
            point = nav.waypoint(x, y)
            if point is not None:
                return math.atan2(point[1] - y, point[0] - x)
        return self._steer(x, y, direct)

    def _steer(self, x: float, y: float, yaw: float) -> float:
        """The heading nearest `yaw` whose next 10 m are walkable."""
        nav = self.nav
        if nav is None:
            return yaw
        for k in range(6):
            for s in (self.side, -self.side) if k else (1,):
                a = yaw + s * k * 0.4
                if nav.clear(x, y, x + 10 * math.cos(a), y + 10 * math.sin(a)):
                    if k:
                        self.side = s
                    return a
        return yaw

    def _strafe(self, me: dict, target: dict, zone: dict, t: float) -> float:
        """Circle the target at a preferred range, switching direction at random to spoil leads."""
        x, y, _ = me["pos"]
        if t >= self.strafe_until:
            self.strafe_sign = self.rng.choice((-1, 1))
            self.strafe_until = t + self.rng.uniform(0.6, 1.5)
        held = {w["kind"] for w in me["weapons"] if w}
        keep = 38.0 if "lightning" in held else 110.0 if "laser" in held else 90.0
        bearing = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
        radial = max(-1.0, min(1.0, (target["_d"] - keep) / 25))
        c = zone["center"]
        for sign in (self.strafe_sign, -self.strafe_sign):
            yaw = math.atan2(
                radial * math.sin(bearing) + sign * math.cos(bearing),
                radial * math.cos(bearing) - sign * math.sin(bearing),
            )
            px, py = x + 12 * math.cos(yaw), y + 12 * math.sin(yaw)
            if math.hypot(px - c[0], py - c[1]) < zone["radius"] - 10:
                self.strafe_sign = sign
                break
        return self._steer(x, y, yaw)

    def _flee(self, me: dict, threats: list, zone: dict, t: float) -> float:
        """Heading that gains distance from the threats, prefers cover, and stays in the zone."""
        if self.flee_yaw is not None and t - self.flee_at < 0.25:
            return self.flee_yaw
        x, y, _ = me["pos"]
        nav = self.nav
        c, nc = zone["center"], zone["next_center"]
        closing = t > zone["shrink_start"] - 20
        best, best_score = None, -1e9
        for k in range(16):
            a = k * TWO_PI / 16
            px, py = x + 24 * math.cos(a), y + 24 * math.sin(a)
            if nav and not nav.clear(x, y, px, py):
                continue
            score = 0.0
            for m in threats[:3]:
                ex, ey = m["pos"][0], m["pos"][1]
                score += math.hypot(px - ex, py - ey) - m["_d"]
                if nav and nav.hidden(px, py, ex, ey):
                    score += 30
            if math.hypot(px - c[0], py - c[1]) > zone["radius"] - 15:
                score -= 200
            if closing and math.hypot(px - nc[0], py - nc[1]) > zone["next_radius"] - 5:
                score -= 60
            if self.flee_yaw is not None and abs(wrap(a - self.flee_yaw)) < 0.5:
                score += 8  # keep running the same way
            if score > best_score:
                best, best_score = a, score
        if best is None:
            ex, ey = threats[0]["pos"][0], threats[0]["pos"][1]
            best = math.atan2(y - ey, x - ex)
        self.flee_yaw, self.flee_at = best, t
        return best

    def _shell_dodge(self, shells: list, x: float, y: float, z: float) -> float | None:
        """Heading out of the blast of a mortar shell about to land on me."""
        mortar = self.cfg["weapons"]["mortar"]
        g = mortar["gravity"]
        for p in shells:
            if p["kind"] != "mortar":
                continue
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            under_root = vz * vz + 2 * g * (pz - z)
            if under_root < 0:
                continue
            fall = (vz + math.sqrt(under_root)) / g  # seconds until it reaches my feet
            dx, dy = x - (px + vx * fall), y - (py + vy * fall)
            if fall < 3 and math.hypot(dx, dy) < mortar["splash_radius"] + 6:
                return math.atan2(dy, dx) if math.hypot(dx, dy) > 0.5 else math.atan2(vx, -vy)
        return None

    def _on_roof(self, pos: list, z: float) -> bool:
        """Is the item well above both me and the ground under it (reachable only by jet)?"""
        nav = self.nav
        return bool(nav) and pos[2] - z > 3 and pos[2] - nav.ground(pos[0], pos[1]) > 3

    def _safe_spot(self, px: float, py: float, zone: dict, t: float) -> bool:
        c, nc = zone["center"], zone["next_center"]
        if math.hypot(px - c[0], py - c[1]) > zone["radius"] - 12:
            return False
        if t > zone["shrink_start"] - 25:
            return math.hypot(px - nc[0], py - nc[1]) < zone["next_radius"] - 3
        return True

    def _loot(self, obs: dict, me: dict, npcs: list, t: float, healthy: bool):
        """(x, y, stop distance) of the most worthwhile thing nearby, or None."""
        x, y, z = me["pos"]
        zone = obs["zone"]
        best, best_cost = None, 1e9

        def consider(item: dict, limit: float, stop: float, bias: float) -> None:
            nonlocal best, best_cost
            px, py, pz = item["pos"]
            d = math.hypot(px - x, py - y)
            if item["id"] in self.skip or d > limit or not self._safe_spot(px, py, zone, t):
                return
            if self._on_roof(item["pos"], z):
                return
            if d + bias < best_cost:
                best, best_cost = (px, py, stop, item["id"]), d + bias

        held = {w["kind"] for w in me["weapons"] if w}
        if None in me["weapons"]:
            for p in obs["pickups"]:
                if p["kind"] in WANTED and p["kind"] not in held:
                    consider(p, 90, 0, -30)
        for cube in obs["cubes"]:
            consider(cube, 70, 0, -20)
        for drop in obs["drops"]:
            if drop["lands_in"] < 12 and drop.get("holder") in (None, me["id"]):
                consider(drop, 130, 1.5, -60)
        if healthy:
            for crate in obs["crates"]:
                consider(crate, 80, 30, 0)
            for npc in npcs:
                consider(npc, 80, 40, 10)
        if best is None:
            return None
        if best[3] != self.loot_id:
            self.loot_id, self.loot_since = best[3], t
        elif t - self.loot_since > 15 and best[3] < 20000:
            self.skip.add(best[3])  # cannot get to it: stop trying
        return best

    # ------------------------------------------------------------------ torso
    def _farm_target(self, obs: dict, me: dict, npcs: list, t: float, healthy: bool):
        """A drone or crate worth shooting for its cubes while no mech is around."""
        x, y, z = me["pos"]
        nav = self.nav
        options = []
        for npc in npcs:
            if npc["_d"] < 70 and (healthy or t - self.npc_hurt_t < 2):
                options.append((npc["_d"], npc, "npc"))
        if healthy:
            for crate in obs["crates"]:
                d = math.hypot(crate["pos"][0] - x, crate["pos"][1] - y)
                if d < 60 and crate["id"] not in self.skip and not self._on_roof(crate["pos"], z):
                    options.append((d + 20, crate, "crate"))
        for _, item, kind in sorted(options, key=lambda o: o[0]):
            if not (nav and nav.hidden(x, y, item["pos"][0], item["pos"][1])):
                item["_d"] = math.hypot(item["pos"][0] - x, item["pos"][1] - y)
                item["kind"] = kind
                return item
        return None

    def _solution(self, me: dict, target: dict, speed: float | None) -> tuple[float, float, float]:
        """(yaw, pitch, distance) to hit the target's center, leading it for a shot at `speed`
        (None for an instant shot, which only allows for the decision delay)."""
        x, y, z = me["pos"]
        mz = z + me["stats"]["muzzle_height"]
        grow = 1 + self.cfg["cubes"]["size_per_cube"] * target.get("cubes", 0)
        tx, ty, tz = target["pos"]
        tz += self.half_height[target["kind"]] * grow
        vx, vy = target["vel"][:2] if "vel" in target else (0.0, 0.0)  # crates stand still
        delay = 0.05
        dx, dy, dz = tx - x, ty - y, tz - mz
        for _ in range(2):
            dx, dy = tx + vx * delay - x, ty + vy * delay - y
            if speed:
                delay = 0.05 + math.sqrt(dx * dx + dy * dy + dz * dz) / speed
        flat = math.hypot(dx, dy)
        return math.atan2(dy, dx), math.atan2(dz, flat), flat

    def _gunnery(self, me: dict, target: dict | None, farm: dict | None, t: float):
        """(aim or None, fire flags, whether the gun is being fired)."""
        fire = [False, False]
        aim_at = target or farm
        if aim_at is None:
            return None, fire, False
        x, y, _ = me["pos"]
        nav = self.nav
        grow = 1 + self.cfg["cubes"]["size_per_cube"] * aim_at.get("cubes", 0)
        half_w = self.half_width[aim_at["kind"]] * grow
        half_h = self.half_height[aim_at["kind"]] * grow
        d = aim_at["_d"]
        seen = not (nav and nav.hidden(x, y, aim_at["pos"][0], aim_at["pos"][1]))
        now_yaw, now_pitch = me["aim"]
        aim = None
        firing = False

        def on_target(sol: tuple[float, float, float], slack: float) -> bool:
            flat = max(sol[2], 1.0)
            return (
                abs(wrap(sol[0] - now_yaw)) < math.atan2(half_w * 0.9, flat) + slack
                and abs(sol[1] - now_pitch) < math.atan2(half_h * 0.9, flat) + slack
            )

        for slot, weapon in enumerate(me["weapons"]):
            if not weapon:
                continue
            kind, stats = weapon["kind"], weapon["stats"]
            if kind == "gun":
                lead = self._solution(me, aim_at, stats["speed"])
                aim = aim or [lead[0], lead[1]]
                ready = weapon["ammo"] > 0 and not weapon["reload"] and d < stats["range"] * 0.95
                if ready and seen and on_target(lead, 0.05):
                    fire[slot] = firing = True
            elif kind == "lightning" and target is not None:
                direct = self._solution(me, target, None)
                near = d < stats["range"] - 2 and abs(wrap(direct[0] - now_yaw)) < 0.2
                fire[slot] = near and seen and me["energy"] > stats["energy"]
            elif kind == "laser" and target is not None:
                direct = self._solution(me, target, None)
                charge = weapon.get("charge", 0)
                full = stats["charge"]
                if charge > 0:
                    if charge >= full - 0.15:
                        aim = [direct[0], direct[1]]  # the beam is instant: stop leading
                    fire[slot] = not (charge >= full - 0.06 and on_target(direct, 0.0))
                else:
                    usable = weapon["cooldown"] <= 0.05 and t >= self.laser_off_until
                    safe = not self.retreat and me["in_zone"]
                    fire[slot] = usable and safe and seen and d < stats["range"] * 0.95
        if aim is None:  # no gun in hand
            direct = self._solution(me, aim_at, None)
            aim = [direct[0], direct[1]]
        return aim, fire, firing

    def _look(self, sounds: list, me: dict, t: float, move_yaw: float | None) -> float:
        """Where to point the torso with nothing to shoot: at whoever hurt me, at a close noise,
        otherwise sweep so the 120 degree view covers the surroundings."""
        if t - self.hurt_t < 2.5:
            return self.hurt_bearing
        for s in sounds:
            if s["kind"] in LOUD and s["dist"] != "far" and s["age"] < 0.5:
                self.heard_t, self.heard_bearing = t, s["bearing"]
                if s["dist"] == "near":
                    break
        if t - self.heard_t < 1.5:
            return self.heard_bearing
        if move_yaw is not None:
            return move_yaw + 1.1 * math.sin(t * 1.6)
        return me["aim"][0] + 0.3


if __name__ == "__main__":
    mf.run(Bot)

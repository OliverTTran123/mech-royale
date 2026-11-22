"""Rule-based mech pilot. Placement is scored, so every rule serves staying alive:

- zone: be inside the next circle before the current one starts to shrink;
- threats: back off behind cover when outnumbered or losing, and let health regenerate;
- loot: collect cubes, a laser or lightning, and supply drops while nobody is near;
- targets: shoot the mech that is cheapest to kill, leading the shot;
- roof camp: after the opening, jet onto the roof in the next circle that the least ground can
  see, stand back from its edges, and change roofs when the circle moves on.

Legs and torso are independent, so movement and aim are decided separately each decision.
Each rule has a flag in config.py; with a flag off, that part does what the kit's starter does.
"""

import hashlib
import math
import random
import struct
import time
from pathlib import Path
from typing import NamedTuple

import config
import mechfight as mf
from nav import CELL, NO_ROOF, Nav

TWO_PI = 2 * math.pi
FIELD_CELLS_PER_TICK = 4000  # flow-field work per decision, about 2 ms
TICK_BUDGET_S = 0.025  # past this, finish the tick with the cheap ground fight instead
ENGAGE = 170.0  # mechs closer than this are threats and targets
WANTED = ("laser", "lightning")  # a mortar is skipped unless MORTAR is on: a laser cooks it off
KITE_RANGE = 90.0  # circle a target at this distance when KITE is on
MORTAR_MIN, MORTAR_MAX = 40.0, 180.0  # lob only inside this; closer, the gun is better
EARLY_S = 90.0  # opening: detour for a weapon even with a mech nearby, and farm cubes
FARM_CLEAR = 80.0  # opening: crates and drones come before a mech that is no closer than this
SWARM_S = 5.0  # two drones shooting at me: stay clear of drones this long
CUBE_M = 70.0  # walk to a cube this close at any health, even with a mech nearby
CUBE_GIVE_UP_S = 15.0  # still walking at the same cube after this long: it cannot be reached
LIGHTNING_CLOSE = 38.0  # holding lightning: walk at the target until it is inside the arc
CAMP_CUBES = 3  # roof camp: farm the opening until this many cubes ...
CAMP_S = 60.0  # ... or this long, then go up
ROOF_LOW_M, ROOF_HIGH_M = 9.0, 50.0  # roofs worth it: above a ground mech's eye, within one jet
ROOF_SETBACK_M = 6.0  # a camp spot is at least this far from every edge of its roof
ROOF_HIDES_M = 8.0  # ... and no mech on the ground this near its walls sees the camper at all
ROOF_GRID = 4  # camp spots are looked for every this many map cells (8 m)
CAMP_HEIGHT_M = 11.7  # how tall the camper is taken to be when counting who sees it (6 cubes)
CLIMB_ENERGY = 80.0  # start a climb with this much: it lifts about 53 m
CAMP_GIVE_UP_S = 20.0  # no nearer the chosen roof after this long: it cannot be reached
HIDE_HP, HEALED_HP = 0.55, 0.9  # on a roof: below the first, out of sight until above the second
CAMP_TRIES = 3  # roofs given up on before the rest of the match is fought on the ground
UPGRADES = [  # most wanted first: less reloading, then gun damage, then staying alive
    "gun_mag", "gun_damage", "gun_rate", "armor", "plating", "nanites", "servos", "laser_damage",
    "lightning_efficiency", "gyros", "laser_charge", "gun_velocity", "lightning_range",
    "lightning_chain", "reactor", "jets", "capacitor", "sensors", "laser_pierce",
]  # fmt: skip
LOUD = {"gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet"}


def wrap(angle: float) -> float:
    return (angle + math.pi) % TWO_PI - math.pi


class Roof(NamedTuple):
    """A spot to camp on: flat roof all around it, and ground beside the building to jet from."""

    x: float
    y: float
    z: float  # height of the roof
    back: float  # distance to the nearest edge of the roof
    seen: int  # how many ground points see a mech standing here (nav.exposure)
    ways: tuple[tuple[float, float], ...]  # ground points to walk to and jet up from


ROOFS_BY_MAP: dict[str, list[Roof]] = {}  # map folder -> its camp spots, loaded once


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
        if config.ZONE or config.THREAT_RETREAT or config.LOOT or config.TARGETING or config.KITE:
            try:
                self.nav = Nav(hello["map"]["dir"])
            except (OSError, ValueError, KeyError) as error:
                mf.log(f"no map, steering blind: {error!r}")

        self.attackers: dict[int, float] = {}  # mech id -> last time it damaged me
        self.drone_hits: dict[int, float] = {}  # drone id -> last time it damaged me
        self.swarm_until = -99.0
        self.hurt_t = self.npc_hurt_t = self.heard_t = self.last_threat_t = -99.0
        self.hurt_bearing = self.heard_bearing = 0.0
        self.laser_off_until = 0.0
        self.retreat = False
        self.flee_yaw: float | None = None
        self.flee_at = -99.0
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.orbit_sign = 1
        self.side = 1  # which way feelers turn first
        self.slow = 0  # decisions spent pushing without moving
        self.jetting = False
        self.jet_start = 0.0
        self.was_moving = False
        self.loot_id = -1
        self.loot_since = 0.0
        self.cube_id = -1
        self.cube_since = 0.0
        self.skip: set[int] = set()  # loot given up on
        # roof camp: every spot worth standing on, found once from the map
        self.roofs: list[Roof] = []
        if self.nav and config.ROOF_CAMP:
            map_dir = hello["map"]["dir"]
            if map_dir not in ROOFS_BY_MAP:
                ROOFS_BY_MAP[map_dir] = self._roof_spots()
            self.roofs = ROOFS_BY_MAP[map_dir]
        self.camp: Roof | None = None  # the roof spot being walked to, climbed to, or held
        self.left: Roof | None = None  # the one before it, which the next circle no longer holds
        self.camp_boost = False  # running for `camp` with mechs close behind
        self.lying_low = False  # hurt and unseen on a roof: hold fire so that health comes back
        self.camp_key: tuple | None = None  # the circle (and failures) `camp` was chosen for
        self.camp_bad: list[Roof] = []  # roofs that could not be reached
        self.camp_since = 0.0  # when the mech last got nearer to `camp`
        self.camp_left = 1e9  # the nearest it has been, in meters to go
        self.way: tuple[float, float] | None = None  # ground walked to: below `camp`, or cover
        self.lined_up = False  # reached `way`: from here it is straight at the roof and up
        self.rush = False  # late for the next circle: the map leads to its center until inside
        self.roof_item = -1  # the cube or weapon being fetched on the roof ...
        self.roof_item_since = 0.0  # ... since when
        self.healing = False  # hurt on the roof: staying out of sight until healed
        self.hide_at = (0.0, 0.0, 0)  # where on the roof to do that, and how seen it still is
        self.hide_until = 0.0  # when to look for a better place
        self.aloft = False  # standing on a roof this decision
        self.climb_to: float | None = None  # roof height being jetted up to ...
        self.climb_yaw: float | None = None  # ... on this heading
        self.budget_hits = 0  # ticks that gave up the slow fight and played the cheap one
        self._t0 = 0.0
        self._fallback = False

    # ------------------------------------------------------------------ one decision
    def _late(self) -> bool:
        return time.perf_counter() - self._t0 > TICK_BUDGET_S

    def _give_up(self) -> None:
        """This tick is out of time: the rest of it is the cheap ground fight."""
        if not self._fallback:
            self._fallback = True
            self.budget_hits += 1

    def act(self, obs: mf.Observation) -> mf.Action:
        self._t0 = time.perf_counter()
        self._fallback = False
        if not (
            config.ZONE
            or config.THREAT_RETREAT
            or config.LOOT
            or config.TARGETING
            or config.ROOF_CAMP
            or config.KITE
            or config.MORTAR
            or config.JUKE
        ):
            return self._starter(obs)
        me = obs["self"]
        t = obs["t"]
        x, y, z = me["pos"]
        zone = obs["zone"]
        nav = self.nav
        if nav and config.ZONE:
            # roof camp: once inside the next circle, the map leads to the ground below the roof
            if self._late():
                self._give_up()
            goal = (
                zone["next_center"] if self.way is None or self.rush or self._fallback else self.way
            )
            nav.set_goal(goal[0], goal[1])
            if not self._fallback and nav.work(FIELD_CELLS_PER_TICK, self._t0 + TICK_BUDGET_S):
                self._give_up()
        roof_on = config.ROOF_CAMP and not self._fallback
        self._events(obs["events"], t)
        roof = nav.roof(x, y) if nav and roof_on else None
        self.aloft = roof is not None and bool(me["grounded"]) and abs(z - roof) < 2.5
        self.lying_low = False

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
        if threats and config.THREAT_RETREAT:
            self.last_threat_t = t
            nearest_hp = threats[0].get("hp", self.full_hp)
            if len(threats) >= 2 or frac < 0.4 or hp < 0.6 * nearest_hp:
                self.retreat = True
            elif frac >= 0.7 and hp >= 0.8 * nearest_hp:
                self.retreat = False
        elif t - self.last_threat_t > 3:
            self.retreat = False
        focus = None  # the threat to circle and, with TARGETING, to shoot
        if threats and config.TARGETING:
            # from a roof most of them are behind its edge: one that can be hit comes first
            open_to_me = [m for m in threats if self._sees(me, m)] if self.aloft else threats
            focus = min(open_to_me or threats, key=lambda m: self._kill_cost(m, t))
        elif threats:
            focus = threats[0]

        # --- legs
        nc, nr = zone["next_center"], zone["next_radius"]
        safe_r = nr - min(20.0, 0.3 * nr) if nr > 0 else max(2.0, 0.4 * zone["radius"])
        d_out = math.hypot(x - nc[0], y - nc[1]) - safe_r
        walk = me["stats"]["walk_speed"]
        urgent = (
            config.ZONE
            and d_out > 0
            and (not me["in_zone"] or t + d_out / walk * 1.3 + 8 >= zone["shrink_start"])
        )
        calm = not threats
        crates_ok = drones_ok = frac >= 0.8  # shooting restarts the regeneration timer
        away: list = []  # drones to back off from
        farming = (
            config.LOOT
            and t < EARLY_S
            and t - self.hurt_t > 4
            and all(m["_d"] >= FARM_CLEAR for m in mechs)
        )
        if farming:  # opening, no mech close or hitting me: cubes come before a far mech
            calm = crates_ok = True
            swarmed = self._swarmed(npcs, x, y, t)
            drones_ok = frac > 0.7 and not swarmed
            if swarmed:
                reach = self.cfg["npc"]["deaggro_range"]
                away = sorted((n for n in npcs if n["_d"] < reach), key=lambda n: n["_d"])
        loot = dodge = None
        weapon = self._early_weapon(obs, me, t) if config.LOOT and not urgent else None
        cube = self._near_cube(obs, me, t) if config.LOOT and not urgent else None
        drop = self._drop_commit(obs, me, mechs, t) if config.LOOT and not urgent else None
        if config.LOOT and calm and not urgent:
            loot = self._loot(obs, me, npcs, t, crates_ok, drones_ok)
        if config.THREAT_RETREAT or config.JUKE:
            dodge = self._shell_dodge(obs["projectiles"], x, y, z)
        camping, camp_yaw = False, None
        if urgent:
            self.rush = True
        elif d_out < -5:
            self.rush = False
        if roof_on and self.roofs and (me["cubes"] >= CAMP_CUBES or t >= CAMP_S):
            if self._late():  # the opening is over, but this tick has no time to climb
                self._give_up()
                roof_on = False
            else:
                self._camp_plan(me, zone, t)
                if not urgent:
                    camping, camp_yaw = self._camp(obs, me, mechs, zone, t)
                if self._fallback:  # a line-of-sight search ran out of time: stay on the ground
                    camping, camp_yaw = False, None
                    self.aloft = False
                    self.lying_low = False
        # on the roof, in the air, or at its wall: nothing on the ground is worth coming down for
        roofward = camping and (self.aloft or self.lined_up or not me["grounded"])
        if roofward:
            away = []
        yaw: float | None = None
        boost = False
        starter_legs = False
        if dodge is not None:
            yaw, boost = dodge, True
        elif urgent:
            yaw = self._toward(me, nc[0], nc[1], True)
            boost = not me["in_zone"] or t >= zone["shrink_start"]
        elif roofward:
            yaw, boost = camp_yaw, self.camp_boost
        elif weapon is not None and math.hypot(weapon[0] - x, weapon[1] - y) > 1.5:
            yaw = self._toward(me, weapon[0], weapon[1], False)
        elif drop is not None:
            yaw = self._drop_yaw(me, drop)
        elif away:
            yaw = self._flee(me, away, zone, t)
            boost = away[0]["_d"] < self.cfg["npc"]["aggro_range"]
        elif cube is not None and math.hypot(cube[0] - x, cube[1] - y) > 1.5:
            yaw = self._toward(me, cube[0], cube[1], False)
        elif camping:  # still on the ground: a weapon, a drop or a cube came first, a chase not
            yaw, boost = camp_yaw, self.camp_boost
        elif config.THREAT_RETREAT and self.retreat and threats:
            yaw = self._flee(me, threats, zone, t)
            boost = threats[0]["_d"] < 120
        elif focus is not None and (
            config.THREAT_RETREAT
            or (not self._fallback and config.KITE)
            or (not self._fallback and config.JUKE and self._aimed_at_us(focus, x, y))
        ):
            yaw = self._strafe(me, focus, zone, t)
        elif (
            focus is not None
            and focus["_d"] > LIGHTNING_CLOSE
            and any(w and w["kind"] == "lightning" for w in me["weapons"])
        ):  # the arc reaches 45 m and the bot otherwise fights from further out
            yaw = self._toward(me, focus["pos"][0], focus["pos"][1], False)
        elif loot is not None:
            if math.hypot(loot[0] - x, loot[1] - y) > loot[2]:
                yaw = self._toward(me, loot[0], loot[1], False)
        elif not config.ZONE:
            starter_legs = True
            yaw = math.atan2(nc[1] - y, nc[0] - x)
        elif d_out > 0:
            yaw = self._toward(me, nc[0], nc[1], True)
        else:
            yaw = self._orbit(me, zone)  # inside, nothing to chase: don't be a stopped target

        if config.ZONE:  # pushing against something: jump and jet over it
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
        else:
            jump = obs["tick"] % 40 == 0
        if self.climb_to is not None and not self._fallback:  # a climb is flown on purpose
            above = z > self.climb_to + 1.0 and nav is not None and nav.roof(x, y) is not None
            if yaw is None or yaw != self.climb_yaw or above or me["energy"] < 4:
                self.climb_to = None
            else:
                jump, boost = bool(me["grounded"]), True
        if boost and me["energy"] < 2:
            boost = False
        if starter_legs:
            boost = not me["in_zone"]

        # --- torso
        firing = False
        if config.TARGETING and away:  # backing off: keep the drones in the front arc, hold fire
            facing = self._solution(me, away[0], None)
            aim, fire = [facing[0], facing[1]], [False, False]
        elif config.TARGETING:
            farm = None
            if config.LOOT and calm and (focus is None or farming):
                # from a roof a drone is shot only in defense: its cubes would fall out of reach
                farm = self._farm_target(obs, me, npcs, t, crates_ok, drones_ok and not self.aloft)
            target = None if farming and farm is not None else focus
            aim, fire, firing = self._gunnery(me, target, farm, t)
            if self.lying_low:  # dealing damage would put off the healing
                fire, firing = [False, False], False
            if aim is None:
                aim = [self._look(obs["sounds"], me, t, yaw), 0.0]
        elif mechs:
            tx, ty, tz = min(mechs, key=lambda m: math.dist(m["pos"], me["pos"]))["pos"]
            flat = math.hypot(tx - x, ty - y)
            aim, fire = [math.atan2(ty - y, tx - x), math.atan2(tz - z - 2.5, flat)], [True, True]
        else:
            aim, fire = [math.atan2(nc[1] - y, nc[0] - x), 0.0], [False, False]
            farm = None
            if config.LOOT:
                farm = self._farm_target(obs, me, npcs, t, crates_ok, drones_ok)
            if farm is not None:
                aim, fire, firing = self._gunnery(me, None, farm, t)

        move = [0, 0] if yaw is None else [round(math.cos(yaw), 3), round(math.sin(yaw), 3)]
        action: mf.Action = {
            "move": move,
            "boost": boost,
            "jump": jump,
            "aim": [round(wrap(aim[0]), 4), round(aim[1], 4)],
            "fire": fire,
        }
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        if config.TARGETING and gun and not gun["reload"]:
            idle = focus is None and not firing and gun["ammo"] < gun["stats"]["mag"]
            if gun["ammo"] == 0 or idle:  # empty, or topping up between fights
                action["reload"] = True
        if obs["offer"]:
            action["pick"] = self._pick(obs["offer"]) if config.LOOT else 0
        take = self._take(obs["pickups"], me) if config.LOOT else self._starter_take(obs, me)
        if take is not None:
            action["take"] = take
        return action

    def _starter(self, obs: mf.Observation) -> mf.Action:
        """The kit's example bot, verbatim in effect: straight at the next center, hop every 2 s,
        shoot the nearest mech with no lead, take the first weapon in reach."""
        me = obs["self"]
        x, y, z = me["pos"]
        goal = obs["zone"]["next_center"]
        act: mf.Action = {"tick": obs["tick"]}
        if obs["offer"]:
            act["pick"] = 0
        near = [p for p in obs["pickups"] if math.dist(p["pos"][:2], (x, y)) < self.reach]
        if me["weapons"][1] is None and near and near[0]["kind"] != "gun":
            act["take"] = near[0]["id"]
        enemies = sorted(
            (m for m in obs["mechs"] if m["kind"] == "mech"),
            key=lambda m: math.dist(m["pos"], me["pos"]),
        )
        yaw = math.atan2(goal[1] - y, goal[0] - x)
        act["move"] = [math.cos(yaw), math.sin(yaw)]
        act["aim"] = [yaw, 0.0]
        act["boost"] = not me["in_zone"]
        act["jump"] = obs["tick"] % 40 == 0
        if enemies:
            tx, ty, tz = enemies[0]["pos"]
            act["aim"] = [
                math.atan2(ty - y, tx - x),
                math.atan2(tz - z - 2.5, math.dist((tx, ty), (x, y))),
            ]
            act["fire"] = [True, True]
            mf.draw.line([x, y, z + 7], [tx, ty, tz + 4.5], "#ff5040")
        mf.draw.point([goal[0], goal[1], z], "#40c0ff", 6)
        mf.draw.label(f"hp {me['hp']:.0f} · {len(enemies)} in sight")
        return act

    def _starter_take(self, obs: dict, me: dict) -> int | None:
        """The starter's rule: with slot 1 empty, take the first weapon in reach unless a gun."""
        near = [p for p in obs["pickups"] if math.dist(p["pos"][:2], me["pos"][:2]) < self.reach]
        if me["weapons"][1] is None and near and near[0]["kind"] != "gun":
            return near[0]["id"]
        return None

    # ------------------------------------------------------------------ bookkeeping
    def _events(self, events: list, t: float) -> None:
        for e in events:
            kind = e["type"]
            if kind == "damaged":
                if e["weapon"] == "npc":
                    self.npc_hurt_t = t
                    if e.get("by") is not None:
                        self.drone_hits[e["by"]] = t
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
            if p["kind"] in self._wanted() and p["kind"] not in held and near:
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

    def _orbit(self, me: dict, zone: dict) -> float:
        """Walk a full-speed circle inside the next safe radius, clear of walls."""
        x, y, _ = me["pos"]
        cx, cy = zone["next_center"]
        nr = zone["next_radius"]
        safe_r = nr - min(20.0, 0.3 * nr) if nr > 0 else max(2.0, 0.4 * zone["radius"])
        dx, dy = x - cx, y - cy
        dist = math.hypot(dx, dy)
        ux, uy = (dx / dist, dy / dist) if dist > 1 else (1.0, 0.0)
        for sign in (self.orbit_sign, -self.orbit_sign):
            tx, ty = -sign * uy, sign * ux
            pull = 0.0 if dist < max(4.0, safe_r - 8) else min(1.5, (dist - (safe_r - 8)) / 8)
            vx, vy = tx - pull * ux, ty - pull * uy
            yaw = self._steer(x, y, math.atan2(vy, vx))
            px = x + 8 * math.cos(yaw)
            py = y + 8 * math.sin(yaw)
            if math.hypot(px - cx, py - cy) <= safe_r:
                self.orbit_sign = sign
                return yaw
        return self._steer(x, y, math.atan2(-uy, -ux))

    def _strafe(self, me: dict, target: dict, zone: dict, t: float) -> float:
        """Circle the target at a preferred range, switching direction at random to spoil leads."""
        x, y, _ = me["pos"]
        if t >= self.strafe_until:
            self.strafe_sign = self.rng.choice((-1, 1))
            self.strafe_until = t + self.rng.uniform(0.6, 1.5)
        held = {w["kind"] for w in me["weapons"] if w}
        if config.KITE:
            keep = KITE_RANGE
        else:
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

    def _near_cube(self, obs: dict, me: dict, t: float) -> tuple[float, float] | None:
        """The nearest cube worth walking to (+4% damage and max HP), even if a mech is nearby."""
        x, y, z = me["pos"]
        best = None
        best_d = 1e9
        best_id = -1
        for cube in obs["cubes"]:
            if cube["id"] in self.skip:
                continue
            px, py = cube["pos"][0], cube["pos"][1]
            if not self._safe_spot(px, py, obs["zone"], t) or self._on_roof(cube["pos"], z):
                continue
            d = math.hypot(px - x, py - y)
            if d <= CUBE_M and d < best_d:
                best_d, best, best_id = d, (px, py), cube["id"]
        if best is None:
            return None
        if best_id != self.cube_id:
            self.cube_id, self.cube_since = best_id, t
        elif t - self.cube_since > CUBE_GIVE_UP_S:
            self.skip.add(best_id)  # cannot get to it: stop trying
        return best

    def _early_weapon(self, obs: dict, me: dict, t: float) -> tuple[float, float] | None:
        """Lightning, else a laser, inside the current circle during the opening."""
        if t >= EARLY_S or None not in me["weapons"]:
            return None
        held = {w["kind"] for w in me["weapons"] if w}
        x, y, z = me["pos"]
        center, radius = obs["zone"]["center"], obs["zone"]["radius"]
        best = None
        best_key = None
        for pickup in obs["pickups"]:
            kind = pickup["kind"]
            if kind not in self._wanted() or kind in held or pickup["id"] in self.skip:
                continue
            px, py = pickup["pos"][0], pickup["pos"][1]
            if math.hypot(px - center[0], py - center[1]) >= radius:
                continue
            if self._on_roof(pickup["pos"], z):
                continue
            d = math.hypot(px - x, py - y)
            if d > 90:
                continue
            key = (0 if kind == "lightning" else 2 if kind == "mortar" else 1, d)
            if best_key is None or key < best_key:
                best_key, best = key, (px, py)
        return best

    def _drop_commit(self, obs: dict, me: dict, mechs: list, t: float) -> dict | None:
        """A supply drop we can reach by the time it lands, and then capture alone."""
        x, y, z = me["pos"]
        walk = max(me["stats"]["walk_speed"], 1.0)
        best = None
        best_key = None
        for drop in obs["drops"]:
            if drop.get("holder") not in (None, me["id"]):
                continue
            px, py = drop["pos"][0], drop["pos"][1]
            if not self._safe_spot(px, py, obs["zone"], t) or self._on_roof(drop["pos"], z):
                continue
            ring = drop.get("radius", 8.0)
            occupied = any(math.hypot(m["pos"][0] - px, m["pos"][1] - py) < ring for m in mechs)
            if occupied and drop.get("holder") != me["id"]:
                continue
            d = math.hypot(px - x, py - y)
            if d > 160 or drop.get("lands_in", 0) > d / walk + 8:
                continue
            key = (drop.get("lands_in", 0), d)
            if best_key is None or key < best_key:
                best_key, best = key, drop
        return best

    def _drop_yaw(self, me: dict, drop: dict) -> float:
        """Walk to the ring, then circle inside it so the 4 s capture is not walked off."""
        x, y, _ = me["pos"]
        px, py = drop["pos"][0], drop["pos"][1]
        ring = drop.get("radius", 8.0)
        if math.hypot(px - x, py - y) > ring - 2:
            return self._toward(me, px, py, False)
        bearing = math.atan2(y - py, x - px)
        return self._steer(x, y, bearing + math.pi / 2)

    # ------------------------------------------------------------------ roof camp
    # Precomputed for this arena at every 8 m roof cell from 5 m to 58 m up (the knob range).
    # Loading it is the startup path; a different map falls back to scanning.
    _ROOF_FILE = Path(__file__).with_name("roofs.dat")
    _ROOF_REC = struct.Struct("<HH16B16BHH")

    def _roof_spots(self) -> list[Roof]:
        """Roof spots worth camping on. This arena loads a table; any other map is scanned."""
        loaded = self._load_roof_table()
        if loaded is not None:
            return loaded
        return self._scan_roof_spots()

    def _load_roof_table(self) -> list[Roof] | None:
        # The table is this arena at an 8 m grid, with sight counted for a 11.7 m camper.
        if ROOF_GRID != 4 or CAMP_HEIGHT_M != 11.7 or CELL != 2.0:
            return None
        nav = self.nav
        path = self._ROOF_FILE
        if not path.is_file():
            return None
        blob = path.read_bytes()
        if len(blob) < 44 or blob[:4] != b"ROOF":
            return None
        tnx, tny = struct.unpack_from("<II", blob, 4)
        digest = blob[12:44]
        count = struct.unpack_from("<I", blob, 44)[0]
        need = 48 + count * self._ROOF_REC.size
        if tnx != nav.tnx or tny != nav.tny or need != len(blob):
            return None
        if hashlib.sha256(nav.heights.tobytes()).digest() != digest:
            return None
        spots = []
        rec = self._ROOF_REC
        for n in range(count):
            i, j, *rest = rec.unpack_from(blob, 48 + n * rec.size)
            d_cells, opens = rest[:16], rest[16:32]
            seen, hides_steps = rest[32], rest[33]
            x, y = nav.center(j * nav.w + i)
            z = nav.top[j * nav.w + i] / 100
            if not ROOF_LOW_M <= z - nav.ground(x, y) <= ROOF_HIGH_M:
                continue
            edges = [(d * CELL, bool(o)) for d, o in zip(d_cells, opens, strict=True)]
            if any(d < ROOF_SETBACK_M for d, _ in edges):
                continue
            back = min(d for d, _ in edges)
            ways = []
            for k, (d, open_ground) in enumerate(edges):
                ax = x + (d + 6) * math.cos(k * math.pi / 8)
                ay = y + (d + 6) * math.sin(k * math.pi / 8)
                walkable = open_ground and not nav.blocked[nav.cell(ax, ay)]
                if walkable and d <= 1.5 * back + 4 and 3 <= z - nav.ground(ax, ay) <= ROOF_HIGH_M:
                    ways.append((ax, ay))
            if ways and hides_steps * CELL >= ROOF_HIDES_M:
                spots.append(Roof(x, y, z, back, seen, tuple(ways)))
        return spots

    def _scan_roof_spots(self) -> list[Roof]:
        """The same catalog, measured from the map. Used when roofs.dat is not this arena."""
        nav = self.nav
        spots = []
        for j in range(2, nav.h - 2, ROOF_GRID):
            for i in range(2, nav.w - 2, ROOF_GRID):
                top = nav.top[j * nav.w + i]
                if top == NO_ROOF:
                    continue
                x, y = nav.center(j * nav.w + i)
                z = top / 100
                if not ROOF_LOW_M <= z - nav.ground(x, y) <= ROOF_HIGH_M:
                    continue
                edges = nav.roof_edges(x, y, ROOF_SETBACK_M)
                if not edges:
                    continue
                back = min(d for d, _ in edges)
                ways = []
                for k, (d, open_ground) in enumerate(edges):
                    # a wall met nearly square on, with walkable ground one jet or less below
                    ax = x + (d + 6) * math.cos(k * math.pi / 8)
                    ay = y + (d + 6) * math.sin(k * math.pi / 8)
                    walkable = open_ground and not nav.blocked[nav.cell(ax, ay)]
                    if (
                        walkable
                        and d <= 1.5 * back + 4
                        and 3 <= z - nav.ground(ax, ay) <= ROOF_HIGH_M
                    ):
                        ways.append((ax, ay))
                if ways:
                    seen, hides = nav.exposure(x, y, CAMP_HEIGHT_M)
                    if hides >= ROOF_HIDES_M:
                        spots.append(Roof(x, y, z, back, seen, tuple(ways)))
        return spots

    @staticmethod
    def _safe_r(zone: dict) -> float:
        """The radius around the next center that the legs keep to."""
        nr = zone["next_radius"]
        return nr - min(20.0, 0.3 * nr) if nr > 0 else max(2.0, 0.4 * zone["radius"])

    def _inside(self, px: float, py: float, zone: dict, margin: float) -> bool:
        """Is the point `margin` meters or more inside the radius the legs keep to?"""
        nc = zone["next_center"]
        return math.hypot(px - nc[0], py - nc[1]) < self._safe_r(zone) - margin

    def _choose_camp(self, me: dict, zone: dict, t: float) -> None:
        """Set `camp` and `way`: the roof spot inside the next circle that the least ground sees,
        marked down for a long walk and for lying far from the center (the circle after this one
        is then more likely to leave it, and coming down among the mechs waiting below is how a
        camper dies), and the nearest ground point to jet up to it from."""
        x, y, _ = me["pos"]
        cx, cy = zone["next_center"]
        safe_r = self._safe_r(zone)
        best, best_way, best_score = None, None, 1e9
        for spot in self.roofs:
            out = math.hypot(spot.x - cx, spot.y - cy)
            score = spot.seen + 0.4 * out
            if out >= safe_r - 4 or score >= best_score:
                continue
            if any(math.hypot(spot.x - bad.x, spot.y - bad.y) < 30 for bad in self.camp_bad):
                continue
            for way in spot.ways:
                total = score + 0.5 * math.hypot(way[0] - x, way[1] - y)
                if total < best_score and math.hypot(way[0] - cx, way[1] - cy) < safe_r - 1:
                    best, best_way, best_score = spot, way, total
        self.camp, self.way = best, best_way
        self.camp_since, self.camp_left, self.lined_up = t, 1e9, False

    def _cover_spot(self, me: dict, zone: dict) -> tuple[float, float] | None:
        """Ground to hold when the next circle has no roof: the nearest walkable point inside it
        with tall buildings on three or more of its eight sides. Looked for out to 48 m from the
        next center (a small circle is the only kind with no roof), every 4 m, or 8 m in a big
        one: 170 points at most."""
        nav = self.nav
        x, y, _ = me["pos"]
        cx, cy = zone["next_center"]
        reach = min(48.0, self._safe_r(zone) - 4.0)
        step = 4 if reach <= 24 else 8
        best, best_d = None, 1e9
        for i in range(-6, 7):
            for j in range(-6, 7):
                px, py = cx + step * i, cy + step * j
                d = math.hypot(px - x, py - y)
                if (
                    math.hypot(step * i, step * j) >= reach
                    or d >= best_d
                    or nav.blocked[nav.cell(px, py)]
                ):
                    continue
                walls = sum(
                    nav.tall[nav.cell(px + 8 * ux, py + 8 * uy)]
                    for ux in (-1, 0, 1)
                    for uy in (-1, 0, 1)
                )
                if walls >= 3:
                    best, best_d = (px, py), d
        return best

    def _camp_plan(self, me: dict, zone: dict, t: float) -> None:
        """Keep `camp` and `way` right for the next circle: when it has left the roof, the best
        roof inside it; when it holds none, ground cover."""
        if self.camp is not None and not self._inside(self.camp.x, self.camp.y, zone, 4.0):
            self.left, self.camp, self.way = self.camp, None, None
        if self.camp is None and self.way and not self._inside(self.way[0], self.way[1], zone, 1.0):
            self.way = None
        nc = zone["next_center"]
        key = (nc[0], nc[1], zone["next_radius"], len(self.camp_bad))
        if self.camp is None and key != self.camp_key:  # a new circle, or a roof given up on
            self.camp_key = key
            self._choose_camp(me, zone, t)
            if self.camp is None:
                self.way = self._cover_spot(me, zone)

    def _roof_item(self, obs: dict, me: dict, roof: float, t: float) -> tuple[float, float] | None:
        """The nearest cube, or weapon for an empty slot, that can be walked to on this roof."""
        x, y, _ = me["pos"]
        items = list(obs["cubes"])
        if None in me["weapons"]:
            held = {w["kind"] for w in me["weapons"] if w}
            items += [
                p for p in obs["pickups"] if p["kind"] in self._wanted() and p["kind"] not in held
            ]
        best, best_d = None, 60.0
        for item in items:
            px, py, pz = item["pos"]
            d = math.hypot(px - x, py - y)
            if d >= best_d or item["id"] in self.skip or abs(pz - roof) > 3:
                continue
            if self.nav.flat(x, y, px, py, roof):
                best, best_d = item, d
        if best is None:
            return None
        if best["id"] != self.roof_item:
            self.roof_item, self.roof_item_since = best["id"], t
        elif t - self.roof_item_since > CUBE_GIVE_UP_S:
            self.skip.add(best["id"])  # cannot get to it: stop trying
        return best["pos"][0], best["pos"][1]

    def _watchers(self, px: float, py: float, pz: float, me: dict, mechs: list) -> int:
        """How well the mechs see me if I stand at this point of a roof: 2 for each whose eye has
        a line to my center (it can hit me), 1 for each that sees no more than my top."""
        size = self.cfg["cubes"]["size_per_cube"]
        height = self.cfg["mech"]["height"] * (1 + size * me["cubes"])
        eye = self.cfg["mech"]["eye_height"]
        seen = 0
        for m in mechs:
            if self._late():
                self._give_up()
                return seen
            ex, ey, ez = m["pos"]
            ez += eye * (1 + size * m.get("cubes", 0))
            if not self.nav.hidden3(px, py, pz + height / 2, ex, ey, ez, 2.0):
                seen += 2
            elif not self.nav.hidden3(px, py, pz + height, ex, ey, ez, 2.0):
                seen += 1
        return seen

    def _hide_spot(self, me: dict, mechs: list, roof: float, here: int) -> tuple[float, float, int]:
        """(x, y, how seen): where within 24 m on this roof the mechs see me least; where I am
        (seen `here`), unless somewhere is better. A place counts only with 4 m of roof around."""
        nav = self.nav
        x, y, _ = me["pos"]
        best = (x, y, here)
        best_cost = here * 100.0
        for k in range(8):
            ux, uy = math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)
            for d in (8.0, 16.0, 24.0):
                px, py = x + d * ux, y + d * uy
                if not (
                    nav.flat(x, y, px, py, roof)
                    and nav.flat(px - 4, py, px + 4, py, roof)
                    and nav.flat(px, py - 4, px, py + 4, roof)
                ):
                    break
                seen = self._watchers(px, py, roof, me, mechs)
                if seen * 100 + d < best_cost:
                    best, best_cost = (px, py, seen), seen * 100 + d
        return best

    def _hold(self, obs: dict, me: dict, mechs: list, camp: Roof, t: float) -> float | None:
        """Heading on the camp's roof, or None to stand. Unseen: stand on the spot, or fetch what
        lies up here. Seen: circle the spot, so that shots miss, and shoot back. Hurt: to where
        the fewest mechs see me, and hold fire there until healed."""
        x, y, _ = me["pos"]
        frac = me["hp"] / me["max_hp"]
        if frac < HIDE_HP:
            self.healing = True
        elif frac >= HEALED_HP:
            self.healing = False
        near = sorted((m for m in mechs if m["_d"] < 230), key=lambda m: m["_d"])[:4]
        here = self._watchers(x, y, camp.z, me, near)
        if self.healing:
            self.lying_low = not here
            if not here:
                return None
            if t >= self.hide_until:
                self.hide_at, self.hide_until = self._hide_spot(me, near, camp.z, here), t + 0.5
            gx, gy, seen = self.hide_at
            radius = 3.0
        else:
            item = self._roof_item(obs, me, camp.z, t)
            if item is not None:
                return math.atan2(item[1] - y, item[0] - x)
            gx, gy, seen = camp.x, camp.y, here
            radius = min(5.0, camp.back - 3.0)
        d = math.hypot(gx - x, gy - y)
        if not seen or d > radius + 3:
            return None if d < 1.5 else math.atan2(gy - y, gx - x)
        tangent = math.atan2(y - gy, x - gx) + math.pi / 2
        return tangent + max(-0.8, min(0.8, 0.3 * (d - radius)))

    def _camp(
        self, obs: dict, me: dict, mechs: list, zone: dict, t: float
    ) -> tuple[bool, float | None]:
        """(camping?, heading or None to stand). To the ground below the chosen roof, straight at
        its wall and up, then to the spot set back from its edges; once up, only as far as what
        lies on the same roof. A roof that the next circle has left is held for as long as the
        zone allows (the legs' zone rule ends it); then to the next roof, or to ground cover."""
        nav = self.nav
        x, y, z = me["pos"]
        camp = self.camp
        self.camp_boost = False
        over = nav.roof(x, y)
        level = camp is not None and over is not None and abs(over - camp.z) < 1.5
        if level and z > camp.z - 1.0 and nav.flat(x, y, camp.x, camp.y, camp.z):
            self.camp_since = t  # on its roof, or about to land on it
            return True, self._hold(obs, me, mechs, camp, t)
        old = self.left
        if old is not None and self.aloft and nav.flat(x, y, old.x, old.y, old.z):
            return True, self._hold(obs, me, mechs, old, t)  # mechs wait below: no sooner than due
        if camp is None:
            if self.way is None:
                return False, None
            wx, wy = self.way
            if math.hypot(wx - x, wy - y) > 4:
                return True, self._toward(me, wx, wy, True)
            return True, math.atan2(y - wy, x - wx) + math.pi / 2 + 0.3  # circle it: keep moving
        left = math.hypot(camp.x - x, camp.y - y) + max(0.0, camp.z - z)
        if left < self.camp_left - 2:
            self.camp_left, self.camp_since = left, t
        elif t - self.camp_since > CAMP_GIVE_UP_S:
            self.camp_bad.append(camp)  # cannot get up there: another roof next decision
            self.camp = self.way = None
            if len(self.camp_bad) >= CAMP_TRIES:
                self.roofs = []  # climbing does not work here: fight on the ground from now on
            return False, None
        wx, wy = self.way
        d_way = math.hypot(wx - x, wy - y)
        if d_way < 4:
            self.lined_up = True
        elif d_way > 30:
            self.lined_up = False
        if not self.lined_up:
            # mechs close behind: run, and stop in time for the energy to come back for the climb
            chased = any(m["_d"] < 80 for m in mechs)
            self.camp_boost = chased and d_way > 60 and me["energy"] > 40 and bool(me["grounded"])
            return True, self._toward(me, wx, wy, True)
        yaw = math.atan2(camp.y - y, camp.x - x)  # straight at the wall, then up it
        below = self.climb_to is None and me["grounded"] and z < camp.z - 1.0
        if below and not nav.clear(x, y, x + 6 * math.cos(yaw), y + 6 * math.sin(yaw)):  # at it
            if me["energy"] < CLIMB_ENERGY:
                return True, None  # wait for the energy to make it in one jet
            self.climb_to = camp.z
        if self.climb_to is not None:
            self.climb_yaw = yaw
        return True, yaw

    def _swarmed(self, npcs: list, x: float, y: float, t: float) -> bool:
        """Are two or more drones shooting at me? Once they are, stay clear for SWARM_S."""
        reach = self.cfg["npc"]["deaggro_range"]
        on_me = 0
        for npc in npcs:
            to_me = math.atan2(y - npc["pos"][1], x - npc["pos"][0])
            aimed = npc["_d"] < reach and abs(wrap(npc["aim"][0] - to_me)) < 0.25
            on_me += aimed or t - self.drone_hits.get(npc["id"], -99) < 3
        if on_me >= 2:
            self.swarm_until = t + SWARM_S
        return t < self.swarm_until

    def _loot(self, obs: dict, me: dict, npcs: list, t: float, crates: bool, drones: bool):
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
                if p["kind"] in self._wanted() and p["kind"] not in held:
                    consider(p, 90, 0, -30)
        for cube in obs["cubes"]:
            consider(cube, 70, 0, -20)
        for drop in obs["drops"]:
            if drop["lands_in"] < 12 and drop.get("holder") in (None, me["id"]):
                consider(drop, 130, 1.5, -60)
        if crates:
            for crate in obs["crates"]:
                consider(crate, 80, 30, 0)
        if drones:
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
    def _farm_target(self, obs: dict, me: dict, npcs: list, t: float, crates: bool, drones: bool):
        """A drone or crate worth shooting for its cubes while no mech is around."""
        x, y, z = me["pos"]
        options = []
        for npc in npcs:
            if npc["_d"] < 70 and (drones or t - self.npc_hurt_t < 2):
                options.append((npc["_d"], npc, "npc"))
        if crates:
            for crate in obs["crates"]:
                d = math.hypot(crate["pos"][0] - x, crate["pos"][1] - y)
                if self.aloft:  # only the crates up here: a cube on the ground is out of reach
                    level = abs(crate["pos"][2] - z) < 4
                else:
                    level = not self._on_roof(crate["pos"], z)
                if d < 60 and crate["id"] not in self.skip and level:
                    options.append((d + 20, crate, "crate"))
        for _, item, kind in sorted(options, key=lambda o: o[0]):
            item["kind"] = kind
            if self._sees(me, item):
                item["_d"] = math.hypot(item["pos"][0] - x, item["pos"][1] - y)
                return item
        return None

    def _sees(self, me: dict, target: dict) -> bool:
        """Is the line to the target's center clear of buildings? On the ground the flat sight
        grid answers; standing on a roof it reads everything as hidden, so heights are used."""
        nav = self.nav
        if nav is None:
            return True
        x, y, z = me["pos"]
        tx, ty, tz = target["pos"]
        if not self.aloft:
            return not nav.hidden(x, y, tx, ty)
        grow = 1 + self.cfg["cubes"]["size_per_cube"] * target.get("cubes", 0)
        tz += self.half_height[target["kind"]] * grow
        return not nav.hidden3(x, y, z + me["stats"]["muzzle_height"], tx, ty, tz)

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
        grow = 1 + self.cfg["cubes"]["size_per_cube"] * aim_at.get("cubes", 0)
        half_w = self.half_width[aim_at["kind"]] * grow
        half_h = self.half_height[aim_at["kind"]] * grow
        d = aim_at["_d"]
        seen = self._sees(me, aim_at)
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
            elif (
                kind == "mortar"
                and config.MORTAR
                and not self._fallback
                and target is not None
                and MORTAR_MIN <= d <= MORTAR_MAX
            ):
                yaw = math.atan2(target["pos"][1] - me["pos"][1], target["pos"][0] - me["pos"][0])
                pitch = self._mortar_pitch(d)
                aim = [yaw, pitch]
                fire[slot] = seen and on_target((yaw, pitch, d), 0.02) and me["in_zone"]
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

    def _wanted(self) -> tuple[str, ...]:
        return ("laser", "lightning", "mortar") if config.MORTAR else WANTED

    def _aimed_at_us(self, mech: dict, x: float, y: float) -> bool:
        ex, ey = mech["pos"][0], mech["pos"][1]
        dist = max(mech.get("_d", math.hypot(ex - x, ey - y)), 1.0)
        return abs(wrap(mech["aim"][0] - math.atan2(y - ey, x - ex))) < max(
            0.12, math.atan2(3.0, dist)
        )

    def _mortar_pitch(self, dist: float) -> float:
        """The low arc that lands a shell at `dist`: range is v² sin(2·pitch) / g."""
        st = self.cfg["weapons"]["mortar"]
        v, g = st["speed"], st["gravity"]
        return 0.5 * math.asin(min(0.999, max(0.0, dist * g / (v * v))))

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

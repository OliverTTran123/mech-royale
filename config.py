"""Feature flags, one per heuristic. Turning a flag off falls back to what the kit's starter bot
(example_bot.py) does for that part; with all four off the bot plays exactly like the starter.
"""

# Legs toward the zone. On: reach the next circle before the shrink, following the map's flow
# field, standing still once inside, jetting over what blocks the way.
# Off (starter): walk straight at the next center, boost outside the zone, hop every 2 s.
ZONE = True

# Legs around enemies. On: back off behind cover when outnumbered or losing, circle a single enemy
# at range, run out from under mortar shells. Off (starter): enemies never change where it walks.
THREAT_RETREAT = True

# Loot. On: walk to cubes, a laser or lightning and supply drops when no mech is near, shoot crates
# and drones for cubes when healthy, never take a mortar, rank upgrades.
# Off (starter): take the first non-gun weapon in reach, always pick upgrade 0, never detour.
LOOT = True

# Torso. On: aim at the mech that is cheapest to kill, lead the shot, fire only when aligned, charge
# and release the laser, reload between fights, look around when nothing is in sight.
# Off (starter): aim straight at the nearest mech and hold both triggers; else face the next center.
TARGETING = True

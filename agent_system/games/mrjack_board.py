"""London 2016 base board: factual hex layout checked against publisher setup p.7.

Cell numbering is column-major, calibrated against Aghajari/Mr.Jack's Apache-2.0
MrJack_Map.c. No SDL code, artwork or AI is included. See MRJACK_RULES.md.
"""
COLUMNS=((5,5),(7,2),(6,3),(6,2),(7,1),(8,0),(8,1),(8,2),(7,3),(6,4),(6,3),(7,2),(5,3))
COORDS=tuple((x,start+2*y) for x,(count,start) in enumerate(COLUMNS) for y in range(count))
LOOKUP={p:i for i,p in enumerate(COORDS)}
DIRECTIONS=((0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1))
NEIGHBORS=tuple(tuple(LOOKUP.get((x+dx,y+dy),-1) for dx,dy in DIRECTIONS) for x,y in COORDS)
BUILDINGS=frozenset((7,8,18,14,20,22,33,35,37,48,50,52,63,65,67,71,77,78))
LAMP_SITES=frozenset((10,12,32,36,49,53,73,75))
SEWERS=frozenset((6,11,15,31,43,54,70,74,79))
STREETS=frozenset(range(86))-BUILDINGS-LAMP_SITES
EXITS=(5,11,74,80)
NAMES=("Lestrade","Stealthy","Bert","Watson","Holmes","Smith","Goodley","Gull")
INITIAL_POSITIONS=(28,2,57,61,44,41,83,24)
INITIAL_LIGHTS=(36,10,75,12,73,49)

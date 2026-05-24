import sys, os, math, random
import numpy as np
import pygame
import pygame.gfxdraw
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Deque, Optional
from enum import Enum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#  SECTION 1 — LAYOUT CONSTANTS
W, H         = 1280, 720
HEADER_H     = 58
FOOTER_H     = 42
LEFT_W       = 245
RIGHT_W      = 318
CTR_X        = LEFT_W
CTR_W        = W - LEFT_W - RIGHT_W   # 717
RIGHT_X      = W - RIGHT_W            # 962
CONTENT_Y    = HEADER_H
CONTENT_H    = H - HEADER_H - FOOTER_H  # 620
FOOTER_Y     = H - FOOTER_H
FPS          = 60

# Panel anchors (left panel)
INET_POS     = (LEFT_W // 2, CONTENT_Y + 48)
LB_POS       = (LEFT_W // 2, CONTENT_Y + 200)
QUEUE_POS    = (LEFT_W // 2, CONTENT_Y + 352)

# Rack / server geometry (4 racks, 5 servers each)
RACK_COUNT   = 4
SRV_PER_RACK = 5
N_SRV        = RACK_COUNT * SRV_PER_RACK  # 20

_RACK_MARGIN = 10
_RACK_GAP    = 12
_AVAIL_W     = CTR_W - 2 * _RACK_MARGIN - (RACK_COUNT - 1) * _RACK_GAP  # 833
RACK_W       = _AVAIL_W // RACK_COUNT    # 208
RACK_H       = CONTENT_H - 16
RACK_HDR_H   = 34
SRV_W        = RACK_W - 12
SRV_GAP      = 9
SRV_H        = (RACK_H - RACK_HDR_H - (SRV_PER_RACK - 1) * SRV_GAP) // SRV_PER_RACK
# SRV_H ≈ (748-34-36)//5 = 678//5 = 135


def rack_rect(i: int) -> pygame.Rect:
    x = CTR_X + _RACK_MARGIN + i * (RACK_W + _RACK_GAP)
    return pygame.Rect(x, CONTENT_Y + 8, RACK_W, RACK_H)


def server_rect(rack_i: int, slot_i: int) -> pygame.Rect:
    rx = CTR_X + _RACK_MARGIN + rack_i * (RACK_W + _RACK_GAP)
    x  = rx + 6
    y  = CONTENT_Y + 8 + RACK_HDR_H + slot_i * (SRV_H + SRV_GAP)
    return pygame.Rect(x, y, SRV_W, SRV_H)

#  SECTION 2 — COLOR PALETTE  (Cyberpunk / Dark Pro)
class C:
    # Backgrounds
    BG            = (  5,  9, 20)
    PANEL         = ( 11, 17, 36)
    PANEL2        = ( 15, 22, 44)
    BORDER        = ( 25, 42, 80)
    BORDER_BRT    = ( 48, 88,158)

    # Server states
    SRV_OFF       = ( 16, 20, 38)
    SRV_BOOT      = ( 22, 42, 76)
    SRV_IDLE      = ( 14, 52, 34)
    SRV_NORMAL    = ( 18,105, 48)
    SRV_HIGH      = (140, 82, 14)
    SRV_CRITICAL  = (158, 20, 20)

    # Text
    TEXT          = (182, 202, 242)
    TEXT_DIM      = ( 70,  90,130)
    TEXT_BRT      = (218, 232, 255)
    TEXT_LBL      = (115, 145,195)

    # Accents
    BLUE          = ( 50, 130, 255)
    CYAN          = (  0, 212, 242)
    GREEN         = ( 42, 210,  92)
    ORANGE        = (248, 142,  24)
    RED           = (248,  52,  52)
    PURPLE        = (172,  72, 252)
    YELLOW        = (252, 212,  42)
    TEAL          = (  0, 185, 175)
    PINK          = (255,  80, 160)

    # UI
    HDR_BG        = (  7, 11, 26)
    FTR_BG        = (  7, 11, 26)
    DIVIDER       = ( 20, 35, 65)
    CHART_BG      = (  8, 13, 28)
    CHART_GRID    = ( 18, 30, 55)

    # Packets
    PKT_NORMAL    = (  0, 165, 252)
    PKT_BURST     = (252, 138,   0)
    PKT_ATTACK    = (252,  38,  38)

    # RL decisions
    DEC_UP        = ( 38, 210,  88)
    DEC_DOWN      = (248, 138,  24)
    DEC_IDLE      = ( 50, 130, 255)
    DEC_ALERT     = (252,  38,  38)
    DEC_MAINT     = (172,  72, 252)

    @staticmethod
    def lerp(c1, c2, t: float):
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    @staticmethod
    def cpu_color(cpu: float):
        if cpu < 0.5:
            return C.lerp(C.GREEN, C.YELLOW, cpu / 0.5)
        return C.lerp(C.YELLOW, C.RED, (cpu - 0.5) / 0.5)

    @staticmethod
    def alpha(color, a: int):
        return (*color[:3], a)


#  SECTION 3 — STATE DATACLASSES

class SrvSt(Enum):
    OFF = 0; BOOT = 1; IDLE = 2; NORMAL = 3; HIGH = 4; CRITICAL = 5


@dataclass
class Server:
    sid:    int
    rack:   int
    slot:   int
    active: bool  = False
    cpu:    float = 0.0
    ram:    float = 0.0
    jobs:   int   = 0
    boot_t: float = 0.0   # 0→1 boot animation
    pulse:  float = 0.0   # oscillating animation phase
    flash:  float = 0.0   # flash when packet arrives

    @property
    def state(self) -> SrvSt:
        if not self.active:
            return SrvSt.BOOT if self.boot_t > 0.01 else SrvSt.OFF
        c = self.cpu
        if c < 0.22: return SrvSt.IDLE
        if c < 0.56: return SrvSt.NORMAL
        if c < 0.78: return SrvSt.HIGH
        return SrvSt.CRITICAL

    @property
    def bg_color(self):
        _MAP = {SrvSt.OFF: C.SRV_OFF, SrvSt.BOOT: C.SRV_BOOT,
                SrvSt.IDLE: C.SRV_IDLE, SrvSt.NORMAL: C.SRV_NORMAL,
                SrvSt.HIGH: C.SRV_HIGH, SrvSt.CRITICAL: C.SRV_CRITICAL}
        base = _MAP[self.state]
        if self.flash > 0:
            return C.lerp(base, (255, 255, 255), self.flash * 0.3)
        return base

    @property
    def border_color(self):
        s = self.state
        if s in (SrvSt.OFF, SrvSt.BOOT):  return C.BORDER
        if s == SrvSt.IDLE:               return C.lerp(C.SRV_NORMAL, C.GREEN, 0.3)
        if s == SrvSt.NORMAL:             return C.GREEN
        if s == SrvSt.HIGH:               return C.ORANGE
        return C.RED

    @property
    def label(self) -> str:
        labels = {SrvSt.OFF: "OFF", SrvSt.BOOT: "BOOT…",
                  SrvSt.IDLE: "IDLE", SrvSt.NORMAL: "RUN",
                  SrvSt.HIGH: "HIGH", SrvSt.CRITICAL: "CRIT!"}
        return labels[self.state]


@dataclass
class Packet:
    pid:       int
    x:         float
    y:         float
    tx:        float
    ty:        float
    phase:     str   = 'to_lb'   # to_lb / to_queue / in_queue / to_server / done
    speed:     float = field(default_factory=lambda: random.uniform(120, 240))
    color:     tuple = field(default_factory=lambda: C.PKT_NORMAL)
    size:      int   = field(default_factory=lambda: random.randint(2, 4))
    alpha:     float = 255.0
    server_id: int   = -1
    age:       float = 0.0
    trail:     list  = field(default_factory=list)


@dataclass
class RLDecision:
    icon:      str
    main_text: str
    sub_text:  str
    color:     tuple
    life:      float = 3.8
    age:       float = 0.0


#  SECTION 4 — SIMULATION STATE

class SimState:
    def __init__(self):
        # 4 racks × 5 servers
        self.servers: List[Server] = [
            Server(sid=r * 5 + s, rack=r, slot=s)
            for r in range(RACK_COUNT)
            for s in range(SRV_PER_RACK)
        ]
        # Boot 3 servers initially
        for s in self.servers[:3]:
            s.active = True
            s.boot_t = 1.0
            s.cpu    = random.uniform(0.30, 0.52)
            s.ram    = random.uniform(0.40, 0.62)
            s.jobs   = random.randint(4, 14)

        # Packets
        self.packets: List[Packet] = []
        self._pid       = 0
        self._arr_acc   = 0.0
        self.arrival_rate = 1.8   # pkts/sec

        # Queue
        self.queue_len = 0
        self.max_queue = 60

        # Metrics
        self.latency    = 52.0
        self.throughput = 0.0
        self.reward     = 0.0
        self.sla_violations = 0
        self.total_steps    = 0

        # Metric histories (200 pts)
        N = 200
        self.cpu_hist    = deque([0.40] * N, maxlen=N)
        self.lat_hist    = deque([52.0] * N, maxlen=N)
        self.queue_hist  = deque([0.0]  * N, maxlen=N)
        self.reward_hist = deque([0.0]  * N, maxlen=N)
        self.nodes_hist  = deque([3.0]  * N, maxlen=N)
        self.tp_hist     = deque([0.0]  * N, maxlen=N)

        # RL decisions display
        self.decisions: List[RLDecision] = []
        self.dec_history: Deque[tuple]   = deque(maxlen=6)

        # Server screen positions (set by renderer after first draw)
        self.server_centers: Dict[int, Tuple[float, float]] = {}

        # Timing
        self.sim_time  = 0.0
        self._rl_t     = 0.0
        self.rl_interval = 2.2   # seconds between agent decisions
        self._thinking = 0.0     # thinking animation timer

        # Scenario
        self.scenario  = "normal"
        self._scen_t   = 0.0

        # Control
        self.paused = False
        self.speed  = 1.0

        # Stats
        self.total_decisions = 0
        self.upscale_count   = 0
        self.downscale_count = 0

    # Properties 

    @property
    def n_active(self) -> int:
        return sum(1 for s in self.servers if s.active)

    @property
    def avg_cpu(self) -> float:
        active = [s for s in self.servers if s.active]
        return sum(s.cpu for s in active) / len(active) if active else 0.0

    @property
    def avg_ram(self) -> float:
        active = [s for s in self.servers if s.active]
        return sum(s.ram for s in active) / len(active) if active else 0.0

    # Control 

    def activate(self, n: int = 1):
        c = 0
        for s in self.servers:
            if not s.active and c < n:
                s.active = True; s.boot_t = 0.0; c += 1

    def deactivate(self, n: int = 1):
        active = sorted([s for s in self.servers if s.active], key=lambda s: s.cpu)
        c = 0
        for s in active:
            if self.n_active - c > 1 and c < n:
                s.active = False; c += 1

    def add_decision(self, icon: str, main: str, sub: str = "", color=C.DEC_IDLE):
        self.decisions.append(RLDecision(icon, main, sub, color))
        self.dec_history.appendleft((icon, main, sub, color))
        self.total_decisions += 1

    def reset(self):
        self.__init__()

    def set_scenario(self, name: str):
        self.scenario = name
        self._scen_t  = 0.0

    # Main update

    def update(self, dt: float):
        if self.paused:
            return
        dt *= self.speed
        self.sim_time  += dt
        self._scen_t   += dt
        self._rl_t     += dt
        self._thinking  = max(0.0, self._thinking - dt)
        self.total_steps += 1

        # Update servers
        for s in self.servers:
            if s.active:
                s.boot_t = min(1.0, s.boot_t + dt * 0.65)
                s.pulse  = (s.pulse + dt * 2.8) % (2 * math.pi)
            else:
                s.boot_t = max(0.0, s.boot_t - dt * 1.4)
                s.cpu    = max(0.0, s.cpu  - dt * 0.75)
                s.ram    = max(0.0, s.ram  - dt * 0.32)
                s.jobs   = max(0, s.jobs - int(dt * 8 + 0.5))
            s.flash = max(0.0, s.flash - dt * 3.0)

        # Scenario load
        self._scenario_update(dt)

        # Spawn packets
        self._arr_acc += self.arrival_rate * dt
        while self._arr_acc >= 1.0:
            self._spawn_packet()
            self._arr_acc -= 1.0

        # Move packets
        self._update_packets(dt)

        # Age-out decisions
        for d in list(self.decisions):
            d.age += dt
            if d.age > d.life:
                self.decisions.remove(d)

        # Metrics
        if self.n_active > 0:
            self.latency = 34.0 + self.avg_cpu * 108.0 + self.queue_len * 1.8
            if self.n_active == 1:
                self.latency *= 1.7
        else:
            self.latency = 999.0

        n_in_flight = sum(1 for p in self.packets if p.phase == 'to_server')
        self.throughput = n_in_flight * 2.8

        sla_ok = self.latency <= 200 and self.queue_len < 45
        if not sla_ok:
            self.sla_violations += 1

        eff = 1.0 if 0.50 <= self.avg_cpu <= 0.80 else 0.0
        self.reward = (
            -self.avg_cpu * 0.2
            + min(1.0, self.throughput / 20.0) * 0.6
            - (0.0 if sla_ok else 0.8)
            + eff * 0.3
        )

        # History sample
        if self.total_steps % 24 == 0:
            self.cpu_hist.append(self.avg_cpu)
            self.lat_hist.append(min(350, self.latency))
            self.queue_hist.append(float(self.queue_len))
            self.reward_hist.append(self.reward)
            self.nodes_hist.append(float(self.n_active))
            self.tp_hist.append(self.throughput)

    # Packet helpers
    def _spawn_packet(self):
        ix, iy = INET_POS
        lx, ly = LB_POS
        scen = self.scenario
        color = (C.PKT_ATTACK if scen == 'attack'
                 else C.PKT_BURST if scen in ('overload', 'spike')
                 else C.PKT_NORMAL)
        p = Packet(pid=self._pid,
                   x=float(ix + random.randint(-32, 32)),
                   y=float(iy + 28),
                   tx=float(lx), ty=float(ly),
                   color=color)
        self._pid += 1
        self.packets.append(p)
        if len(self.packets) > 160:
            self.packets.pop(0)

    def _update_packets(self, dt: float):
        qx, qy = QUEUE_POS
        active = [s for s in self.servers if s.active and s.boot_t > 0.88]
        to_rm  = []

        for p in self.packets:
            p.age += dt
            p.trail.append((p.x, p.y))
            if len(p.trail) > 7:
                p.trail.pop(0)

            if p.phase == 'to_lb':
                p.x, p.y, done = _mv(p.x, p.y, p.tx, p.ty, p.speed, dt)
                if done:
                    p.phase = 'to_queue'
                    p.tx    = qx + random.randint(-10, 10)
                    p.ty    = qy

            elif p.phase == 'to_queue':
                p.x, p.y, done = _mv(p.x, p.y, p.tx, p.ty, p.speed, dt)
                if done:
                    if active and self.queue_len < self.max_queue:
                        self.queue_len += 1
                        p.phase = 'in_queue'
                        p.age   = 0.0
                    else:
                        p.phase = 'done'
                        p.alpha = 0.0   # dropped

            elif p.phase == 'in_queue':
                wait = max(0.06, self.latency / 320.0)
                if p.age > wait and active:
                    best = min(active, key=lambda s: s.cpu)
                    if best.sid in self.server_centers:
                        cx, cy = self.server_centers[best.sid]
                        p.tx       = cx + random.randint(-4, 4)
                        p.ty       = cy + random.randint(-4, 4)
                        p.server_id = best.sid
                        p.phase    = 'to_server'
                        p.age      = 0.0
                        self.queue_len = max(0, self.queue_len - 1)

            elif p.phase == 'to_server':
                p.x, p.y, done = _mv(p.x, p.y, p.tx, p.ty, p.speed * 1.5, dt)
                if done:
                    srv = next((s for s in self.servers if s.sid == p.server_id), None)
                    if srv and srv.active:
                        srv.jobs  = min(99, srv.jobs + 1)
                        srv.flash = 0.6
                    p.phase = 'done'
                    p.alpha = 160.0

            elif p.phase == 'done':
                p.alpha -= dt * 260
                if p.alpha <= 0:
                    to_rm.append(p)

        for p in to_rm:
            if p in self.packets:
                self.packets.remove(p)

    # Scenario

    def _scenario_update(self, dt: float):
        t     = self._scen_t
        s     = self.scenario
        active = [sv for sv in self.servers if sv.active and sv.boot_t > 0.90]
        if not active:
            return

        if s == 'normal':
            base = 0.42 + 0.14 * math.sin(t * 0.22) + 0.06 * math.sin(t * 0.75)
            self.arrival_rate = 1.8
        elif s == 'spike':
            phase = (t % 20.0) < 4.5
            base  = 0.87 if phase else 0.38
            self.arrival_rate = 5.8 if phase else 1.5
        elif s == 'overload':
            base = 0.91 + random.gauss(0, 0.025)
            self.arrival_rate = 9.0
        elif s == 'attack':
            if   t < 7.0:  base = 0.97; self.arrival_rate = 14.0
            elif t < 20.0: base = 0.78; self.arrival_rate =  5.5
            else:          base = 0.42; self.arrival_rate =  1.8
        elif s == 'recovery':
            if t < 4.5:
                base = 0.93
                self.arrival_rate = 7.5
            else:
                base = max(0.30, 0.93 - (t - 4.5) * 0.054)
                self.arrival_rate = max(1.5, 7.5 - (t - 4.5) * 0.38)
        else:
            base = 0.42

        for sv in active:
            tgt    = max(0.05, min(0.99, base + random.gauss(0, 0.06)))
            sv.cpu = sv.cpu + (tgt - sv.cpu)  * min(1.0, dt * 1.9)
            sv.ram = sv.ram + (tgt * 0.72 + random.gauss(0, 0.03) - sv.ram) * min(1.0, dt * 0.9)
            sv.ram = max(0.08, min(0.99, sv.ram))
            sv.jobs = max(0, min(99, int(sv.cpu * 28 + random.gauss(0, 2))))


#  SECTION 5 — RL AGENT BRIDGE

ACTION_META = {
    0: ("⏸",  "IDLE",          "Optimal — aucune action",          C.DEC_IDLE),
    1: ("⬆",  "SCALE UP  +1",  "Ajout 1 nœud pour absorber charge",C.DEC_UP),
    2: ("⬆⬆", "SCALE UP  +2",  "Charge critique — 2 nœuds ajoutés",C.DEC_UP),
    3: ("⬇",  "SCALE DOWN −1", "Charge réduite — 1 nœud libéré",   C.DEC_DOWN),
    4: ("⬇⬇", "SCALE DOWN −2", "Économie max — 2 nœuds éteints",   C.DEC_DOWN),
    5: ("↔",  "MIGRATE",       "Rééquilibrage de la charge",        C.DEC_MAINT),
}


class RLBridge:
    def __init__(self, ckpt_path: Optional[str] = None):
        self.agent    = None
        self.use_real = False
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                from rl_autoscaling.utils.config import get_default_config
                from rl_autoscaling.agent.dqn_agent import DQNAgent
                cfg = get_default_config()
                self.agent = DQNAgent(cfg)
                self.agent.load(ckpt_path)
                self.agent.online_net.eval()
                self.use_real = True
                print(f"[RLBridge] ✓ Real agent loaded — {ckpt_path}")
            except Exception as e:
                print(f"[RLBridge] ✗ {e}  →  heuristic agent")
        else:
            print("[RLBridge] Heuristic agent (no checkpoint)")

    def get_action(self, state: SimState) -> int:
        if self.use_real and self.agent:
            obs = self._make_obs(state)
            return self.agent.select_action(obs, training=False)
        return self._heuristic(state)

    def _make_obs(self, state: SimState) -> np.ndarray:
        cpu = state.avg_cpu
        ram = state.avg_ram
        lat = min(1.0, state.latency / 300.0)
        q   = state.queue_len / max(1, state.max_queue)
        tp  = min(1.0, state.throughput / 30.0)
        feat = np.array([cpu, ram, lat, q, tp, 0.0, 0.0, cpu * ram], dtype=np.float32)
        return np.tile(feat, 8)   # obs_dim = 64

    def _heuristic(self, state: SimState) -> int:
        cpu = state.avg_cpu
        q   = state.queue_len
        n   = state.n_active
        lat = state.latency
        if cpu > 0.83 or q > 32 or lat > 185:  return 2
        if cpu > 0.70 or q > 16:               return 1
        if cpu < 0.21 and q < 3 and n > 3:     return 4
        if cpu < 0.32 and q < 6 and n > 2:     return 3
        if q > 12 and cpu < 0.60:              return 5
        return 0


#  SECTION 6 — DRAWING UTILITIES

def _mv(x, y, tx, ty, spd, dt) -> Tuple[float, float, bool]:
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy)
    if d < spd * dt or d < 0.5:
        return tx, ty, True
    f = spd * dt / d
    return x + dx * f, y + dy * f, False


def _rr(surf, color, rect, r=8, bw=0, bc=None):
    """Draw rounded rectangle with optional border."""
    pygame.draw.rect(surf, color, rect, border_radius=r)
    if bw > 0 and bc:
        pygame.draw.rect(surf, bc, rect, bw, border_radius=r)


def _bar(surf, rect, frac, fg, bg=(20, 30, 52), r=3):
    """Draw horizontal progress bar."""
    frac = max(0.0, min(1.0, frac))
    pygame.draw.rect(surf, bg, rect, border_radius=r)
    if frac > 0:
        fr = pygame.Rect(rect.x, rect.y, max(r * 2, int(rect.w * frac)), rect.h)
        pygame.draw.rect(surf, fg, fr, border_radius=r)


def _txt(surf, text, font, color, x, y, anchor='topleft'):
    """Blit text with optional anchor."""
    s = font.render(text, True, color)
    r = s.get_rect()
    setattr(r, anchor, (x, y))
    surf.blit(s, r)
    return r


def _glow(surf, cx, cy, radius, color, alpha=55):
    """Draw radial glow using SRCALPHA surface."""
    s = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
    for i in range(radius, 0, -max(1, radius // 6)):
        a = int(alpha * (i / radius) ** 0.7)
        pygame.draw.circle(s, (*color[:3], a), (radius + 1, radius + 1), i)
    surf.blit(s, (cx - radius - 1, cy - radius - 1))


def _chart(surf, rect, data, line_col, min_v=0.0, max_v=1.0,
           fill=True, threshold=None, thr_col=None):
    """Draw a rolling line chart."""
    pygame.draw.rect(surf, C.CHART_BG, rect, border_radius=4)
    # Grid
    for i in range(1, 4):
        gy = rect.y + rect.h * i // 4
        pygame.draw.line(surf, C.CHART_GRID, (rect.x, gy), (rect.right, gy))
    # Threshold line
    if threshold is not None:
        t  = (threshold - min_v) / max(0.001, max_v - min_v)
        ty = rect.y + rect.h - int(t * rect.h)
        pygame.draw.line(surf, thr_col or C.RED, (rect.x, ty), (rect.right, ty), 1)
    dlist = list(data)
    if len(dlist) < 2:
        pygame.draw.rect(surf, C.BORDER, rect, 1, border_radius=4)
        return
    pts = []
    for i, v in enumerate(dlist):
        px = rect.x + int(i * (rect.w - 1) / max(1, len(dlist) - 1))
        t  = (v - min_v) / max(0.001, max_v - min_v)
        py = rect.y + rect.h - int(t * rect.h)
        py = max(rect.y + 1, min(rect.bottom - 1, py))
        pts.append((px, py))
    if fill:
        poly = [(rect.x, rect.bottom)] + pts + [(rect.right, rect.bottom)]
        fs   = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        lp   = [(p[0] - rect.x, p[1] - rect.y) for p in poly]
        pygame.draw.polygon(fs, (*line_col[:3], 38), lp)
        surf.blit(fs, rect.topleft)
    pygame.draw.lines(surf, line_col, False, pts, 2)
    pygame.draw.rect(surf, C.BORDER, rect, 1, border_radius=4)


#  SECTION 7 — RENDERER

class Renderer:
    def __init__(self):
        pygame.font.init()
        pref = ['Consolas', 'Courier New', 'Monaco', 'DejaVu Sans Mono', None]
        def mf(sz, bold=False):
            for name in pref:
                try:
                    if name:
                        f = pygame.font.SysFont(name, sz, bold=bold)
                    else:
                        f = pygame.font.Font(None, sz)
                    return f
                except Exception:
                    pass
            return pygame.font.Font(None, sz)

        self.f_big   = mf(26, bold=True)
        self.f_lg    = mf(19, bold=True)
        self.f_md    = mf(14)
        self.f_sm    = mf(12)
        self.f_xs    = mf(10)
        self.f_title = mf(22, bold=True)

        # Pre-cache server rects
        self._srv_rects: Dict[int, pygame.Rect] = {}
        for r in range(RACK_COUNT):
            for s in range(SRV_PER_RACK):
                sid = r * 5 + s
                self._srv_rects[sid] = server_rect(r, s)

    #  Main draw 

    def draw(self, surf: pygame.Surface, state: SimState):
        surf.fill(C.BG)
        self._draw_panels(surf)
        self._draw_header(surf, state)
        self._draw_left(surf, state)
        self._draw_center(surf, state)
        self._draw_right(surf, state)
        self._draw_footer(surf, state)
        self._draw_packets(surf, state)
        self._draw_decisions(surf, state)
        if state.paused:
            self._draw_pause_overlay(surf)

    # Panel backgrounds 

    def _draw_panels(self, surf):
        # Left panel
        _rr(surf, C.PANEL,  pygame.Rect(0, CONTENT_Y, LEFT_W, CONTENT_H),  r=0)
        # Center panel
        _rr(surf, C.PANEL2, pygame.Rect(CTR_X, CONTENT_Y, CTR_W, CONTENT_H), r=0)
        # Right panel
        _rr(surf, C.PANEL,  pygame.Rect(RIGHT_X, CONTENT_Y, RIGHT_W, CONTENT_H), r=0)
        # Dividers
        pygame.draw.line(surf, C.DIVIDER, (LEFT_W,   CONTENT_Y), (LEFT_W,   H - FOOTER_H))
        pygame.draw.line(surf, C.DIVIDER, (RIGHT_X,  CONTENT_Y), (RIGHT_X,  H - FOOTER_H))
        # Header / footer backgrounds
        pygame.draw.rect(surf, C.HDR_BG, pygame.Rect(0, 0,       W, HEADER_H))
        pygame.draw.rect(surf, C.FTR_BG, pygame.Rect(0, FOOTER_Y, W, FOOTER_H))
        pygame.draw.line(surf, C.BORDER_BRT, (0, HEADER_H),  (W, HEADER_H))
        pygame.draw.line(surf, C.BORDER_BRT, (0, FOOTER_Y),  (W, FOOTER_Y))

    # Header

    def _draw_header(self, surf, state: SimState):
        # Title
        _txt(surf, "🤖  Smart Datacenter AI", self.f_title, C.CYAN,
             18, HEADER_H // 2, anchor='midleft')
        # RL agent stats (center)
        cx = W // 2
        _txt(surf, f"Dueling Double DQN + PER", self.f_md, C.TEXT_LBL,
             cx, 8, anchor='midtop')
        _txt(surf, f"ε = {0.01:.3f}   |   Steps : {state.total_steps:,}   |   Decisions : {state.total_decisions}",
             self.f_sm, C.TEXT_DIM, cx, 30, anchor='midtop')
        _txt(surf, f"▲ {state.upscale_count}  ▼ {state.downscale_count}  ↔ 0",
             self.f_sm, C.TEXT_DIM, cx, 48, anchor='midtop')
        # Scenario badge (right)
        scen_colors = {
            'normal': C.GREEN, 'spike': C.ORANGE, 'overload': C.RED,
            'attack': C.RED,   'recovery': C.TEAL
        }
        sc = state.scenario.upper()
        sc_col = scen_colors.get(state.scenario, C.BLUE)
        badge = pygame.Rect(W - 180, 14, 164, 38)
        _rr(surf, C.lerp(sc_col, C.BG, 0.78), badge, r=8, bw=2, bc=sc_col)
        _txt(surf, f"▶  {sc}", self.f_md, sc_col, badge.centerx, badge.centery, anchor='center')

    # Left panel : Internet → LB → Queue

    def _draw_left(self, surf, state: SimState):
        cx = LEFT_W // 2

        # Internet cloud
        iy = INET_POS[1]
        # Draw stylized cloud icon
        for ox, oy, r in [(cx-18, iy-8, 20), (cx, iy-16, 26), (cx+18, iy-6, 20),
                          (cx-8,  iy+4,  18), (cx+10, iy+6, 18)]:
            pygame.draw.circle(surf, C.lerp(C.BLUE, C.PANEL, 0.6), (ox, oy), r)
        _glow(surf, cx, iy - 8, 40, C.BLUE, alpha=30)
        _txt(surf, "INTERNET", self.f_sm, C.CYAN,  cx, iy + 18, anchor='midtop')
        _txt(surf, f"{state.arrival_rate:.1f} req/s", self.f_xs, C.TEXT_DIM,
             cx, iy + 34, anchor='midtop')

        # Arrow down
        self._draw_arrow_down(surf, cx, iy + 52, 36, C.BORDER_BRT)

        # Load Balancer 
        lbx, lby = LB_POS
        lb_rect = pygame.Rect(cx - 70, lby - 26, 140, 52)
        pulse   = 0.5 + 0.5 * math.sin(state.sim_time * 3.0)
        bc      = C.lerp(C.BLUE, C.CYAN, pulse)
        _rr(surf, C.lerp(C.BLUE, C.PANEL, 0.72), lb_rect, r=8, bw=2, bc=bc)
        _glow(surf, lbx, lby, 30, C.BLUE, alpha=35)
        _txt(surf, "⚖  LOAD BALANCER", self.f_sm, C.CYAN, lbx, lby - 10, anchor='midtop')
        _txt(surf, "Round-Robin + Least-Conn", self.f_xs, C.TEXT_DIM,
             lbx, lby + 8, anchor='midtop')

        # Arrow down
        self._draw_arrow_down(surf, cx, lby + 30, 36, C.BORDER_BRT)

        # Queue
        qx, qy = QUEUE_POS
        q_rect = pygame.Rect(cx - 80, qy - 30, 160, 60)
        fill   = state.queue_len / max(1, state.max_queue)
        qcol   = C.cpu_color(fill)
        _rr(surf, C.lerp(qcol, C.PANEL, 0.75), q_rect, r=8, bw=2, bc=qcol)
        _txt(surf, "📋  QUEUE", self.f_sm, qcol, qx, qy - 26, anchor='midtop')
        # Fill bar
        bar = pygame.Rect(cx - 62, qy, 124, 14)
        _bar(surf, bar, fill, qcol)
        _txt(surf, f"{state.queue_len} / {state.max_queue}", self.f_xs, C.TEXT_LBL,
             qx, qy + 18, anchor='midtop')

        # Arrow to datacenter
        self._draw_arrow_right(surf, cx + 80, qy, 60, C.BORDER_BRT)

        # Stats box 
        sy = qy + 80
        stats = [
            ("Latency",    f"{state.latency:.0f} ms", C.cpu_color(min(1.0, state.latency/300))),
            ("Throughput", f"{state.throughput:.1f}/s", C.TEAL),
            ("SLA Viol.",  f"{state.sla_violations}", C.RED if state.sla_violations > 0 else C.GREEN),
            ("Reward",     f"{state.reward:+.3f}", C.GREEN if state.reward > 0 else C.ORANGE),
        ]
        for i, (lbl, val, col) in enumerate(stats):
            ry = sy + i * 44
            box = pygame.Rect(12, ry, LEFT_W - 24, 38)
            _rr(surf, C.lerp(col, C.PANEL, 0.88), box, r=6, bw=1, bc=C.lerp(col, C.PANEL, 0.60))
            _txt(surf, lbl, self.f_xs, C.TEXT_DIM,  box.x + 8,         box.centery, anchor='midleft')
            _txt(surf, val, self.f_md, col,           box.right - 8,      box.centery, anchor='midright')

    # Center panel : Datacenter racks 

    def _draw_center(self, surf, state: SimState):
        # Panel title
        _txt(surf, "DATACENTER  CLUSTER", self.f_sm, C.TEXT_DIM,
             CTR_X + CTR_W // 2, CONTENT_Y + 2, anchor='midtop')

        for rack_i in range(RACK_COUNT):
            self._draw_rack(surf, state, rack_i)

    def _draw_rack(self, surf, state: SimState, rack_i: int):
        rr = rack_rect(rack_i)
        # Rack background
        _rr(surf, C.lerp(C.PANEL, C.BG, 0.4), rr, r=8, bw=1, bc=C.BORDER)
        # Rack header
        hdr = pygame.Rect(rr.x, rr.y, rr.w, RACK_HDR_H)
        _rr(surf, C.BORDER, hdr, r=8)
        _txt(surf, f"RACK  {rack_i + 1:02d}", self.f_sm, C.TEXT_LBL,
             rr.centerx, rr.y + RACK_HDR_H // 2, anchor='center')
        # Rack LED
        srvs_in_rack = [s for s in state.servers if s.rack == rack_i]
        any_crit = any(s.state == SrvSt.CRITICAL for s in srvs_in_rack)
        any_high = any(s.state == SrvSt.HIGH for s in srvs_in_rack)
        led_col  = C.RED if any_crit else (C.ORANGE if any_high else C.GREEN)
        pygame.draw.circle(surf, led_col, (rr.right - 12, rr.y + RACK_HDR_H // 2), 5)
        _glow(surf, rr.right - 12, rr.y + RACK_HDR_H // 2, 10, led_col, alpha=50)

        # Draw each server in this rack
        for srv in srvs_in_rack:
            self._draw_server(surf, state, srv)

    def _draw_server(self, surf, state: SimState, srv: Server):
        rect = self._srv_rects[srv.sid]

        # Register center for packet routing
        state.server_centers[srv.sid] = rect.center

        # Background + glow
        bg  = srv.bg_color
        bc  = srv.border_color
        bw  = 2 if srv.active else 1
        _rr(surf, bg, rect, r=6, bw=bw, bc=bc)

        # Glow on active servers
        if srv.active and srv.boot_t > 0.5:
            gcol = (C.RED if srv.state == SrvSt.CRITICAL
                    else C.ORANGE if srv.state == SrvSt.HIGH
                    else C.GREEN)
            gintensity = int(40 + 20 * math.sin(srv.pulse))
            _glow(surf, rect.centerx, rect.centery, rect.w // 2 + 4, gcol, alpha=gintensity)

        # Boot progress bar (replaces content while booting)
        if not srv.active and srv.boot_t < 0.01:
            # OFF state
            _txt(surf, f"SRV-{srv.sid + 1:02d}", self.f_xs, C.TEXT_DIM,
                 rect.centerx, rect.centery - 8, anchor='center')
            _txt(surf, "● OFF", self.f_xs, C.TEXT_DIM,
                 rect.centerx, rect.centery + 6, anchor='center')
            return

        if not srv.active and srv.boot_t > 0.01:
            # Shutting down animation
            _bar(surf, pygame.Rect(rect.x + 8, rect.centery - 5, rect.w - 16, 10),
                 srv.boot_t, C.BLUE)
            _txt(surf, "SHUTDOWN…", self.f_xs, C.BLUE,
                 rect.centerx, rect.centery + 10, anchor='center')
            return

        if srv.boot_t < 0.95:
            # Booting animation
            _bar(surf, pygame.Rect(rect.x + 8, rect.centery - 5, rect.w - 16, 10),
                 srv.boot_t, C.CYAN)
            _txt(surf, f"BOOT  {int(srv.boot_t * 100)}%", self.f_xs, C.CYAN,
                 rect.centerx, rect.centery + 10, anchor='center')
            return

        # Active server content
        px = rect.x + 7
        py = rect.y + 5

        # Server name + status badge
        _txt(surf, f"SRV-{srv.sid + 1:02d}", self.f_sm, C.TEXT_BRT, px, py)
        badge_col = bc
        _txt(surf, srv.label, self.f_xs, badge_col, rect.right - 6, py, anchor='topright')

        py += 18
        # CPU bar
        cpu_col = C.cpu_color(srv.cpu)
        _txt(surf, "CPU", self.f_xs, C.TEXT_DIM, px, py)
        _txt(surf, f"{srv.cpu * 100:.0f}%", self.f_xs, cpu_col, rect.right - 6, py, anchor='topright')
        py += 14
        _bar(surf, pygame.Rect(px, py, rect.w - 14, 11), srv.cpu, cpu_col)
        py += 16

        # RAM bar
        ram_col = C.lerp(C.BLUE, C.PURPLE, srv.ram)
        _txt(surf, "RAM", self.f_xs, C.TEXT_DIM, px, py)
        _txt(surf, f"{srv.ram * 100:.0f}%", self.f_xs, ram_col, rect.right - 6, py, anchor='topright')
        py += 14
        _bar(surf, pygame.Rect(px, py, rect.w - 14, 11), srv.ram, ram_col)
        py += 16

        # Jobs
        _txt(surf, f"Jobs: {srv.jobs}", self.f_xs, C.TEXT_DIM, px, py)
        # Activity pulse dots
        for d in range(5):
            dot_on = srv.jobs > d * 6
            dcol   = C.lerp(cpu_col, C.PANEL, 0.0 if dot_on else 0.85)
            px2    = rect.right - 8 - d * 9
            pygame.draw.circle(surf, dcol, (px2, py + 5), 3)

    # Right panel : Dashboard 

    def _draw_right(self, surf, state: SimState):
        rx = RIGHT_X + 10
        rw = RIGHT_W - 20
        cy = CONTENT_Y + 8

        # Title
        _txt(surf, "DASHBOARD", self.f_md, C.TEXT_LBL,
             RIGHT_X + RIGHT_W // 2, cy, anchor='midtop')
        cy += 26

        # Big metrics row
        big = [
            (f"{state.n_active}", f"/ {N_SRV} nodes", C.CYAN),
            (f"{state.avg_cpu * 100:.0f}%", "avg CPU",      C.cpu_color(state.avg_cpu)),
            (f"{state.latency:.0f}", "ms latency", C.cpu_color(min(1, state.latency / 300))),
        ]
        bw = (rw - 8) // 3
        for i, (val, lbl, col) in enumerate(big):
            bx = rx + i * (bw + 4)
            br = pygame.Rect(bx, cy, bw, 56)
            _rr(surf, C.lerp(col, C.PANEL, 0.88), br, r=8, bw=1, bc=C.lerp(col, C.PANEL, 0.55))
            _txt(surf, val, self.f_big, col, br.centerx, br.y + 4, anchor='midtop')
            _txt(surf, lbl, self.f_xs, C.TEXT_DIM, br.centerx, br.bottom - 4, anchor='midbottom')
        cy += 64

        # Charts
        charts = [
            ("CPU Utilization (%)",  state.cpu_hist,    C.GREEN,  0.0, 1.0,
             0.80, C.ORANGE, lambda v: f"{v * 100:.0f}%"),
            ("Latency (ms)",         state.lat_hist,    C.CYAN,   0.0, 350,
             200,  C.RED,    lambda v: f"{v:.0f}ms"),
            ("Queue Length",         state.queue_hist,  C.ORANGE, 0.0, 60,
             45,   C.RED,    lambda v: f"{v:.0f}"),
            ("Reward Signal",        state.reward_hist, C.PURPLE, -1.0, 1.0,
             None, None,     lambda v: f"{v:+.2f}"),
            ("Active Nodes",         state.nodes_hist,  C.TEAL,   0.0, N_SRV,
             None, None,     lambda v: f"{v:.0f}"),
        ]
        chart_h = 85
        for (title, data, col, mn, mx, thr, thr_c, fmt) in charts:
            # Current value
            cur = list(data)[-1] if data else 0.0
            _txt(surf, title, self.f_xs, C.TEXT_DIM, rx, cy)
            _txt(surf, fmt(cur), self.f_sm, col, rx + rw, cy, anchor='topright')
            cy += 16
            _chart(surf, pygame.Rect(rx, cy, rw, chart_h), data, col,
                   min_v=mn, max_v=mx, threshold=thr, thr_col=thr_c)
            cy += chart_h + 10

        # RL Decision history
        _txt(surf, "RL  DECISION  LOG", self.f_sm, C.TEXT_LBL,
             RIGHT_X + RIGHT_W // 2, cy, anchor='midtop')
        cy += 20

        for i, (icon, main, sub, col) in enumerate(list(state.dec_history)[:5]):
            row = pygame.Rect(rx, cy, rw, 30)
            _rr(surf, C.lerp(col, C.PANEL, 0.88), row, r=5, bw=1, bc=C.lerp(col, C.PANEL, 0.55))
            a = max(80, 255 - i * 40)
            _txt(surf, f"{icon}  {main}", self.f_xs, (*col[:3],), rx + 6, row.centery, anchor='midleft')
            cy += 34

        # SLA violations
        cy += 4
        sla_col = C.RED if state.sla_violations > 0 else C.GREEN
        sla_box = pygame.Rect(rx, cy, rw, 36)
        _rr(surf, C.lerp(sla_col, C.PANEL, 0.88), sla_box, r=6, bw=2, bc=sla_col)
        _txt(surf, "SLA Violations", self.f_xs, C.TEXT_DIM, rx + 8, sla_box.centery, anchor='midleft')
        _txt(surf, str(state.sla_violations), self.f_lg, sla_col, sla_box.right - 8,
             sla_box.centery, anchor='midright')

    # Footer : scenario buttons 

    def _draw_footer(self, surf, state: SimState):
        scenarios = [
            ('1', 'Normal',   'normal',   C.GREEN),
            ('2', 'Spike',    'spike',    C.ORANGE),
            ('3', 'Overload', 'overload', C.RED),
            ('4', 'Attack',   'attack',   C.PINK),
            ('5', 'Recovery', 'recovery', C.TEAL),
        ]
        bw = 150
        start_x = 20
        by      = FOOTER_Y + 7
        bh      = FOOTER_H - 14

        for i, (key, label, name, col) in enumerate(scenarios):
            bx     = start_x + i * (bw + 8)
            active = (state.scenario == name)
            bg     = C.lerp(col, C.PANEL, 0.5 if active else 0.88)
            bc     = col if active else C.lerp(col, C.PANEL, 0.55)
            br     = pygame.Rect(bx, by, bw, bh)
            _rr(surf, bg, br, r=6, bw=2 if active else 1, bc=bc)
            if active:
                _glow(surf, br.centerx, br.centery, bw // 2, col, alpha=30)
            _txt(surf, f"[{key}]  {label}", self.f_sm, col if active else C.TEXT_DIM,
                 br.centerx, br.centery, anchor='center')

        # Controls (right side)
        ctrl_x = W - 340
        speed_col = C.YELLOW
        _txt(surf, f"SPACE: Pause  |  +/- Speed ×{state.speed:.1f}  |  R: Reset  |  ESC: Quit",
             self.f_xs, C.TEXT_DIM, ctrl_x, FOOTER_Y + FOOTER_H // 2, anchor='midleft')

    # Packets 

    def _draw_packets(self, surf, state: SimState):
        for p in state.packets:
            if p.alpha <= 0:
                continue
            a = int(min(255, p.alpha))
            # Trail
            for j, (tx, ty) in enumerate(p.trail):
                ta = int(a * (j + 1) / len(p.trail) * 0.5)
                if ta > 0:
                    ts = max(1, p.size - 1)
                    s  = pygame.Surface((ts * 2 + 2, ts * 2 + 2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*p.color, ta), (ts + 1, ts + 1), ts)
                    surf.blit(s, (int(tx) - ts - 1, int(ty) - ts - 1))
            # Packet dot
            ps = pygame.Surface((p.size * 2 + 4, p.size * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ps, (*p.color, a), (p.size + 2, p.size + 2), p.size)
            surf.blit(ps, (int(p.x) - p.size - 2, int(p.y) - p.size - 2))

    # RL Decision overlays
    def _draw_decisions(self, surf, state: SimState):
        for i, d in enumerate(state.decisions[:3]):
            # Fade in/out
            fi  = min(1.0, d.age * 4.0)
            fo  = 1.0 if d.age < d.life - 1.0 else max(0.0, d.life - d.age)
            alpha = int(255 * min(fi, fo))
            if alpha <= 0:
                continue

            dw, dh = 340, 68
            dx = CTR_X + (CTR_W - dw) // 2 + i * 20 - 20
            dy = CONTENT_Y + 18 + i * 78

            card = pygame.Surface((dw, dh), pygame.SRCALPHA)
            pygame.draw.rect(card, (*d.color, int(50 * fo)), (0, 0, dw, dh), border_radius=10)
            pygame.draw.rect(card, (*d.color, int(200 * fo)), (0, 0, dw, dh), 2, border_radius=10)

            # Icon
            ic_s = self.f_lg.render(d.icon, True, (*d.color, alpha))
            card.blit(ic_s, (10, 12))
            # Main text
            mt_s = self.f_md.render(d.main_text, True, (*C.TEXT_BRT, alpha))
            card.blit(mt_s, (46, 8))
            # Sub text
            st_s = self.f_xs.render(d.sub_text, True, (*C.TEXT_DIM, alpha))
            card.blit(st_s, (46, 32))
            # Progress bar at bottom
            prog = 1.0 - d.age / d.life
            bar_rect = pygame.Rect(10, dh - 8, int((dw - 20) * prog), 4)
            pygame.draw.rect(card, (*d.color, int(180 * fo)), bar_rect, border_radius=2)

            surf.blit(card, (dx, dy))

    # Helpers

    def _draw_arrow_down(self, surf, cx, y, length, col):
        pygame.draw.line(surf, col, (cx, y), (cx, y + length - 8), 2)
        pygame.draw.polygon(surf, col, [
            (cx, y + length), (cx - 6, y + length - 10), (cx + 6, y + length - 10)
        ])

    def _draw_arrow_right(self, surf, x, cy, length, col):
        pygame.draw.line(surf, col, (x, cy), (x + length - 8, cy), 2)
        pygame.draw.polygon(surf, col, [
            (x + length, cy), (x + length - 10, cy - 6), (x + length - 10, cy + 6)
        ])

    def _draw_pause_overlay(self, surf):
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(ov, (0, 0, 0, 120), (0, 0, W, H))
        surf.blit(ov, (0, 0))
        _txt(surf, "⏸  PAUSE  —  appuyez SPACE pour reprendre",
             self.f_big, C.YELLOW, W // 2, H // 2, anchor='center')


#  SECTION 8 — MAIN APPLICATION

class DatacenterApp:
    CKPT = os.path.join(ROOT, "checkpoints", "best.pt")

    def __init__(self):
        pygame.init()
        self.surf = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Smart Datacenter AI — Dueling Double DQN Autoscaling")
        self.clock    = pygame.time.Clock()
        self.state    = SimState()
        self.renderer = Renderer()
        self.bridge   = RLBridge(ckpt_path=self.CKPT)
        self._running = True

    def run(self):
        print("\n" + "═" * 60)
        print("  Smart Autonomous Datacenter Simulator")
        print("  Controls: 1-5 Scenarios | SPACE Pause | +/- Speed | ESC Quit")
        print("═" * 60 + "\n")

        while self._running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)   # cap at 50ms

            self._handle_events()
            self._rl_step()
            self.state.update(dt)
            self.renderer.draw(self.surf, self.state)
            pygame.display.flip()

        pygame.quit()

    # RL agent decision step

    def _rl_step(self):
        if self.state.paused:
            return
        if self.state._rl_t < self.state.rl_interval:
            return

        self.state._rl_t = 0.0
        self.state._thinking = 0.4

        action = self.bridge.get_action(self.state)

        # --- Vérifier si l'action est réellement applicable ---
        n = self.state.n_active
        if action == 3 and n <= 1:   action = 0  # impossible de descendre sous 1
        if action == 4 and n <= 1:   action = 0
        if action == 4 and n == 2:   action = 3  # max réduction possible = 1
        if action == 1 and n >= N_SRV: action = 0  # déjà au max
        if action == 2 and n >= N_SRV: action = 0

        icon, main, sub, color = ACTION_META[action]

        if action == 1:
            self.state.activate(1)
            self.state.upscale_count += 1
        elif action == 2:
            self.state.activate(2)
            self.state.upscale_count += 1
        elif action == 3:
            self.state.deactivate(1)
            self.state.downscale_count += 1
        elif action == 4:
            self.state.deactivate(2)
            self.state.downscale_count += 1
        elif action == 5:
            pass   # migrate: visual only

        self.state.add_decision(icon, main, sub, color)

        # SLA alert if needed
        if self.state.latency > 200 or self.state.queue_len > 40:
            self.state.add_decision("⚠", "SLA ALERT",
                                    f"Latency {self.state.latency:.0f}ms — queue={self.state.queue_len}",
                                    C.DEC_ALERT)

    # Events 
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                k = event.key

                if k == pygame.K_ESCAPE:
                    self._running = False

                elif k == pygame.K_SPACE:
                    self.state.paused = not self.state.paused

                elif k == pygame.K_r:
                    self.state.reset()

                elif k == pygame.K_EQUALS or k == pygame.K_PLUS or k == pygame.K_KP_PLUS:
                    self.state.speed = min(8.0, self.state.speed * 2.0)

                elif k == pygame.K_MINUS or k == pygame.K_KP_MINUS:
                    self.state.speed = max(0.25, self.state.speed / 2.0)

                elif k == pygame.K_1:
                    self.state.set_scenario('normal')
                elif k == pygame.K_2:
                    self.state.set_scenario('spike')
                elif k == pygame.K_3:
                    self.state.set_scenario('overload')
                elif k == pygame.K_4:
                    self.state.set_scenario('attack')
                elif k == pygame.K_5:
                    self.state.set_scenario('recovery')


#  ENTRY POINT

def main():
    DatacenterApp().run()


if __name__ == "__main__":
    main()

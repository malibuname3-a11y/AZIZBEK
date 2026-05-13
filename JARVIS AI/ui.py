from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import cv2
import numpy as np

from PyQt6.QtCore import (
    QMimeData, QPointF, QRectF, QSize, Qt,
    QTimer, QUrl, pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush, QColor, QDragEnterEvent, QDropEvent, QFont,
    QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QImage,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QTextEdit,
    QVBoxLayout, QWidget,
)


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR   = _base_dir()
CONFIG_DIR = BASE_DIR / "config"
API_FILE   = CONFIG_DIR / "api_keys.json"

_DEFAULT_W, _DEFAULT_H = 980, 700
_MIN_W,     _MIN_H     = 820, 580
_LEFT_W  = 148
_RIGHT_W = 340
_OS = platform.system()


# ---------------------------------------------------------------------------
# Ranglar
# ---------------------------------------------------------------------------
class C:
    BG       = "#00060a"
    PANEL    = "#010d14"
    PANEL2   = "#010f18"
    BORDER   = "#0d3347"
    BORDER_B = "#1a5c7a"
    BORDER_A = "#0f4060"
    PRI      = "#00d4ff"
    PRI_DIM  = "#007a99"
    PRI_GHO  = "#001f2e"
    ACC      = "#ff6b00"
    ACC2     = "#ffcc00"
    GREEN    = "#00ff88"
    RED      = "#ff3355"
    MUTED_C  = "#ff3366"
    TEXT     = "#8ffcff"
    TEXT_DIM = "#3a8a9a"
    TEXT_MED = "#5ab8cc"
    WHITE    = "#d8f8ff"
    DARK     = "#000d14"
    BAR_BG   = "#011520"


def qcol(h: str, a: int = 255) -> QColor:
    c = QColor(h); c.setAlpha(a); return c


# ---------------------------------------------------------------------------
# Tizim ko'rsatkichlari
# ---------------------------------------------------------------------------
class _SysMetrics:
    def __init__(self):
        self.cpu = self.mem = self.net = 0.0
        self.gpu = self.tmp = -1.0
        self._lock = threading.Lock()
        self._last_net = psutil.net_io_counters()
        self._last_net_t = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try: self._update()
            except Exception: pass
            time.sleep(1.5)

    def _update(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        nc  = psutil.net_io_counters()
        now = time.time()
        dt  = now - self._last_net_t
        net = ((nc.bytes_sent - self._last_net.bytes_sent) +
               (nc.bytes_recv - self._last_net.bytes_recv)) / max(dt, 0.001) / (1024*1024)
        self._last_net = nc; self._last_net_t = now
        gpu = self._get_gpu(); tmp = self._get_temp()
        with self._lock:
            self.cpu, self.mem, self.net, self.gpu, self.tmp = cpu, mem, net, gpu, tmp

    def _get_gpu(self) -> float:
        try:
            r = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                vals = [float(v.strip()) for v in r.stdout.strip().split("\n") if v.strip()]
                if vals: return sum(vals)/len(vals)
        except Exception: pass
        return -1.0

    def _get_temp(self) -> float:
        try:
            temps = psutil.sensors_temperatures()
            for name in ["coretemp","k10temp","cpu_thermal","acpitz","zenpower"]:
                if name in temps and temps[name]: return temps[name][0].current
            for entries in temps.values():
                if entries: return entries[0].current
        except Exception: pass
        return -1.0

    def snapshot(self) -> dict:
        with self._lock:
            return {"cpu":self.cpu,"mem":self.mem,"net":self.net,"gpu":self.gpu,"tmp":self.tmp}


_metrics = _SysMetrics()


# ---------------------------------------------------------------------------
# Yuz landmark aniqlash (OpenCV Haar cascade)
# ---------------------------------------------------------------------------
class FaceLandmarks:
    """Rasm faylidan yuz, ko'z va lab koordinatalarini aniqlaydi."""

    def __init__(self, image_path: str):
        self.valid = False
        self.img_w = self.img_h = 0
        self.face  = (0, 0, 0, 0)   # x, y, w, h  (piksel)
        self.eye_l = (0, 0, 0)      # cx, cy, radius
        self.eye_r = (0, 0, 0)
        self.mouth = (0, 0, 0, 0)   # cx, cy, half_w, half_h
        self._detect(image_path)

    def _detect(self, path: str):
        try:
            from PIL import Image
            pil    = Image.open(path).convert("RGB")
            img_np = np.array(pil)
            self.img_h, self.img_w = img_np.shape[:2]
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            fc = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            ec = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml")

            faces = fc.detectMultiScale(gray, 1.05, 5)
            if len(faces) == 0:
                return

            fx, fy, fw, fh = faces[0]
            self.face = (fx, fy, fw, fh)

            # Ko'zlar — faqat yuzning yuqori 55% qismida izlaymiz
            roi      = gray[fy : fy + int(fh * 0.55), fx : fx + fw]
            eyes     = ec.detectMultiScale(roi, 1.1, 5, minSize=(25, 25))
            eyes_s   = sorted(eyes, key=lambda e: e[0])[:2]

            if len(eyes_s) >= 2:
                ex0,ey0,ew0,eh0 = eyes_s[0]
                ex1,ey1,ew1,eh1 = eyes_s[1]
                self.eye_l = (fx+ex0+ew0//2, fy+ey0+eh0//2, ew0//2)
                self.eye_r = (fx+ex1+ew1//2, fy+ey1+eh1//2, ew1//2)
            elif len(eyes_s) == 1:
                ex0,ey0,ew0,_ = eyes_s[0]
                mid = fx + fw//2
                cy0 = fy + ey0 + ew0//2
                r0  = ew0//2
                if (fx+ex0+ew0//2) < mid:
                    self.eye_l = (fx+ex0+ew0//2, cy0, r0)
                    self.eye_r = (fx+fw-ex0-ew0//2, cy0, r0)
                else:
                    self.eye_r = (fx+ex0+ew0//2, cy0, r0)
                    self.eye_l = (fx+fw-ex0-ew0//2, cy0, r0)
            else:
                # Taxminiy hisoblash
                self.eye_l = (fx+fw//3,     fy+int(fh*0.35), fw//10)
                self.eye_r = (fx+2*fw//3,   fy+int(fh*0.35), fw//10)

            # Lab — yuzning ~77% balandligida
            self.mouth = (fx+fw//2, fy+int(fh*0.77), int(fw*0.22), int(fh*0.07))
            self.valid = True

        except Exception as e:
            print(f"[FaceLandmarks] {e}")


# ---------------------------------------------------------------------------
# HudCanvas  —  rasm + animatsiyali overlay
# ---------------------------------------------------------------------------
class HudCanvas(QWidget):
    def __init__(self, face_path: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.muted    = False
        self.speaking = False
        self.state    = "INITIALISING"

        # HUD animatsiya holati
        self._tick      = 0
        self._halo      = 55.0
        self._tgt_halo  = 55.0
        self._scale     = 1.0
        self._tgt_scale = 1.0
        self._last_t    = time.time()
        self._scan      = 0.0
        self._scan2     = 180.0
        self._rings     = [0.0, 120.0, 240.0]
        self._pulses: list[float] = [0.0, 50.0, 100.0]
        self._blink_sym = True
        self._blink_cnt = 0
        self._particles: list[list[float]] = []

        # Ko'z blink & lab animatsiyasi
        self._eye_blink  = 0.0   # 0=ochiq  1=yumiq
        self._eye_phase  = 0.0
        self._mouth_open = 0.0   # 0..1
        self._mouth_tgt  = 0.0

        # Rasm va landmark
        self._face_px:  QPixmap | None      = None
        self._face_lm:  FaceLandmarks | None = None
        self._load_face(face_path)

        self._tmr = QTimer(self)
        self._tmr.timeout.connect(self._step)
        self._tmr.start(16)   # ~60 fps

    # ------------------------------------------------------------------
    def _load_face(self, path: str):
        if not path or not Path(path).is_file():
            return
        try:
            from PIL import Image, ImageDraw
            import io
            img = Image.open(path).convert("RGBA")
            sz  = min(img.size)
            img = img.resize((sz, sz), Image.LANCZOS)
            mk  = Image.new("L", (sz, sz), 0)
            ImageDraw.Draw(mk).ellipse((2, 2, sz-2, sz-2), fill=255)
            img.putalpha(mk)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            px = QPixmap(); px.loadFromData(buf.getvalue())
            self._face_px = px
        except Exception as e:
            print(f"[load_face] {e}")
            self._face_px = None

        lm = FaceLandmarks(path)
        self._face_lm = lm if lm.valid else None

    # ------------------------------------------------------------------
    def _step(self):
        self._tick += 1
        now = time.time()

        if now - self._last_t > (0.12 if self.speaking else 0.5):
            if self.speaking:
                self._tgt_scale = random.uniform(1.06, 1.14)
                self._tgt_halo  = random.uniform(145, 190)
            elif self.muted:
                self._tgt_scale = random.uniform(0.998, 1.002)
                self._tgt_halo  = random.uniform(15, 28)
            else:
                self._tgt_scale = random.uniform(1.001, 1.008)
                self._tgt_halo  = random.uniform(48, 68)
            self._last_t = now

        sp = 0.38 if self.speaking else 0.15
        self._scale += (self._tgt_scale - self._scale) * sp
        self._halo  += (self._tgt_halo  - self._halo)  * sp

        speeds = [1.3, -0.9, 2.0] if self.speaking else [0.55, -0.35, 0.9]
        for i, spd in enumerate(speeds):
            self._rings[i] = (self._rings[i] + spd) % 360
        self._scan  = (self._scan  + (3.0 if self.speaking else 1.3)) % 360
        self._scan2 = (self._scan2 + (-2.0 if self.speaking else -0.75)) % 360

        fw  = min(self.width(), self.height())
        lim = fw * 0.74
        spd = 4.2 if self.speaking else 2.0
        self._pulses = [r + spd for r in self._pulses if r + spd < lim]
        if len(self._pulses) < 3 and random.random() < (0.07 if self.speaking else 0.025):
            self._pulses.append(0.0)

        if self.speaking and random.random() < 0.28:
            cx, cy = self.width()/2, self.height()/2
            ang = random.uniform(0, 2*math.pi)
            r_s = fw * 0.28
            self._particles.append([
                cx + math.cos(ang)*r_s, cy + math.sin(ang)*r_s,
                math.cos(ang)*random.uniform(0.9, 2.4),
                math.sin(ang)*random.uniform(0.9, 2.4) - 0.4, 1.0,
            ])
        self._particles = [
            [p[0]+p[2], p[1]+p[3], p[2]*0.97, p[3]*0.97, p[4]-0.028]
            for p in self._particles if p[4] > 0
        ]

        self._blink_cnt += 1
        if self._blink_cnt >= 38:
            self._blink_sym = not self._blink_sym
            self._blink_cnt = 0

        # Ko'z blink sikli (har 2-4 soniyada)
        self._eye_phase = (self._eye_phase + 0.04) % (2 * math.pi)
        blink_cycle = math.sin(self._eye_phase * 0.28)
        if blink_cycle > 0.95:
            self._eye_blink = min(self._eye_blink + 0.30, 1.0)
        else:
            self._eye_blink = max(self._eye_blink - 0.14, 0.0)

        # Lab animatsiyasi
        if self.speaking:
            self._mouth_tgt = random.uniform(0.3, 1.0)
        elif self.muted:
            self._mouth_tgt = 0.0
        else:
            self._mouth_tgt = 0.04 + 0.03 * math.sin(self._tick * 0.08)
        self._mouth_open += (self._mouth_tgt - self._mouth_open) * 0.35

        self.update()

    # ------------------------------------------------------------------
    # paintEvent
    # ------------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), qcol(C.BG))

        W, H   = self.width(), self.height()
        cx, cy = W/2, H/2
        fw     = min(W, H)

        # Fon nuqtalari
        p.setPen(QPen(qcol(C.PRI_GHO), 1))
        for x in range(0, W, 48):
            for y in range(0, H, 48):
                p.drawPoint(x, y)

        r_face = fw * 0.31

        # Halo halqalari
        for i in range(10):
            r   = r_face * (1.8 - i*0.08)
            frc = 1.0 - i/10
            a   = max(0, min(255, int(self._halo * 0.085 * frc)))
            col = qcol(C.MUTED_C if self.muted else C.PRI, a)
            p.setPen(QPen(col, 1.5)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(cx-r, cy-r, r*2, r*2))

        # Puls halqalari
        for pr in self._pulses:
            a   = max(0, int(230 * (1.0 - pr/(fw*0.74))))
            col = qcol(C.MUTED_C if self.muted else C.PRI, a)
            p.setPen(QPen(col, 1.5)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(cx-pr, cy-pr, pr*2, pr*2))

        # Aylanuvchi yoy halqalari
        for idx, (r_frac, w_r, arc_l, gap) in enumerate(
            [(0.48,3,115,78),(0.40,2,78,55),(0.32,1,56,40)]
        ):
            ring_r = fw * r_frac
            base   = self._rings[idx]
            a_val  = max(0, min(255, int(self._halo*(1.0-idx*0.18))))
            col    = qcol(C.MUTED_C if self.muted else C.PRI, a_val)
            p.setPen(QPen(col, w_r)); p.setBrush(Qt.BrushStyle.NoBrush)
            angle = base
            rect  = QRectF(cx-ring_r, cy-ring_r, ring_r*2, ring_r*2)
            while angle < base+360:
                p.drawArc(rect, int(angle*16), int(arc_l*16))
                angle += arc_l + gap

        # Scanner yoylari
        sr  = fw * 0.50
        sa  = min(255, int(self._halo*1.5))
        ext = 75 if self.speaking else 44
        p.setPen(QPen(qcol(C.MUTED_C if self.muted else C.PRI, sa), 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        srect = QRectF(cx-sr, cy-sr, sr*2, sr*2)
        p.drawArc(srect, int(self._scan*16),  int(ext*16))
        p.setPen(QPen(qcol(C.ACC, sa//2), 1.5))
        p.drawArc(srect, int(self._scan2*16), int(ext*16))

        # Graduslash belgilari
        t_out, t_in = fw*0.497, fw*0.474
        p.setPen(QPen(qcol(C.PRI, 140), 1))
        for deg in range(0, 360, 10):
            rad = math.radians(deg)
            inn = t_in if deg%30==0 else t_in+6
            p.drawLine(
                QPointF(cx + t_out*math.cos(rad), cy - t_out*math.sin(rad)),
                QPointF(cx + inn *math.cos(rad), cy - inn *math.sin(rad)),
            )

        # Crosshair
        ch_r, gap_h = fw*0.51, fw*0.16
        p.setPen(QPen(qcol(C.PRI, int(self._halo*0.5)), 1))
        p.drawLine(QPointF(cx-ch_r, cy), QPointF(cx-gap_h, cy))
        p.drawLine(QPointF(cx+gap_h, cy), QPointF(cx+ch_r, cy))
        p.drawLine(QPointF(cx, cy-ch_r), QPointF(cx, cy-gap_h))
        p.drawLine(QPointF(cx, cy+gap_h), QPointF(cx, cy+ch_r))

        # Burchak braketlari
        bl = 24
        bc = qcol(C.PRI, 210)
        hl, hr = cx - fw//2, cx + fw//2
        ht, hb = cy - fw//2, cy + fw//2
        p.setPen(QPen(bc, 2))
        for bx, by, dx, dy in [(hl,ht,1,1),(hr,ht,-1,1),(hl,hb,1,-1),(hr,hb,-1,-1)]:
            p.drawLine(QPointF(bx,by), QPointF(bx+dx*bl, by))
            p.drawLine(QPointF(bx,by), QPointF(bx, by+dy*bl))

        # --- ASOSIY YUZ ---
        if self._face_px:
            self._draw_face_with_overlay(p, cx, cy, fw)
        else:
            self._draw_robot_face(p, cx, cy, fw)

        # Zarralar
        for pt in self._particles:
            a = max(0, min(255, int(pt[4]*255)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(qcol(C.PRI, a)))
            p.drawEllipse(QPointF(pt[0], pt[1]), 2.5, 2.5)

        # Status matni
        sy = cy + fw*0.40
        if self.muted:
            txt, col = "⊘  MUTED",        qcol(C.MUTED_C)
        elif self.speaking:
            txt, col = "●  SPEAKING",      qcol(C.ACC)
        elif self.state == "THINKING":
            sym = "◈" if self._blink_sym else "◇"
            txt, col = f"{sym}  THINKING",    qcol(C.ACC2)
        elif self.state == "PROCESSING":
            sym = "▷" if self._blink_sym else "▶"
            txt, col = f"{sym}  PROCESSING",  qcol(C.ACC2)
        elif self.state == "LISTENING":
            sym = "●" if self._blink_sym else "○"
            txt, col = f"{sym}  LISTENING",   qcol(C.GREEN)
        else:
            sym = "●" if self._blink_sym else "○"
            txt, col = f"{sym}  {self.state}", qcol(C.PRI)

        p.setPen(QPen(col, 1))
        p.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        p.drawText(QRectF(0, sy, W, 26), Qt.AlignmentFlag.AlignCenter, txt)

        # Tovush to'lqini
        wy = sy + 30
        N, bw = 36, 8
        wx0  = (W - N*bw) / 2
        for i in range(N):
            if self.muted:
                hgt, cl = 2, qcol(C.MUTED_C)
            elif self.speaking:
                hgt = random.randint(3, 20)
                cl  = qcol(C.PRI) if hgt > 12 else qcol(C.PRI_DIM)
            else:
                hgt = int(3 + 2*math.sin(self._tick*0.09 + i*0.6))
                cl  = qcol(C.BORDER_B)
            p.fillRect(QRectF(wx0+i*bw, wy+20-hgt, bw-1, hgt), cl)

    # ------------------------------------------------------------------
    # Rasm + animatsiyali ko'z/lab overlay
    # ------------------------------------------------------------------
    def _draw_face_with_overlay(self, p: QPainter, cx: float, cy: float, fw: float):
        # 1) Aylana kesimli rasm
        fsz    = int(fw * 0.62 * self._scale)
        scaled = self._face_px.scaled(
            fsz, fsz,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        img_x = int(cx - fsz/2)
        img_y = int(cy - fsz/2)
        p.drawPixmap(img_x, img_y, scaled)

        # 2) HUD yozuvi — rasm tepasida
        p.setFont(QFont("Courier New", 7, QFont.Weight.Bold))
        p.setPen(QPen(qcol(C.PRI, 160), 1))
        p.drawText(QRectF(img_x, img_y + fsz*0.04, fsz, 14),
                   Qt.AlignmentFlag.AlignCenter, "◈ FACE-ID  LOCKED")
        p.setFont(QFont("Courier New", 6))
        p.setPen(QPen(qcol(C.TEXT_DIM, 120), 1))
        p.drawText(QRectF(img_x, img_y + fsz*0.90, fsz, 12),
                   Qt.AlignmentFlag.AlignCenter, "BIO-AUTH: VERIFIED")

        # 3) Landmark overlay
        if self._face_lm is None:
            return

        lm      = self._face_lm
        orig_sz = min(lm.img_w, lm.img_h)
        sf      = fsz / orig_sz          # miqyos koeffitsienti
        crop_ox = (lm.img_w - orig_sz) / 2
        crop_oy = (lm.img_h - orig_sz) / 2

        def to_screen(px: float, py: float):
            return (img_x + (px - crop_ox) * sf,
                    img_y + (py - crop_oy) * sf)

        face_col = "#ff3366" if self.muted else "#00d4ff"

        # --- Ko'z overlay ---
        for eye_data in [lm.eye_l, lm.eye_r]:
            ex, ey_px, er = eye_data
            sx, sy = to_screen(ex, ey_px)
            sr     = er * sf

            open_h = sr * 0.85 * (1.0 - self._eye_blink)

            # Tashqi diqqat ellipsi (scanning ring)
            pulse_a = int(55 + 35*math.sin(self._eye_phase) + self._mouth_open*50)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(qcol(face_col, min(255, pulse_a)), 1.2))
            p.drawEllipse(QRectF(sx - sr*1.3, sy - sr*0.75,
                                 sr*2.6,       sr*1.5))

            if open_h > 1.5:
                # Iris parlaqligi
                ig_a = int(28 + self._mouth_open*38)
                p.setBrush(QBrush(qcol(face_col, ig_a)))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(sx-sr*0.45, sy-open_h*0.45,
                                     sr*0.90,      open_h*0.90))

                # Ko'z yumayotganda qovoq qorayadi
                if self._eye_blink > 0.08:
                    lid_a = int(self._eye_blink * 210)
                    p.setBrush(QBrush(qcol("#0a0500", lid_a)))
                    p.setPen(Qt.PenStyle.NoPen)
                    # Yuqori qovoq yarimmoon
                    path = QPainterPath()
                    path.moveTo(sx - sr, sy)
                    path.cubicTo(sx-sr, sy - open_h*1.3,
                                 sx+sr, sy - open_h*1.3,
                                 sx+sr, sy)
                    path.closeSubpath()
                    p.drawPath(path)

                # Yiltiroq nuqta
                p.setBrush(QBrush(qcol("#ffffff", 130)))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(sx - sr*0.22, sy - open_h*0.40,
                                     sr*0.16,       open_h*0.13))

                # Ko'z halqasi (glow)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(qcol(face_col, min(255, pulse_a+30)), 1.5))
                p.drawEllipse(QRectF(sx-sr*0.50, sy-open_h*0.50,
                                     sr*1.00,      open_h*1.00))

            else:
                # Ko'z to'liq yumiq — ingichka chiziq
                p.setPen(QPen(qcol(face_col, 170), 1.8))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(QPointF(sx - sr*0.80, sy),
                           QPointF(sx + sr*0.80, sy))

        # --- Lab overlay ---
        mx, my_px, mhw, mhh = lm.mouth
        smx, smy = to_screen(mx, my_px)
        sm_hw    = mhw * sf
        sm_hh    = mhh * sf
        open_amt = self._mouth_open * sm_hh * 3.8

        # Lab kontur
        mouth_a = int(140 + self._mouth_open * 100)
        p.setPen(QPen(qcol(face_col, mouth_a), 1.5))
        p.setBrush(QBrush(qcol("#050d1a", int(self._mouth_open * 130))))
        p.drawRoundedRect(
            QRectF(smx - sm_hw, smy - sm_hh/2,
                   sm_hw*2,     sm_hh + open_amt), 5, 5
        )

        # Lab ichidagi to'lqin barlar
        bars  = 10
        bar_w = (sm_hw*2 - 6) / bars
        for i in range(bars):
            if self._mouth_open > 0.04:
                bh = 1.2 + abs(math.sin(
                    self._tick*0.14 + i*0.65
                )) * (sm_hh*0.7 + self._mouth_open * sm_hh*2.2)
            else:
                bh = 0.8 + math.sin(i*0.9) * 0.6
            bx   = smx - sm_hw + 3 + i*bar_w
            midy = smy + (sm_hh + open_amt) / 2
            p.fillRect(
                QRectF(bx, midy - bh/2, max(bar_w-1.5, 1), bh),
                qcol(face_col, int(90 + self._mouth_open*150)),
            )

    # ------------------------------------------------------------------
    # Fallback: rasm bo'lmasa — sodda robot yuzi
    # ------------------------------------------------------------------
    def _draw_robot_face(self, p: QPainter, cx: float, cy: float, fw: float):
        face_col  = "#ff3366" if self.muted else "#00d4ff"
        mouth_col = "#ff6688" if self.muted else "#00d4ff"
        speak_lvl = self._mouth_open
        scale     = self._scale

        head_w = fw*0.44*scale; head_h = fw*0.52*scale
        hx = cx - head_w/2;    hy = cy - head_h/2 - fw*0.04

        p.setBrush(QBrush(qcol("#010f18")))
        p.setPen(QPen(qcol(face_col, 160), 1.5))
        p.drawRoundedRect(QRectF(hx, hy, head_w, head_h), 14, 14)

        for side in (-1, 1):
            ew = fw*0.055*scale; eh = fw*0.18*scale
            ex = cx + side*(head_w/2 + ew*0.4) - ew/2
            ey = cy - eh/2 - fw*0.04
            p.setBrush(QBrush(qcol("#010d14")))
            p.setPen(QPen(qcol(face_col, 100), 1))
            p.drawRoundedRect(QRectF(ex, ey, ew, eh), 3, 3)
            p.setPen(QPen(qcol(face_col, 180), 1.5))
            for i in range(3):
                ly = ey + eh*0.18 + i*eh*0.24
                p.drawLine(QPointF(ex+2, ly), QPointF(ex+ew-2, ly))

        eye_y = cy - fw*0.10*scale; eye_dist = fw*0.125*scale
        eye_w = fw*0.135*scale;     eye_h    = fw*0.088*scale
        open_h = eye_h * (1.0 - self._eye_blink)

        for side in (-1, 1):
            ex = cx + side*eye_dist
            p.setBrush(QBrush(qcol("#050d1a")))
            p.setPen(QPen(qcol(face_col, 70), 0.5))
            p.drawRoundedRect(QRectF(ex-eye_w/2, eye_y-eye_h/2, eye_w, eye_h), 5, 5)
            if open_h > 1.5:
                p.setBrush(QBrush(qcol(face_col, int(55+speak_lvl*75))))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(ex-eye_w*0.4, eye_y-open_h*0.4, eye_w*0.8, open_h*0.8))
                p.setBrush(QBrush(qcol("#050d1a")))
                p.drawEllipse(QRectF(ex-eye_w*0.15, eye_y-open_h*0.15, eye_w*0.3, open_h*0.3))
                p.setBrush(QBrush(qcol("#ffffff", 170)))
                p.drawEllipse(QRectF(ex-eye_w*0.13, eye_y-open_h*0.27, eye_w*0.09, open_h*0.11))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(qcol(face_col, 160), 1.5))
            p.drawRoundedRect(QRectF(ex-eye_w/2, eye_y-eye_h/2, eye_w, eye_h), 5, 5)

        my = cy + fw*0.095*scale; mw = fw*0.295*scale; mh = fw*0.058*scale
        oa = speak_lvl * fw*0.065*scale
        p.setBrush(QBrush(qcol("#050d1a")))
        p.setPen(QPen(qcol(mouth_col, 190), 1.5))
        p.drawRoundedRect(QRectF(cx-mw/2, my-mh/2, mw, mh+oa), 6, 6)
        bars = 11; bw2 = (mw-10)/bars
        for i in range(bars):
            bh = (fw*0.01*scale + abs(math.sin(self._tick*0.14+i*0.65))*(fw*0.022*scale+speak_lvl*fw*0.022*scale)
                  if speak_lvl>0.05 else fw*0.007*scale+math.sin(i*0.85)*fw*0.003*scale)
            bx2  = cx - mw/2 + 5 + i*bw2
            midy = my + (mh+oa)/2
            p.fillRect(QRectF(bx2, midy-bh/2, max(bw2-1.5,1), bh),
                       qcol(mouth_col, int(130+speak_lvl*110)))


# ---------------------------------------------------------------------------
# MetricBar
# ---------------------------------------------------------------------------
class MetricBar(QWidget):
    def __init__(self, label: str, color: str = C.PRI, parent=None):
        super().__init__(parent)
        self._label=label; self._color=color; self._value=0.0; self._text="--"
        self.setFixedHeight(38); self.setMinimumWidth(80)

    def set_value(self, pct: float, text: str):
        self._value=max(0.0,min(100.0,pct)); self._text=text; self.update()

    def paintEvent(self, _):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W,H=self.width(),self.height()
        p.setBrush(QBrush(qcol(C.PANEL2))); p.setPen(QPen(qcol(C.BORDER_A),1))
        p.drawRoundedRect(QRectF(1,1,W-2,H-2),4,4)
        bar_h=4; bar_y=H-bar_h-5; bar_w=W-12; bar_x=6; fill_w=int(bar_w*self._value/100)
        p.setBrush(QBrush(qcol(C.BAR_BG))); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(bar_x,bar_y,bar_w,bar_h),2,2)
        bar_col=(qcol(C.RED) if self._value>85 else qcol(C.ACC) if self._value>65 else qcol(self._color))
        if fill_w>0:
            p.setBrush(QBrush(bar_col)); p.drawRoundedRect(QRectF(bar_x,bar_y,fill_w,bar_h),2,2)
        p.setFont(QFont("Courier New",7,QFont.Weight.Bold)); p.setPen(QPen(qcol(C.TEXT_DIM),1))
        p.drawText(QRectF(8,5,50,14),Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,self._label)
        p.setFont(QFont("Courier New",9,QFont.Weight.Bold))
        p.setPen(QPen(bar_col if self._text!="--" else qcol(C.TEXT_DIM),1))
        p.drawText(QRectF(0,4,W-6,16),Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter,self._text)


# ---------------------------------------------------------------------------
# LogWidget
# ---------------------------------------------------------------------------
class LogWidget(QTextEdit):
    _sig = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True); self.setFont(QFont("Courier New",9))
        self.setStyleSheet(f"""
            QTextEdit{{background:{C.PANEL};color:{C.TEXT};border:1px solid {C.BORDER};
                       border-radius:4px;padding:6px;selection-background-color:{C.PRI_GHO};}}
            QScrollBar:vertical{{background:{C.BG};width:8px;border:none;}}
            QScrollBar::handle:vertical{{background:{C.BORDER_B};border-radius:4px;min-height:20px;}}
        """)
        self._queue:list[str]=[]; self._typing=False; self._text=""; self._pos=0; self._tag="sys"
        self._tmr=QTimer(self); self._tmr.timeout.connect(self._step)
        self._sig.connect(self._enqueue)

    def append_log(self,text:str): self._sig.emit(text)

    def _enqueue(self,text:str):
        self._queue.append(text)
        if not self._typing: self._next()

    def _next(self):
        if not self._queue: self._typing=False; return
        self._typing=True; self._text=self._queue.pop(0); self._pos=0
        tl=self._text.lower()
        self._tag=("you" if tl.startswith("you:") else "ai" if tl.startswith("jarvis:")
                   else "file" if tl.startswith("file:") else "err" if "err" in tl else "sys")
        self._tmr.start(6)

    def _step(self):
        if self._pos<len(self._text):
            ch=self._text[self._pos]; cur=self.textCursor(); fmt=cur.charFormat()
            col={"you":qcol(C.WHITE),"ai":qcol(C.PRI),"err":qcol(C.RED),
                 "file":qcol(C.GREEN),"sys":qcol(C.ACC2)}.get(self._tag,qcol(C.TEXT))
            fmt.setForeground(QBrush(col)); cur.movePosition(cur.MoveOperation.End)
            cur.insertText(ch,fmt); self.setTextCursor(cur); self.ensureCursorVisible(); self._pos+=1
        else:
            self._tmr.stop(); cur=self.textCursor(); cur.movePosition(cur.MoveOperation.End)
            cur.insertText("\n"); self.setTextCursor(cur); self.ensureCursorVisible()
            QTimer.singleShot(20,self._next)


# ---------------------------------------------------------------------------
# FileDropZone
# ---------------------------------------------------------------------------
_FILE_ICONS={
    "image":("🖼","#00d4ff"),"video":("🎬","#ff6b00"),"audio":("🎵","#cc44ff"),
    "pdf":("📄","#ff4444"),"word":("📝","#4488ff"),"excel":("📊","#44bb44"),
    "code":("💻","#ffcc00"),"archive":("📦","#ff8844"),"pptx":("📊","#ff6622"),
    "text":("📃","#aaaaaa"),"data":("🔧","#88ddff"),"unknown":("📎","#888888"),
}
_EXT_TO_CAT={
    **dict.fromkeys(["jpg","jpeg","png","gif","webp","bmp","tiff","svg","ico"],"image"),
    **dict.fromkeys(["mp4","avi","mov","mkv","wmv","flv","webm","m4v"],"video"),
    **dict.fromkeys(["mp3","wav","ogg","m4a","aac","flac","wma","opus"],"audio"),
    **dict.fromkeys(["pdf"],"pdf"),**dict.fromkeys(["doc","docx"],"word"),
    **dict.fromkeys(["xls","xlsx","ods"],"excel"),**dict.fromkeys(["ppt","pptx"],"pptx"),
    **dict.fromkeys(["py","js","ts","jsx","tsx","html","css","java","c","cpp",
                     "cs","go","rs","rb","php","swift","kt","sh","sql","lua"],"code"),
    **dict.fromkeys(["zip","rar","tar","gz","7z","bz2","xz"],"archive"),
    **dict.fromkeys(["txt","md","rst","log"],"text"),
    **dict.fromkeys(["csv","tsv","json","xml"],"data"),
}

def _file_category(path:Path)->str: return _EXT_TO_CAT.get(path.suffix.lower().lstrip("."),"unknown")
def _fmt_size(size:int)->str:
    if size<1024: return f"{size} B"
    elif size<1024**2: return f"{size/1024:.1f} KB"
    elif size<1024**3: return f"{size/1024**2:.1f} MB"
    return f"{size/1024**3:.1f} GB"


class FileDropZone(QWidget):
    file_selected=pyqtSignal(str)
    def __init__(self,parent=None):
        super().__init__(parent); self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor); self.setFixedHeight(100)
        self._current_file:str|None=None; self._hovering=False; self._drag_over=False; self._dash_offset=0.0
        self._anim_tmr=QTimer(self); self._anim_tmr.timeout.connect(self._animate); self._anim_tmr.start(40)
        lay=QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        self._canvas=_DropCanvas(self); lay.addWidget(self._canvas)
    def _animate(self): self._dash_offset=(self._dash_offset+0.8)%20; self._canvas.update()
    def dragEnterEvent(self,e):
        if e.mimeData().hasUrls(): e.acceptProposedAction(); self._drag_over=True; self._canvas.update()
    def dragLeaveEvent(self,e): self._drag_over=False; self._canvas.update()
    def dropEvent(self,e):
        self._drag_over=False
        if e.mimeData().hasUrls():
            path=e.mimeData().urls()[0].toLocalFile()
            if Path(path).is_file(): self._set_file(path)
        self._canvas.update()
    def mousePressEvent(self,e):
        if e.button()==Qt.MouseButton.LeftButton: self._browse()
    def enterEvent(self,e): self._hovering=True; self._canvas.update()
    def leaveEvent(self,e): self._hovering=False; self._canvas.update()
    def current_file(self)->str|None: return self._current_file
    def clear_file(self): self._current_file=None; self._canvas.update()
    def _browse(self):
        path,_=QFileDialog.getOpenFileName(self,"Select file",str(Path.home()),"All Files (*.*)")
        if path: self._set_file(path)
    def _set_file(self,path:str):
        self._current_file=path; self._canvas.update(); self.file_selected.emit(path)


class _DropCanvas(QWidget):
    def __init__(self,zone): super().__init__(zone); self._z=zone
    def paintEvent(self,_):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        z=self._z; W,H=self.width(),self.height(); pad=6; rect=QRectF(pad,pad,W-pad*2,H-pad*2)
        p.setBrush(QBrush(qcol("#001a24" if z._drag_over else ("#001218" if z._hovering else C.PANEL))))
        p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(rect,6,6)
        bc=(qcol(C.GREEN,200) if z._current_file else qcol(C.PRI,230) if z._drag_over
            else qcol(C.BORDER_B,200) if z._hovering else qcol(C.BORDER,160))
        pen=QPen(bc,1.5,Qt.PenStyle.DashLine); pen.setDashOffset(z._dash_offset)
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRoundedRect(rect,6,6)
        if z._current_file: self._paint_file(p,W,H)
        elif z._drag_over:  self._paint_drop(p,W,H)
        else:               self._paint_idle(p,W,H,z._hovering)
    def _paint_idle(self,p,W,H,hover):
        cx,cy=W/2,H/2; col=qcol(C.PRI_DIM if not hover else C.PRI)
        p.setPen(QPen(col,2)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(cx,cy-14),QPointF(cx,cy+4))
        p.drawLine(QPointF(cx-8,cy-6),QPointF(cx,cy-14))
        p.drawLine(QPointF(cx+8,cy-6),QPointF(cx,cy-14))
        p.drawLine(QPointF(cx-14,cy+4),QPointF(cx+14,cy+4))
        p.setFont(QFont("Courier New",8)); p.setPen(QPen(qcol(C.PRI_DIM if not hover else C.TEXT),1))
        p.drawText(QRectF(0,cy+8,W,16),Qt.AlignmentFlag.AlignCenter,"Drop file here  or  Click to Browse")
    def _paint_drop(self,p,W,H):
        cx,cy=W/2,H/2; p.setFont(QFont("Courier New",20)); p.setPen(QPen(qcol(C.PRI),1))
        p.drawText(QRectF(0,cy-24,W,32),Qt.AlignmentFlag.AlignCenter,"⬇")
    def _paint_file(self,p,W,H):
        path=Path(self._z._current_file); cat=_file_category(path)
        icon,ic=_FILE_ICONS.get(cat,_FILE_ICONS["unknown"]); size=_fmt_size(path.stat().st_size)
        p.setFont(QFont("Segoe UI Emoji",22) if _OS=="Windows" else QFont("Arial",22))
        p.setPen(QPen(qcol(ic),1))
        p.drawText(QRectF(10,0,60,H),Qt.AlignmentFlag.AlignCenter,icon)
        p.setFont(QFont("Courier New",8,QFont.Weight.Bold)); p.setPen(QPen(qcol(C.WHITE),1))
        name=path.name[:31]+"..." if len(path.name)>34 else path.name
        p.drawText(QRectF(76,H*0.18,W-114,16),Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,name)
        p.setFont(QFont("Courier New",7)); p.setPen(QPen(qcol(C.TEXT_DIM),1))
        p.drawText(QRectF(76,H*0.18+18,W-114,14),Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,
                   f"{path.suffix.upper().lstrip('.')}  ·  {size}")
        p.setFont(QFont("Courier New",9,QFont.Weight.Bold)); p.setPen(QPen(qcol(C.RED,180),1))
        p.drawText(QRectF(W-34,0,28,H),Qt.AlignmentFlag.AlignCenter,"✕")
    def mousePressEvent(self,e):
        z=self._z
        if z._current_file and e.pos().x()>self.width()-34: z.clear_file()
        else: z.mousePressEvent(e)


# ---------------------------------------------------------------------------
# SetupOverlay
# ---------------------------------------------------------------------------
class SetupOverlay(QWidget):
    done=pyqtSignal(str,str)
    def __init__(self,parent=None):
        super().__init__(parent); self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.setStyleSheet(f"SetupOverlay{{background:rgba(0,6,10,245);border:1px solid {C.BORDER_B};border-radius:6px;}}")
        detected={"darwin":"mac","windows":"windows"}.get(_OS.lower(),"linux"); self._sel_os=detected
        lay=QVBoxLayout(self); lay.setContentsMargins(30,22,30,22); lay.setSpacing(8)
        def _lbl(txt,fs=9,bold=False,color=C.PRI,align=Qt.AlignmentFlag.AlignCenter):
            w=QLabel(txt); w.setAlignment(align)
            w.setFont(QFont("Courier New",fs,QFont.Weight.Bold if bold else QFont.Weight.Normal))
            w.setStyleSheet(f"color:{color};background:transparent;"); return w
        lay.addWidget(_lbl("◈  INITIALISATION REQUIRED",13,True))
        lay.addWidget(_lbl("Configure J.A.R.V.I.S. before first boot.",9,color=C.PRI_DIM))
        lay.addSpacing(6)
        sep=QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet(f"color:{C.BORDER};"); lay.addWidget(sep)
        lay.addSpacing(4)
        lay.addWidget(_lbl("GEMINI API KEY",8,color=C.TEXT_DIM,align=Qt.AlignmentFlag.AlignLeft))
        self._key_input=QLineEdit(); self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("AIza…"); self._key_input.setFont(QFont("Courier New",10))
        self._key_input.setFixedHeight(32)
        self._key_input.setStyleSheet(f"QLineEdit{{background:#000d12;color:{C.TEXT};border:1px solid {C.BORDER};border-radius:3px;padding:4px 8px;}}QLineEdit:focus{{border:1px solid {C.PRI};}}")
        lay.addWidget(self._key_input); lay.addSpacing(12)
        sep2=QFrame(); sep2.setFrameShape(QFrame.Shape.HLine); sep2.setStyleSheet(f"color:{C.BORDER};"); lay.addWidget(sep2)
        lay.addSpacing(4)
        lay.addWidget(_lbl("OPERATING SYSTEM",8,color=C.TEXT_DIM,align=Qt.AlignmentFlag.AlignLeft))
        det_name={"windows":"Windows","mac":"macOS","linux":"Linux"}[detected]
        lay.addWidget(_lbl(f"Auto-detected: {det_name}",8,color=C.ACC2,align=Qt.AlignmentFlag.AlignLeft))
        os_row=QHBoxLayout(); os_row.setSpacing(6); self._os_btns:dict[str,QPushButton]={}
        for key,label in [("windows","⊞  Windows"),("mac","  macOS"),("linux","🐧  Linux")]:
            btn=QPushButton(label); btn.setFont(QFont("Courier New",9,QFont.Weight.Bold))
            btn.setFixedHeight(32); btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _,k=key:self._sel(k)); os_row.addWidget(btn); self._os_btns[key]=btn
        lay.addLayout(os_row); self._sel(detected); lay.addSpacing(12)
        init_btn=QPushButton("▸  INITIALISE SYSTEMS")
        init_btn.setFont(QFont("Courier New",10,QFont.Weight.Bold)); init_btn.setFixedHeight(36)
        init_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        init_btn.setStyleSheet(f"QPushButton{{background:transparent;color:{C.PRI};border:1px solid {C.PRI_DIM};border-radius:3px;}}QPushButton:hover{{background:{C.PRI_GHO};border:1px solid {C.PRI};}}")
        init_btn.clicked.connect(self._submit); lay.addWidget(init_btn)
    def _sel(self,key:str):
        self._sel_os=key
        pal={"windows":(C.PRI,"#001a22"),"mac":(C.ACC2,"#1a1400"),"linux":(C.GREEN,"#001a0d")}
        for k,btn in self._os_btns.items():
            if k==key:
                fg,bg=pal[k]
                btn.setStyleSheet(f"QPushButton{{background:{fg};color:{bg};border:none;border-radius:3px;font-weight:bold;}}")
            else:
                btn.setStyleSheet(f"QPushButton{{background:#000d12;color:{C.TEXT_DIM};border:1px solid {C.BORDER};border-radius:3px;}}QPushButton:hover{{color:{C.TEXT};border:1px solid {C.BORDER_B};}}")
    def _submit(self):
        key=self._key_input.text().strip()
        if not key:
            self._key_input.setStyleSheet(self._key_input.styleSheet()+f" QLineEdit{{border:1px solid {C.RED};}}")
            return
        self.done.emit(key,self._sel_os)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    _log_sig  =pyqtSignal(str)
    _state_sig=pyqtSignal(str)

    def __init__(self,face_path:str):
        super().__init__()
        self.setWindowTitle("J.A.R.V.I.S — MARK XXXIX")
        self.setMinimumSize(_MIN_W,_MIN_H); self.resize(_DEFAULT_W,_DEFAULT_H)
        screen=QApplication.primaryScreen().availableGeometry()
        self.move((screen.width()-_DEFAULT_W)//2,(screen.height()-_DEFAULT_H)//2)
        self.on_text_command=None; self._muted=False; self._current_file:str|None=None

        central=QWidget(); central.setStyleSheet(f"background:{C.BG};")
        self.setCentralWidget(central)
        root=QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._build_header())
        body=QHBoxLayout(); body.setContentsMargins(0,0,0,0); body.setSpacing(0)
        body.addWidget(self._build_left_panel(),stretch=0)
        self.hud=HudCanvas(face_path)
        self.hud.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        body.addWidget(self.hud,stretch=5)
        body.addWidget(self._build_right_panel(),stretch=0)
        root.addLayout(body,stretch=1); root.addWidget(self._build_footer())

        self._clock_tmr=QTimer(self); self._clock_tmr.timeout.connect(self._tick_clock)
        self._clock_tmr.start(1000); self._tick_clock()
        self._metric_tmr=QTimer(self); self._metric_tmr.timeout.connect(self._update_metrics)
        self._metric_tmr.start(2000); self._update_metrics()

        self._log_sig.connect(self._log.append_log)
        self._state_sig.connect(self._apply_state)
        self._overlay:SetupOverlay|None=None
        self._ready=self._check_config()
        if not self._ready: self._show_setup()

        QShortcut(QKeySequence("F4"),self).activated.connect(self._toggle_mute)
        QShortcut(QKeySequence("F11"),self).activated.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self._overlay and self._overlay.isVisible():
            cw=self.centralWidget()
            self._overlay.setGeometry((cw.width()-460)//2,(cw.height()-390)//2,460,390)

    def _update_metrics(self):
        snap=_metrics.snapshot()
        self._bar_cpu.set_value(snap["cpu"],f"{snap['cpu']:.0f}%")
        self._bar_mem.set_value(snap["mem"],f"{snap['mem']:.0f}%")
        net=snap["net"]
        self._bar_net.set_value(min(100,net*10),f"{net*1024:.0f}KB/s" if net<1 else f"{net:.1f}MB/s")
        gpu=snap["gpu"]
        self._bar_gpu.set_value(gpu if gpu>=0 else 0,f"{gpu:.0f}%" if gpu>=0 else "N/A")
        tmp=snap["tmp"]
        self._bar_tmp.set_value(min(100,(tmp/100)*100) if tmp>=0 else 0,f"{tmp:.0f}°C" if tmp>=0 else "N/A")
        try:
            el=time.time()-psutil.boot_time(); h=int(el//3600); m=int((el%3600)//60)
            self._uptime_lbl.setText(f"UP  {h:02d}:{m:02d}")
        except: self._uptime_lbl.setText("UP  --:--")
        try: self._proc_lbl.setText(f"PROC  {len(psutil.pids())}")
        except: self._proc_lbl.setText("PROC  --")

    def _build_header(self)->QWidget:
        w=QWidget(); w.setFixedHeight(54)
        w.setStyleSheet(f"background:{C.DARK};border-bottom:1px solid {C.BORDER_B};")
        lay=QHBoxLayout(w); lay.setContentsMargins(16,0,16,0)
        def _badge(txt,color=C.TEXT_MED):
            l=QLabel(txt); l.setFont(QFont("Courier New",8))
            l.setStyleSheet(f"color:{color};background:transparent;"); return l
        lay.addWidget(_badge("MARK XXXIX",C.PRI_DIM)); lay.addStretch()
        mid=QVBoxLayout(); mid.setSpacing(1)
        t=QLabel("ELDOR AGENT"); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setFont(QFont("Courier New",17,QFont.Weight.Bold))
        t.setStyleSheet(f"color:{C.PRI};background:transparent;"); mid.addWidget(t)
        s=QLabel("Just A Rather Very Intelligent System"); s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setFont(QFont("Courier New",7)); s.setStyleSheet(f"color:{C.PRI_DIM};background:transparent;")
        mid.addWidget(s); lay.addLayout(mid); lay.addStretch()
        rc=QVBoxLayout(); rc.setSpacing(2)
        self._clock_lbl=QLabel("00:00:00"); self._clock_lbl.setFont(QFont("Courier New",14,QFont.Weight.Bold))
        self._clock_lbl.setStyleSheet(f"color:{C.PRI};background:transparent;")
        self._clock_lbl.setAlignment(Qt.AlignmentFlag.AlignRight); rc.addWidget(self._clock_lbl)
        self._date_lbl=QLabel(""); self._date_lbl.setFont(QFont("Courier New",7))
        self._date_lbl.setStyleSheet(f"color:{C.TEXT_DIM};background:transparent;")
        self._date_lbl.setAlignment(Qt.AlignmentFlag.AlignRight); rc.addWidget(self._date_lbl)
        lay.addLayout(rc); return w

    def _tick_clock(self):
        self._clock_lbl.setText(time.strftime("%H:%M:%S"))
        self._date_lbl.setText(time.strftime("%a %d %b %Y"))

    def _build_left_panel(self)->QWidget:
        w=QWidget(); w.setFixedWidth(_LEFT_W)
        w.setStyleSheet(f"background:{C.DARK};border-right:1px solid {C.BORDER};")
        lay=QVBoxLayout(w); lay.setContentsMargins(8,10,8,10); lay.setSpacing(6)
        hdr=QLabel("◈ SYS MONITOR"); hdr.setFont(QFont("Courier New",7,QFont.Weight.Bold))
        hdr.setStyleSheet(f"color:{C.PRI};background:transparent;border-bottom:1px solid {C.BORDER};padding-bottom:4px;")
        lay.addWidget(hdr); lay.addSpacing(2)
        self._bar_cpu=MetricBar("CPU",C.PRI); self._bar_mem=MetricBar("MEM",C.ACC2)
        self._bar_net=MetricBar("NET",C.GREEN); self._bar_gpu=MetricBar("GPU",C.ACC)
        self._bar_tmp=MetricBar("TMP","#ff6688")
        for bar in [self._bar_cpu,self._bar_mem,self._bar_net,self._bar_gpu,self._bar_tmp]:
            lay.addWidget(bar)
        lay.addSpacing(4)
        ip=QWidget(); ip.setStyleSheet(f"background:{C.PANEL2};border:1px solid {C.BORDER};border-radius:4px;")
        ipl=QVBoxLayout(ip); ipl.setContentsMargins(6,5,6,5); ipl.setSpacing(3)
        self._uptime_lbl=QLabel("UP  --:--"); self._uptime_lbl.setFont(QFont("Courier New",8,QFont.Weight.Bold))
        self._uptime_lbl.setStyleSheet(f"color:{C.GREEN};background:transparent;border:none;"); ipl.addWidget(self._uptime_lbl)
        self._proc_lbl=QLabel("PROC  --"); self._proc_lbl.setFont(QFont("Courier New",8))
        self._proc_lbl.setStyleSheet(f"color:{C.TEXT_MED};background:transparent;border:none;"); ipl.addWidget(self._proc_lbl)
        os_name={"Windows":"WIN","Darwin":"macOS","Linux":"LINUX"}.get(_OS,_OS.upper())
        ol=QLabel(f"OS  {os_name}"); ol.setFont(QFont("Courier New",8))
        ol.setStyleSheet(f"color:{C.ACC2};background:transparent;border:none;"); ipl.addWidget(ol)
        lay.addWidget(ip); lay.addStretch()
        for txt,col in [("AI CORE\nACTIVE",C.GREEN),("SEC\nCLEARED",C.PRI),("PROTOCOL\nXXXVIII",C.TEXT_DIM)]:
            lb=QLabel(txt); lb.setFont(QFont("Courier New",7,QFont.Weight.Bold))
            lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb.setStyleSheet(f"color:{col};background:{C.PANEL2};border:1px solid {C.BORDER_A};border-radius:3px;padding:4px;")
            lay.addWidget(lb)
        return w

    def _build_right_panel(self)->QWidget:
        w=QWidget(); w.setFixedWidth(_RIGHT_W)
        w.setStyleSheet(f"background:{C.DARK};border-left:1px solid {C.BORDER};")
        lay=QVBoxLayout(w); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)
        def _sec(txt):
            l=QLabel(f"▸ {txt}"); l.setFont(QFont("Courier New",7,QFont.Weight.Bold))
            l.setStyleSheet(f"color:{C.TEXT_MED};background:transparent;"); return l
        lay.addWidget(_sec("ACTIVITY LOG"))
        self._log=LogWidget(); lay.addWidget(self._log,stretch=1)
        sep=QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet(f"color:{C.BORDER};margin:2px 0;"); lay.addWidget(sep)
        lay.addWidget(_sec("FILE UPLOAD"))
        self._drop_zone=FileDropZone(); self._drop_zone.file_selected.connect(self._on_file_selected)
        lay.addWidget(self._drop_zone)
        self._file_hint=QLabel("No file loaded — drop or click above to upload")
        self._file_hint.setFont(QFont("Courier New",7))
        self._file_hint.setStyleSheet(f"color:{C.TEXT_MED};background:transparent;"); self._file_hint.setWordWrap(True)
        lay.addWidget(self._file_hint)
        sep2=QFrame(); sep2.setFrameShape(QFrame.Shape.HLine); sep2.setStyleSheet(f"color:{C.BORDER};margin:2px 0;"); lay.addWidget(sep2)
        lay.addWidget(_sec("COMMAND INPUT")); lay.addLayout(self._build_input_row())
        self._mute_btn=QPushButton("🎙  MICROPHONE ACTIVE"); self._mute_btn.setFixedHeight(30)
        self._mute_btn.setFont(QFont("Courier New",8,QFont.Weight.Bold))
        self._mute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mute_btn.clicked.connect(self._toggle_mute); self._style_mute_btn(); lay.addWidget(self._mute_btn)
        fs=QPushButton("⛶  FULLSCREEN  [F11]"); fs.setFixedHeight(26); fs.setFont(QFont("Courier New",7))
        fs.setCursor(Qt.CursorShape.PointingHandCursor)
        fs.setStyleSheet(f"QPushButton{{background:transparent;color:{C.TEXT_MED};border:1px solid {C.BORDER};border-radius:3px;}}QPushButton:hover{{color:{C.PRI};border:1px solid {C.BORDER_B};}}")
        fs.clicked.connect(self._toggle_fullscreen); lay.addWidget(fs)
        return w

    def _build_input_row(self)->QHBoxLayout:
        row=QHBoxLayout(); row.setSpacing(5)
        self._input=QLineEdit(); self._input.setPlaceholderText("Type a command or question…")
        self._input.setFont(QFont("Courier New",9)); self._input.setFixedHeight(30)
        self._input.setStyleSheet(f"QLineEdit{{background:#000d14;color:{C.WHITE};border:1px solid {C.BORDER};border-radius:3px;padding:3px 7px;}}QLineEdit:focus{{border:1px solid {C.PRI};}}")
        self._input.returnPressed.connect(self._send); row.addWidget(self._input)
        send=QPushButton("▸"); send.setFixedSize(30,30); send.setFont(QFont("Courier New",11,QFont.Weight.Bold))
        send.setCursor(Qt.CursorShape.PointingHandCursor)
        send.setStyleSheet(f"QPushButton{{background:{C.PANEL};color:{C.PRI};border:1px solid {C.PRI_DIM};border-radius:3px;}}QPushButton:hover{{background:{C.PRI_GHO};border:1px solid {C.PRI};}}")
        send.clicked.connect(self._send); row.addWidget(send); return row

    def _build_footer(self)->QWidget:
        w=QWidget(); w.setFixedHeight(22)
        w.setStyleSheet(f"background:{C.DARK};border-top:1px solid {C.BORDER};")
        lay=QHBoxLayout(w); lay.setContentsMargins(14,0,14,0)
        def _fl(txt,color=C.TEXT_MED):
            l=QLabel(txt); l.setFont(QFont("Courier New",7))
            l.setStyleSheet(f"color:{color};background:transparent;"); return l
        lay.addWidget(_fl("[F4] Mute  ·  [F11] Fullscreen")); lay.addStretch()
        lay.addWidget(_fl("FatihMakes Industries  ·  MARK XXXIX  ·  CLASSIFIED")); lay.addStretch()
        lay.addWidget(_fl("© FATIHMAKES",C.PRI_DIM)); return w

    def _on_file_selected(self,path:str):
        self._current_file=path; p2=Path(path); cat=_file_category(p2)
        icon,_=_FILE_ICONS.get(cat,_FILE_ICONS["unknown"]); size=_fmt_size(p2.stat().st_size)
        self._file_hint.setText(f"{icon}  {p2.name}  ·  {size}  ·  Tell JARVIS what to do with it")
        self._log.append_log(f"FILE: {p2.name} ({size}) loaded")
        if self.on_text_command:
            msg=(f"[FILE_UPLOADED] path={path} | name={p2.name} | type={p2.suffix.lstrip('.')} | size={size} | "
                 f"Briefly tell the user you can see the file '{p2.name}' ({size}) has been uploaded.")
            threading.Thread(target=self.on_text_command,args=(msg,),daemon=True).start()

    def _toggle_mute(self):
        self._muted=not self._muted; self.hud.muted=self._muted; self._style_mute_btn()
        if self._muted: self._apply_state("MUTED"); self._log.append_log("SYS: Microphone muted.")
        else:           self._apply_state("LISTENING"); self._log.append_log("SYS: Microphone active.")

    def _style_mute_btn(self):
        if self._muted:
            self._mute_btn.setText("🔇  MICROPHONE MUTED")
            self._mute_btn.setStyleSheet(f"QPushButton{{background:#140006;color:{C.MUTED_C};border:1px solid {C.MUTED_C};border-radius:3px;}}")
        else:
            self._mute_btn.setText("🎙  MICROPHONE ACTIVE")
            self._mute_btn.setStyleSheet(f"QPushButton{{background:#00140a;color:{C.GREEN};border:1px solid {C.GREEN};border-radius:3px;}}QPushButton:hover{{background:#001f10;}}")

    def _send(self):
        txt=self._input.text().strip()
        if not txt: return
        self._input.clear(); self._log.append_log(f"You: {txt}")
        if self.on_text_command:
            threading.Thread(target=self.on_text_command,args=(txt,),daemon=True).start()

    def _apply_state(self,state:str):
        self.hud.state=state; self.hud.speaking=(state=="SPEAKING")

    def _check_config(self)->bool:
        if not API_FILE.exists(): return False
        try:
            d=json.loads(API_FILE.read_text(encoding="utf-8"))
            return bool(d.get("gemini_api_key")) and bool(d.get("os_system"))
        except: return False

    def _show_setup(self):
        ov=SetupOverlay(self.centralWidget()); cw=self.centralWidget()
        ov.setGeometry((cw.width()-460)//2,(cw.height()-390)//2,460,390)
        ov.done.connect(self._on_setup_done); ov.show(); self._overlay=ov

    def _on_setup_done(self,key:str,os_name:str):
        os.makedirs(CONFIG_DIR,exist_ok=True)
        API_FILE.write_text(json.dumps({"gemini_api_key":key,"os_system":os_name},indent=4),encoding="utf-8")
        self._ready=True
        if self._overlay: self._overlay.hide(); self._overlay=None
        self._apply_state("LISTENING")
        self._log.append_log(f"SYS: Initialised. OS={os_name.upper()}. JARVIS online.")


# ---------------------------------------------------------------------------
# JarvisUI — public API
# ---------------------------------------------------------------------------
class _RootShim:
    def __init__(self,app): self._app=app
    def mainloop(self): self._app.exec()
    def protocol(self,*_): pass


class JarvisUI:
    def __init__(self, face_path: str, size=None):
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._app.setStyle("Fusion")
        self._win = MainWindow(face_path)
        self._win.show()
        self.root = _RootShim(self._app)

    @property
    def muted(self)->bool: return self._win._muted
    @muted.setter
    def muted(self,v:bool):
        if v!=self._win._muted: self._win._toggle_mute()

    @property
    def current_file(self)->str|None: return self._win._drop_zone.current_file()

    @property
    def on_text_command(self): return self._win.on_text_command
    @on_text_command.setter
    def on_text_command(self,cb): self._win.on_text_command=cb

    def set_state(self,state:str): self._win._state_sig.emit(state)
    def write_log(self,text:str):  self._win._log_sig.emit(text)

    def wait_for_api_key(self):
        while not self._win._ready: time.sleep(0.1)

    def start_speaking(self): self.set_state("SPEAKING")
    def stop_speaking(self):
        if not self.muted: self.set_state("LISTENING")
from __future__ import annotations

import math
import os
import random
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

WIDTH, HEIGHT = 1000, 700
FPS = 30
CAMERA_INDEX = 0
PREVIEW_W, PREVIEW_H = 240, 180

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

SKY_TOP = (210, 239, 255)
SKY_BOTTOM = (255, 228, 239)
WHITE = (255, 255, 255)
CREAM = (255, 249, 238)
PINK = (244, 146, 177)
DARK_PINK = (206, 91, 134)
RED = (233, 78, 95)
DARK_RED = (174, 49, 67)
GREEN = (92, 177, 104)
BROWN = (135, 84, 55)
DARK_BROWN = (91, 55, 41)
GOLD = (255, 206, 84)
INK = (62, 58, 72)
BLACK = (35, 37, 47)
LAVENDER = (194, 177, 240)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    for name in ("comicsansms", "segoeui", "arialrounded", "arial"):
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


def draw_text(surface, text, font, color, *, center=None, topleft=None):
    image = font.render(text, True, color)
    rect = image.get_rect(center=center) if center else image.get_rect(topleft=topleft)
    surface.blit(image, rect)
    return rect


def download_model() -> None:
    if MODEL_PATH.exists():
        return
    print("Preuzima se MediaPipe hand model")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Model nije mogao biti preuzet. Provjeri internet pa pokreni ponovo."
        ) from exc
    print("Model je preuzet.")


def make_background() -> pygame.Surface:
    bg = pygame.Surface((WIDTH, HEIGHT))
    for y in range(HEIGHT):
        t = y / HEIGHT
        color = tuple(int(SKY_TOP[i] * (1 - t) + SKY_BOTTOM[i] * t) for i in range(3))
        pygame.draw.line(bg, color, (0, y), (WIDTH, y))

    pygame.draw.ellipse(bg, (181, 224, 171), (-180, HEIGHT - 155, 700, 250))
    pygame.draw.ellipse(bg, (162, 214, 159), (350, HEIGHT - 140, 850, 250))
    pygame.draw.rect(bg, (173, 220, 164), (0, HEIGHT - 70, WIDTH, 70))

    for cx, cy, s in ((120, 95, 1.0), (460, 135, 0.8), (790, 85, 1.1)):
        pygame.draw.circle(bg, WHITE, (cx, cy), int(29 * s))
        pygame.draw.circle(bg, WHITE, (cx + int(35 * s), cy - 8), int(37 * s))
        pygame.draw.circle(bg, WHITE, (cx + int(73 * s), cy), int(28 * s))
        pygame.draw.ellipse(bg, WHITE, (cx - 8, cy, int(112 * s), int(40 * s)))
    return bg


def draw_sparkle(surface, x, y, size=6, color=GOLD):
    pygame.draw.line(surface, color, (x - size, y), (x + size, y), 2)
    pygame.draw.line(surface, color, (x, y - size), (x, y + size), 2)


def draw_heart(surface, x, y, size, color=RED):
    r = size // 4
    pygame.draw.circle(surface, color, (x - r, y - r // 2), r)
    pygame.draw.circle(surface, color, (x + r, y - r // 2), r)
    pygame.draw.polygon(surface, color, [(x - size // 2, y), (x + size // 2, y), (x, y + size // 2)])


def draw_apple(surface, x, y, r):

    scale = 4
    R = max(1, int(r * scale))
    canvas_size = 4 * R

    apple = pygame.Surface(
        (canvas_size, canvas_size),
        pygame.SRCALPHA
    )

    cx = 2 * R
    cy = int(2.18 * R)

    def pt(px, py):
        return int(px), int(py)

    def ellipse(rect, color, width=0):
        pygame.draw.ellipse(
            apple,
            color,
            tuple(int(value) for value in rect),
            width
        )

    stem_start = pt(
        cx - 0.04 * R,
        cy - 0.88 * R
    )

    stem_end = pt(
        cx + 0.12 * R,
        cy - 1.58 * R
    )

    stem_width = max(3, int(0.18 * R))

    pygame.draw.line(
        apple,
        (92, 48, 25),
        stem_start,
        stem_end,
        stem_width
    )

    pygame.draw.line(
        apple,
        (151, 82, 38),
        pt(cx, cy - 0.90 * R),
        pt(cx + 0.10 * R, cy - 1.56 * R),
        max(2, int(0.09 * R))
    )

    pygame.draw.circle(
        apple,
        (181, 103, 47),
        stem_end,
        max(2, int(0.09 * R))
    )

    leaf_outline = [
        pt(cx + 0.05 * R, cy - 1.03 * R),
        pt(cx + 0.38 * R, cy - 1.55 * R),
        pt(cx + 1.12 * R, cy - 1.52 * R),
        pt(cx + 0.92 * R, cy - 0.98 * R),
        pt(cx + 0.30 * R, cy - 0.82 * R)
    ]

    pygame.draw.polygon(
        apple,
        (47, 113, 44),
        leaf_outline
    )

    leaf_inner = [
        pt(cx + 0.13 * R, cy - 1.04 * R),
        pt(cx + 0.43 * R, cy - 1.46 * R),
        pt(cx + 1.00 * R, cy - 1.45 * R),
        pt(cx + 0.84 * R, cy - 1.06 * R),
        pt(cx + 0.34 * R, cy - 0.89 * R)
    ]

    pygame.draw.polygon(
        apple,
        (106, 190, 73),
        leaf_inner
    )

    pygame.draw.line(
        apple,
        (57, 126, 48),
        pt(cx + 0.20 * R, cy - 0.98 * R),
        pt(cx + 0.88 * R, cy - 1.37 * R),
        max(2, int(0.045 * R))
    )

    body_points = [
        pt(cx, cy - 0.91 * R),

        pt(cx - 0.22 * R, cy - 1.04 * R),
        pt(cx - 0.58 * R, cy - 1.10 * R),
        pt(cx - 0.93 * R, cy - 0.91 * R),
        pt(cx - 1.15 * R, cy - 0.55 * R),
        pt(cx - 1.24 * R, cy - 0.05 * R),
        pt(cx - 1.16 * R, cy + 0.49 * R),
        pt(cx - 0.96 * R, cy + 0.86 * R),
        pt(cx - 0.66 * R, cy + 1.10 * R),
        pt(cx - 0.33 * R, cy + 1.16 * R),

        pt(cx, cy + 1.06 * R),

        pt(cx + 0.33 * R, cy + 1.16 * R),
        pt(cx + 0.66 * R, cy + 1.10 * R),
        pt(cx + 0.96 * R, cy + 0.86 * R),
        pt(cx + 1.16 * R, cy + 0.49 * R),
        pt(cx + 1.24 * R, cy - 0.05 * R),
        pt(cx + 1.15 * R, cy - 0.55 * R),
        pt(cx + 0.93 * R, cy - 0.91 * R),
        pt(cx + 0.58 * R, cy - 1.10 * R),
        pt(cx + 0.22 * R, cy - 1.04 * R)
    ]

    pygame.draw.polygon(
        apple,
        (148, 30, 43),
        body_points
    )

    inner_points = []

    for px, py in body_points:
        inner_points.append(
            pt(
                cx + (px - cx) * 0.93,
                cy + (py - cy) * 0.93
            )
        )

    pygame.draw.polygon(
        apple,
        (239, 48, 65),
        inner_points
    )

    ellipse(
        (
            cx - 0.93 * R,
            cy - 0.82 * R,
            1.42 * R,
            0.90 * R
        ),
        (255, 79, 87)
    )

    ellipse(
        (
            cx - 0.83 * R,
            cy + 0.47 * R,
            1.78 * R,
            0.55 * R
        ),
        (200, 34, 52)
    )

    ellipse(
        (
            cx + 0.55 * R,
            cy - 0.38 * R,
            0.46 * R,
            1.08 * R
        ),
        (213, 37, 54)
    )

    ellipse(
        (
            cx - 0.36 * R,
            cy - 1.03 * R,
            0.72 * R,
            0.26 * R
        ),
        (174, 28, 47)
    )

    ellipse(
        (
            cx - 0.22 * R,
            cy - 0.99 * R,
            0.44 * R,
            0.12 * R
        ),
        (132, 31, 41)
    )

    ellipse(
        (
            cx - 0.88 * R,
            cy - 0.69 * R,
            0.39 * R,
            0.72 * R
        ),
        (255, 205, 205)
    )

    ellipse(
        (
            cx - 0.50 * R,
            cy - 0.63 * R,
            0.18 * R,
            0.18 * R
        ),
        (255, 231, 229)
    )

    eye_y = cy + int(0.08 * R)
    eye_w = int(0.31 * R)
    eye_h = int(0.43 * R)

    eye_positions = [
        cx - int(0.42 * R),
        cx + int(0.42 * R)
    ]

    for eye_x in eye_positions:
       
        ellipse(
            (
                eye_x - eye_w // 2,
                eye_y - eye_h // 2,
                eye_w,
                eye_h
            ),
            (61, 25, 27)
        )

        ellipse(
            (
                eye_x - int(0.09 * R),
                eye_y - int(0.15 * R),
                int(0.15 * R),
                int(0.17 * R)
            ),
            (255, 255, 255)
        )

        ellipse(
            (
                eye_x + int(0.045 * R),
                eye_y + int(0.01 * R),
                int(0.07 * R),
                int(0.08 * R)
            ),
            (255, 255, 255)
        )

        ellipse(
            (
                eye_x - int(0.10 * R),
                eye_y + int(0.11 * R),
                int(0.20 * R),
                int(0.08 * R)
            ),
            (122, 54, 37)
        )


    ellipse(
        (
            cx - 0.83 * R,
            cy + 0.31 * R,
            0.38 * R,
            0.22 * R
        ),
        (255, 125, 142)
    )

    ellipse(
        (
            cx + 0.45 * R,
            cy + 0.31 * R,
            0.38 * R,
            0.22 * R
        ),
        (255, 125, 142)
    )


    pygame.draw.arc(
        apple,
        (104, 20, 31),
        (
            cx - int(0.20 * R),
            cy + int(0.28 * R),
            int(0.40 * R),
            int(0.28 * R)
        ),
        math.pi,
        2 * math.pi,
        max(2, int(0.055 * R))
    )

    final_size = max(
        4,
        int(canvas_size / scale)
    )

    apple = pygame.transform.smoothscale(
        apple,
        (final_size, final_size)
    )

    surface.blit(
        apple,
        (
            int(x - 2 * r),
            int(y - 2.18 * r)
        )
    )
    

def draw_bomb(surface, x, y, r, elapsed):
    pygame.draw.circle(surface, (20, 22, 30), (x + 3, y + 4), r)
    pygame.draw.circle(surface, BLACK, (x, y), r)
    pygame.draw.ellipse(surface, (90, 93, 108), (x - r // 2, y - r // 2, r // 3, r // 2))
    pygame.draw.line(surface, BROWN, (x + r // 2, y - r // 2), (x + r, y - r), 5)
    draw_sparkle(surface, x + r + int(math.sin(elapsed * 10) * 4), y - r, 8)

    pygame.draw.line(surface, WHITE, (x - 14, y - 6), (x - 4, y - 1), 3)
    pygame.draw.line(surface, WHITE, (x + 14, y - 6), (x + 4, y - 1), 3)
    pygame.draw.circle(surface, WHITE, (x - 8, y + 2), 3)
    pygame.draw.circle(surface, WHITE, (x + 8, y + 2), 3)
    pygame.draw.arc(surface, WHITE, (x - 10, y + 8, 20, 12), math.pi, math.tau, 2)


def draw_basket(surface, rect):
    
    handle = pygame.Rect(
        rect.x + 20,
        rect.y - 48,
        rect.w - 40,
        82
    )

    pygame.draw.arc(
        surface,
        DARK_BROWN,
        handle,
        math.pi,
        math.tau,
        12
    )

    pygame.draw.arc(
        surface,
        (238, 190, 137),
        handle,
        math.pi,
        math.tau,
        7
    )

    shadow_points = [
        (rect.x + 13, rect.y + 10),
        (rect.right - 3, rect.y + 10),
        (rect.right - 20, rect.bottom + 7),
        (rect.x + 30, rect.bottom + 7)
    ]

    pygame.draw.polygon(
        surface,
        (104, 66, 52),
        shadow_points
    )

    basket_points = [
        (rect.x + 5, rect.y + 5),
        (rect.right - 5, rect.y + 5),
        (rect.right - 22, rect.bottom),
        (rect.x + 22, rect.bottom)
    ]

    pygame.draw.polygon(
        surface,
        (226, 165, 108),
        basket_points
    )

    pygame.draw.polygon(
        surface,
        DARK_BROWN,
        basket_points,
        4
    )

    for y_line in range(rect.y + 18, rect.bottom - 4, 12):
        pygame.draw.line(
            surface,
            (173, 111, 73),
            (rect.x + 17, y_line),
            (rect.right - 17, y_line),
            2
        )

    for x_line in range(rect.x + 28, rect.right - 15, 25):
        pygame.draw.line(
            surface,
            (191, 128, 82),
            (x_line, rect.y + 8),
            (x_line - 8, rect.bottom - 3),
            2
        )

    liner = pygame.Rect(
        rect.x + 3,
        rect.y - 2,
        rect.w - 6,
        21
    )

    pygame.draw.rect(
        surface,
        (255, 207, 224),
        liner,
        border_radius=10
    )

    pygame.draw.rect(
        surface,
        DARK_PINK,
        liner,
        3,
        border_radius=10
    )

    for cloth_x in range(rect.x + 18, rect.right - 5, 24):
        pygame.draw.circle(
            surface,
            (255, 207, 224),
            (cloth_x, rect.y + 17),
            8
        )

    face_y = rect.y + 35

    pygame.draw.circle(
        surface,
        INK,
        (rect.centerx - 18, face_y),
        4
    )

    pygame.draw.circle(
        surface,
        INK,
        (rect.centerx + 18, face_y),
        4
    )

    pygame.draw.circle(
        surface,
        WHITE,
        (rect.centerx - 19, face_y - 1),
        1
    )

    pygame.draw.circle(
        surface,
        WHITE,
        (rect.centerx + 17, face_y - 1),
        1
    )

    smile_points = [
        (rect.centerx - 10, face_y + 7),
        (rect.centerx - 5, face_y + 11),
        (rect.centerx, face_y + 13),
        (rect.centerx + 5, face_y + 11),
        (rect.centerx + 10, face_y + 7)
    ]

    pygame.draw.lines(
        surface,
        INK,
        False,
        smile_points,
        2
    )

    bow_y = rect.y + 8

    pygame.draw.ellipse(
        surface,
        PINK,
        (rect.centerx - 34, bow_y - 12, 28, 22)
    )

    pygame.draw.ellipse(
        surface,
        PINK,
        (rect.centerx + 6, bow_y - 12, 28, 22)
    )

    pygame.draw.circle(
        surface,
        DARK_PINK,
        (rect.centerx, bow_y),
        9
    )


class HandTracker:
    def __init__(self):
        download_model()
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.camera = cv2.VideoCapture(CAMERA_INDEX, backend)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.camera.isOpened():
            raise RuntimeError(
                "camera is not available"
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.smooth_x = None
        self.last_timestamp = 0
        self.frame_rgb = None
        self.hand_found = False

    def update(self):
        ok, frame = self.camera.read()
        if not ok:
            self.hand_found = False
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = max(int(time.perf_counter() * 1000), self.last_timestamp + 1)
        self.last_timestamp = timestamp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, timestamp)
        self.hand_found = bool(result.hand_landmarks)

        if self.hand_found:
            landmarks = result.hand_landmarks[0]
            palm_x = sum(landmarks[i].x for i in (0, 5, 9, 13, 17)) / 5
            self.smooth_x = palm_x if self.smooth_x is None else 0.72 * self.smooth_x + 0.28 * palm_x

            h, w, _ = rgb.shape
            for a, b in HAND_CONNECTIONS:
                p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
                p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
                cv2.line(rgb, p1, p2, (255, 170, 210), 3)
            for point in landmarks:
                p = (int(point.x * w), int(point.y * h))
                cv2.circle(rgb, p, 5, (255, 255, 255), -1)
                cv2.circle(rgb, p, 5, (218, 91, 151), 2)

        cv2.putText(
            rgb,
            "Hand detected!" if self.hand_found else "Show your hand",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self.frame_rgb = rgb
        return self.smooth_x if self.hand_found else None

    def preview(self):
        if self.frame_rgb is None:
            return None
        frame = cv2.resize(self.frame_rgb, (PREVIEW_W, PREVIEW_H))
        frame = np.ascontiguousarray(np.transpose(frame, (1, 0, 2)))
        return pygame.surfarray.make_surface(frame)

    def close(self):
        self.detector.close()
        self.camera.release()


class FallingItem:
    def __init__(self, score):
        self.kind = "bomb" if random.random() < 0.20 else "apple"
        self.r = random.randint(24, 31)
        self.base_x = random.randint(self.r + 15, WIDTH - self.r - 15)
        self.x = float(self.base_x)
        self.y = float(-self.r - random.randint(0, 80))
        self.phase = random.uniform(0, math.tau)
        self.sway = random.uniform(10, 30)
        self.speed = min(520, 205 + score * 6 + random.uniform(-15, 25))

    def update(self, dt, elapsed):
        self.y += self.speed * dt
        self.x = self.base_x + math.sin(elapsed * 2.2 + self.phase) * self.sway

    def draw(self, surface, elapsed):
        if self.kind == "apple":
            draw_apple(surface, int(self.x), int(self.y), self.r)
        else:
            draw_bomb(surface, int(self.x), int(self.y), self.r, elapsed)

    def rect(self):
        return pygame.Rect(int(self.x - self.r), int(self.y - self.r), self.r * 2, self.r * 2)


class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(70, 210)
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 45
        self.life = self.max_life = random.uniform(0.45, 0.85)
        self.color = color
        self.size = random.randint(3, 7)

    def update(self, dt):
        self.life -= dt
        self.vy += 230 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, surface):
        if self.life > 0:
            size = max(1, int(self.size * self.life / self.max_life))
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), size)


def draw_camera_card(screen, tracker, tiny_font):
    px, py = WIDTH - PREVIEW_W - 22, 20
    pygame.draw.rect(screen, WHITE, (px - 8, py - 8, PREVIEW_W + 16, PREVIEW_H + 40), border_radius=20)
    pygame.draw.rect(
        screen,
        PINK if tracker.hand_found else LAVENDER,
        (px - 8, py - 8, PREVIEW_W + 16, PREVIEW_H + 40),
        4,
        border_radius=20,
    )
    preview = tracker.preview()
    if preview is not None:
        screen.blit(preview, (px, py))
    draw_text(
        screen,
        "Hand found!" if tracker.hand_found else "SHOW YOUR HAND",
        tiny_font,
        DARK_PINK if tracker.hand_found else INK,
        center=(px + PREVIEW_W // 2, py + PREVIEW_H + 18),
    )


def main():
    pygame.init()
    pygame.display.set_caption("AI Fruit Catcher")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    title_font = get_font(45, True)
    big_font = get_font(40, True)
    medium_font = get_font(27, True)
    small_font = get_font(20)
    tiny_font = get_font(17)
    background = make_background()
    tracker = HandTracker()

    basket_w, basket_h = 150, 55
    basket_y = HEIGHT - 105
    basket_x = WIDTH / 2 - basket_w / 2
    state = "start"
    score = lives = best_score = 0
    items = []
    particles = []
    spawn_timer = elapsed = 0.0
    start_button = pygame.Rect(
        WIDTH // 2 - 150,
        410,
        300,
        65
    )

    restart_button = pygame.Rect(
        WIDTH // 2 - 175,
        420,
        350,
        62
    )
    def reset():
        nonlocal basket_x, score, lives, items, particles, spawn_timer, elapsed
        basket_x = WIDTH / 2 - basket_w / 2
        score, lives = 0, 3
        items, particles = [], []
        spawn_timer = elapsed = 0.0

    running = True
    try:
        while running:
            dt = min(clock.tick(FPS) / 1000.0, 0.05)
            elapsed += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if state == "start" and start_button.collidepoint(event.pos):
                            reset()
                            state = "playing"

                        elif state == "game_over" and restart_button.collidepoint(event.pos):
                            reset()
                            state = "playing"

            mouse_pos = pygame.mouse.get_pos()

            hand_x = tracker.update()
            keys = pygame.key.get_pressed()

            if state == "playing":
                if hand_x is not None:
                    target = hand_x * WIDTH - basket_w / 2
                    basket_x += (target - basket_x) * min(1.0, 12.0 * dt)
                else:
                    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                        basket_x -= 430 * dt
                    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                        basket_x += 430 * dt

                basket_x = clamp(basket_x, 10, WIDTH - basket_w - 10)
                basket_rect = pygame.Rect(int(basket_x), basket_y, basket_w, basket_h)

                spawn_timer += dt
                if spawn_timer >= max(0.34, 0.90 - score * 0.014):
                    spawn_timer = 0.0
                    items.append(FallingItem(score))

                for item in items:
                    item.update(dt, elapsed)

                for item in items[:]:
                    if item.rect().colliderect(basket_rect):
                        if item.kind == "apple":
                            score += 1
                            colors = (RED, PINK, GOLD)
                            amount = 18
                        else:
                            lives -= 1
                            colors = (BLACK, RED, GOLD)
                            amount = 24
                        particles.extend(Particle(item.x, item.y, random.choice(colors)) for _ in range(amount))
                        items.remove(item)
                        if lives <= 0:
                            best_score = max(best_score, score)
                            state = "game_over"
                            break
                    elif item.y - item.r > HEIGHT:
                        items.remove(item)

                for particle in particles:
                    particle.update(dt)
                particles = [p for p in particles if p.life > 0]

            screen.blit(background, (0, 0))
            for i in range(8):
                draw_sparkle(screen, 70 + i * 125, 190 + int(math.sin(elapsed * 1.7 + i) * 13), 4, WHITE)

            if state in ("playing", "game_over"):
                for item in items:
                    item.draw(screen, elapsed)
                basket_rect = pygame.Rect(int(basket_x), basket_y, basket_w, basket_h)
                draw_basket(screen, basket_rect)
                for particle in particles:
                    particle.draw(screen)

                pygame.draw.rect(screen, WHITE, (22, 20, 235, 92), border_radius=22)
                pygame.draw.rect(screen, PINK, (22, 20, 235, 92), 3, border_radius=22)
                draw_text(screen, f"Score: {score}", medium_font, DARK_PINK, topleft=(42, 34))
                draw_text(screen, f"Level: {1 + score // 10}", small_font, INK, topleft=(43, 73))
                for i in range(3):
                    draw_heart(screen, 55 + i * 42, 140, 28, RED if i < lives else (214, 207, 214))

            if state == "start":
                shade = pygame.Surface(
                    (WIDTH, HEIGHT),
                    pygame.SRCALPHA
                )
                shade.fill((255, 245, 250, 175))
                screen.blit(shade, (0, 0))

                card = pygame.Rect(
                    190,
                    250,
                    620,
                    260
                )

                pygame.draw.rect(
                    screen,
                    WHITE,
                    card,
                    border_radius=35
                )

                pygame.draw.rect(
                    screen,
                    PINK,
                    card,
                    5,
                    border_radius=35
                )

                draw_apple(
                    screen,
                    card.x + 80,
                    card.y + 75,
                    32
                )

                draw_apple(
                    screen,
                    card.right - 80,
                    card.y + 75,
                    32
                )

                draw_text(
                    screen,
                    "AI Fruit Catcher",
                    title_font,
                    DARK_PINK,
                    center=(WIDTH // 2, 320)
                )

                start_color = (
                    DARK_PINK
                    if start_button.collidepoint(mouse_pos)
                    else PINK
                )

                pygame.draw.rect(
                    screen,
                    start_color,
                    start_button,
                    border_radius=25
                )

                pygame.draw.rect(
                    screen,
                    DARK_PINK,
                    start_button,
                    4,
                    border_radius=25
                )

                draw_text(
                    screen,
                    "START THE GAME",
                    medium_font,
                    WHITE,
                    center=start_button.center
                )

                start_color = (
                    DARK_PINK
                    if start_button.collidepoint(mouse_pos)
                    else PINK
                )

                pygame.draw.rect(
                    screen,
                    start_color,
                    start_button,
                    border_radius=25
                )

                pygame.draw.rect(
                    screen,
                    DARK_PINK,
                    start_button,
                    4,
                    border_radius=25
                )

                draw_text(
                    screen,
                    "START THE GAME",
                    medium_font,
                    WHITE,
                    center=start_button.center
                )

            elif state == "game_over":
                shade = pygame.Surface(
                    (WIDTH, HEIGHT),
                    pygame.SRCALPHA
                )
                shade.fill((70, 45, 70, 150))
                screen.blit(shade, (0, 0))

                card = pygame.Rect(
                    250,
                    190,
                    500,
                    320
                )

                pygame.draw.rect(
                    screen,
                    CREAM,
                    card,
                    border_radius=35
                )

                pygame.draw.rect(
                    screen,
                    PINK,
                    card,
                    5,
                    border_radius=35
                )

                draw_text(
                    screen,
                    "Game over!",
                    title_font,
                    DARK_PINK,
                    center=(WIDTH // 2, 255)
                )

                draw_text(
                    screen,
                    f"Your score: {score}",
                    big_font,
                    INK,
                    center=(WIDTH // 2, 335)
                )

                draw_text(
                    screen,
                    f"Best score: {best_score}",
                    small_font,
                    DARK_BROWN,
                    center=(WIDTH // 2, 382)
                )

                restart_color = (
                    DARK_PINK
                    if restart_button.collidepoint(mouse_pos)
                    else PINK
                )

                pygame.draw.rect(
                    screen,
                    restart_color,
                    restart_button,
                    border_radius=22
                )

                pygame.draw.rect(
                    screen,
                    DARK_PINK,
                    restart_button,
                    3,
                    border_radius=22
                )

                draw_text(
                    screen,
                    "PLAY AGAIN",
                    medium_font,
                    WHITE,
                    center=restart_button.center
                )
            draw_camera_card(screen, tracker, tiny_font)
            pygame.display.flip()

    finally:
        tracker.close()
        pygame.quit()


if __name__ == "__main__":
    main()
import math
import random
import time
from dataclasses import dataclass

import pygame

# -----------------------------
# Config
# -----------------------------
WIDTH, HEIGHT = 800, 600
FPS = 60

GRID_SIZE = 3
TILE_SIZE = 90
GAP = 18
TOP_OFFSET = 110

START_SEQUENCE_LEN = 3
MAX_SEQUENCE_LEN = 25
INPUT_TIME_LIMIT = 10  # seconds

# Flash timing (will speed up with rounds)
BASE_FLASH_ON_MS = 550
BASE_FLASH_OFF_MS = 200
MIN_FLASH_ON_MS = 220
MIN_FLASH_OFF_MS = 90
POST_SHOW_PAUSE_MS = 450

# Colors
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (60, 60, 60)
DARK = (10, 10, 10)

RED = (231, 76, 60)
GREEN = (46, 204, 113)
BLUE = (52, 152, 219)
YELLOW = (241, 196, 15)
PALETTE = [RED, GREEN, BLUE, YELLOW]


# -----------------------------
# Simple tone synthesis (no assets)
# -----------------------------
def make_tone(freq_hz: float, duration_ms: int, volume: float = 0.25, sample_rate: int = 44100) -> pygame.mixer.Sound:
    """Generate a mono sine-wave tone as a pygame Sound."""
    n_samples = int(sample_rate * duration_ms / 1000.0)
    buf = bytearray()
    amp = int(32767 * max(0.0, min(volume, 1.0)))

    for i in range(n_samples):
        t = i / sample_rate
        sample = int(amp * math.sin(2.0 * math.pi * freq_hz * t))
        buf += int(sample).to_bytes(2, byteorder="little", signed=True)

    return pygame.mixer.Sound(buffer=bytes(buf))


@dataclass
class Sfx:
    tile_tones: list[pygame.mixer.Sound]
    success: pygame.mixer.Sound
    fail: pygame.mixer.Sound


def init_audio(num_tiles: int) -> Sfx:
    base = 440.0  # A4
    tile_tones = []
    for i in range(num_tiles):
        freq = base * (2 ** (i / 12))  # semitone steps
        tile_tones.append(make_tone(freq, duration_ms=160, volume=0.22))

    success = make_tone(880.0, 140, 0.25)
    fail = make_tone(220.0, 260, 0.30)
    return Sfx(tile_tones=tile_tones, success=success, fail=fail)


# -----------------------------
# Tile
# -----------------------------
@dataclass
class Tile:
    rect: pygame.Rect
    base_color: tuple[int, int, int]
    label: str
    is_lit: bool = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        color = self.base_color if self.is_lit else DARK
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        pygame.draw.rect(surface, GRAY, self.rect, 2, border_radius=12)

        text = font.render(self.label, True, WHITE)
        surface.blit(
            text,
            (self.rect.centerx - text.get_width() // 2, self.rect.centery - text.get_height() // 2),
        )

    def contains(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


def build_grid() -> list[Tile]:
    grid_px = GRID_SIZE * TILE_SIZE + (GRID_SIZE - 1) * GAP
    start_x = (WIDTH - grid_px) // 2
    start_y = TOP_OFFSET + (HEIGHT - TOP_OFFSET - grid_px) // 2

    tiles: list[Tile] = []
    idx = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x = start_x + c * (TILE_SIZE + GAP)
            y = start_y + r * (TILE_SIZE + GAP)
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
            base_color = PALETTE[idx % len(PALETTE)]
            label = str(idx + 1)
            tiles.append(Tile(rect=rect, base_color=base_color, label=label))
            idx += 1
    return tiles


def draw_sound_toggle(screen: pygame.Surface, ui_font: pygame.font.Font, sound_enabled: bool) -> pygame.Rect:
    """Draw a checkbox + label. Returns the clickable rect."""
    box_size = 18

    # Right-side UI block (score/round) lives near WIDTH-20; reserve space for it.
    right_block_w = 140
    padding = 20

    # Place sound toggle to the left of the right-side block.
    x = WIDTH - padding - right_block_w - 140
    y = 62  # aligns with row 2

    label = ui_font.render("Sound (M)", True, WHITE)
    screen.blit(label, (x, y - 2))

    box_x = x + label.get_width() + 10
    box_y = y
    box_rect = pygame.Rect(box_x, box_y, box_size, box_size)
    pygame.draw.rect(screen, WHITE, box_rect, 2)

    if sound_enabled:
        pygame.draw.line(screen, WHITE, (box_x + 3, box_y + 9), (box_x + 7, box_y + 13), 2)
        pygame.draw.line(screen, WHITE, (box_x + 7, box_y + 13), (box_x + 14, box_y + 4), 2)

    # Easier click area (label + box)
    click_rect = pygame.Rect(x - 6, y - 6, label.get_width() + 10 + box_size + 12, box_size + 12)
    return click_rect

def draw_ui(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    ui_font: pygame.font.Font,
    *,
    score: int,
    round_no: int,
    message: str,
    timer_s: int | None,
    progress: str | None,
) -> None:
    pygame.draw.rect(screen, BLACK, pygame.Rect(0, 0, WIDTH, TOP_OFFSET))

    padding = 20
    row1_y = 12
    row2_y = 58

    # Row 1: title (left)
    title = title_font.render("Short-Term Memory Game", True, WHITE)
    screen.blit(title, (padding, row1_y))

    # Row 1: score/round (right)
    score_surf = ui_font.render(f"Score: {score}", True, WHITE)
    round_surf = ui_font.render(f"Round: {round_no}", True, WHITE)

    right_x = WIDTH - padding - max(score_surf.get_width(), round_surf.get_width())
    screen.blit(score_surf, (right_x, row1_y + 6))
    screen.blit(round_surf, (right_x, row1_y + 30))

    # Row 2: message (left)
    if message:
        msg = ui_font.render(message, True, WHITE)
        screen.blit(msg, (padding, row2_y))

    # Row 2: time + progress (left/middle, after message area)
    info_x = padding
    info_y = row2_y + 26

    if timer_s is not None:
        t_surf = ui_font.render(f"Time: {timer_s}", True, WHITE)
        screen.blit(t_surf, (info_x, info_y))
        info_x += t_surf.get_width() + 24

    if progress is not None:
        p_surf = ui_font.render(progress, True, WHITE)
        screen.blit(p_surf, (info_x, info_y))

def render(
    screen: pygame.Surface,
    tiles: list[Tile],
    tile_font: pygame.font.Font,
    title_font: pygame.font.Font,
    ui_font: pygame.font.Font,
    sound_enabled: bool,
    *,
    score: int,
    round_no: int,
    message: str,
    timer_s: int | None = None,
    progress: str | None = None,
) -> pygame.Rect:
    """Draw frame. Returns the sound toggle clickable rect."""
    screen.fill((15, 15, 18))
    draw_ui(
        screen,
        title_font,
        ui_font,
        score=score,
        round_no=round_no,
        message=message,
        timer_s=timer_s,
        progress=progress,
    )

    for t in tiles:
        t.draw(screen, tile_font)

    toggle_rect = draw_sound_toggle(screen, ui_font, sound_enabled)
    pygame.display.flip()
    return toggle_rect


def calc_flash_ms(round_no: int) -> tuple[int, int]:
    on_ms = max(MIN_FLASH_ON_MS, BASE_FLASH_ON_MS - (round_no - 1) * 20)
    off_ms = max(MIN_FLASH_OFF_MS, BASE_FLASH_OFF_MS - (round_no - 1) * 8)
    return on_ms, off_ms


def pump_quit_events() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False


def flash_tile(
    screen: pygame.Surface,
    tiles: list[Tile],
    sfx: Sfx,
    tile_font: pygame.font.Font,
    title_font: pygame.font.Font,
    ui_font: pygame.font.Font,
    sound_enabled: bool,
    *,
    index: int,
    score: int,
    round_no: int,
    message: str,
) -> bool:
    on_ms, off_ms = calc_flash_ms(round_no)

    tiles[index].is_lit = True
    if sound_enabled:
        sfx.tile_tones[index].play()

    render(
        screen, tiles, tile_font, title_font, ui_font, sound_enabled,
        score=score, round_no=round_no, message=message
    )

    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < on_ms:
        if pump_quit_events():
            return True

    tiles[index].is_lit = False
    render(
        screen, tiles, tile_font, title_font, ui_font, sound_enabled,
        score=score, round_no=round_no, message=message
    )

    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < off_ms:
        if pump_quit_events():
            return True

    return False


def show_sequence(
    screen: pygame.Surface,
    tiles: list[Tile],
    sequence: list[int],
    sfx: Sfx,
    tile_font: pygame.font.Font,
    title_font: pygame.font.Font,
    ui_font: pygame.font.Font,
    sound_enabled: bool,
    *,
    score: int,
    round_no: int,
) -> bool:
    if pump_quit_events():
        return True

    render(
        screen, tiles, tile_font, title_font, ui_font, sound_enabled,
        score=score, round_no=round_no, message="Watch the sequence…"
    )

    pygame.time.wait(300)
    for idx in sequence:
        if flash_tile(
            screen, tiles, sfx, tile_font, title_font, ui_font, sound_enabled,
            index=idx, score=score, round_no=round_no, message="Watch the sequence…"
        ):
            return True

    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < POST_SHOW_PAUSE_MS:
        if pump_quit_events():
            return True

    return False


def find_clicked_tile(tiles: list[Tile], pos: tuple[int, int]) -> int | None:
    for i, tile in enumerate(tiles):
        if tile.contains(pos):
            return i
    return None


class RestartGame(Exception):
    pass


def game_over_loop(
    screen: pygame.Surface,
    tiles: list[Tile],
    sfx: Sfx,
    tile_font: pygame.font.Font,
    title_font: pygame.font.Font,
    ui_font: pygame.font.Font,
    sound_enabled: bool,
    *,
    score: int,
    round_no: int,
) -> bool:
    """Return updated sound_enabled. Raises RestartGame on restart."""
    if sound_enabled:
        sfx.fail.play()

    message = "Game Over! Press R to restart, M to mute, ESC to quit."
    while True:
        toggle_rect = render(
            screen, tiles, tile_font, title_font, ui_font, sound_enabled,
            score=score, round_no=round_no, message=message
        )
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return sound_enabled
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return sound_enabled
                if event.key == pygame.K_r:
                    raise RestartGame
                if event.key == pygame.K_m:
                    sound_enabled = not sound_enabled
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if toggle_rect.collidepoint(event.pos):
                    sound_enabled = not sound_enabled


def main() -> None:
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Short-Term Memory Game")
    clock = pygame.time.Clock()

    title_font = pygame.font.Font(None, 44)
    ui_font = pygame.font.Font(None, 28)
    tile_font = pygame.font.Font(None, 36)

    tiles = build_grid()
    sfx = init_audio(len(tiles))
    sound_enabled = False

    while True:
        try:
            score = 0
            round_no = 1
            seq_len = START_SEQUENCE_LEN

            while True:
                sequence = [random.randrange(len(tiles)) for _ in range(seq_len)]

                if show_sequence(
                    screen, tiles, sequence, sfx, tile_font, title_font, ui_font, sound_enabled,
                    score=score, round_no=round_no
                ):
                    return

                player: list[int] = []
                deadline = time.time() + INPUT_TIME_LIMIT

                while time.time() < deadline and len(player) < len(sequence):
                    remaining = max(0, int(deadline - time.time()))
                    progress = f"Progress: {len(player)}/{len(sequence)}"
                    toggle_rect = render(
                        screen, tiles, tile_font, title_font, ui_font, sound_enabled,
                        score=score, round_no=round_no,
                        message="Your turn: repeat the sequence.",
                        timer_s=remaining,
                        progress=progress,
                    )

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                return
                            if event.key == pygame.K_m:
                                sound_enabled = not sound_enabled
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            # toggle click?
                            if toggle_rect.collidepoint(event.pos):
                                sound_enabled = not sound_enabled
                                continue

                            clicked = find_clicked_tile(tiles, event.pos)
                            if clicked is None:
                                continue

                            tiles[clicked].is_lit = True
                            if sound_enabled:
                                sfx.tile_tones[clicked].play()

                            render(
                                screen, tiles, tile_font, title_font, ui_font, sound_enabled,
                                score=score, round_no=round_no,
                                message="Your turn: repeat the sequence.",
                                timer_s=remaining,
                                progress=progress,
                            )
                            pygame.time.wait(120)
                            tiles[clicked].is_lit = False

                            expected = sequence[len(player)]
                            if clicked != expected:
                                sound_enabled = game_over_loop(
                                    screen, tiles, sfx, tile_font, title_font, ui_font, sound_enabled,
                                    score=score, round_no=round_no
                                )
                                return

                            player.append(clicked)

                    clock.tick(FPS)

                if len(player) != len(sequence):
                    sound_enabled = game_over_loop(
                        screen, tiles, sfx, tile_font, title_font, ui_font, sound_enabled,
                        score=score, round_no=round_no
                    )
                    return

                if sound_enabled:
                    sfx.success.play()
                score += 1
                round_no += 1
                seq_len = min(seq_len + 1, MAX_SEQUENCE_LEN)

                end = pygame.time.get_ticks() + 450
                while pygame.time.get_ticks() < end:
                    if pump_quit_events():
                        return
                    render(
                        screen, tiles, tile_font, title_font, ui_font, sound_enabled,
                        score=score, round_no=round_no - 1,
                        message="Correct! Next round…"
                    )
                    clock.tick(FPS)

        except RestartGame:
            continue
        finally:
            pygame.event.clear()


if __name__ == "__main__":
    main()
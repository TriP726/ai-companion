"""Design-token theme system.

Every colour, radius, and type size in the app comes from here. Nothing else
may hardcode a hex value — that is what makes a light/dark toggle possible
without touching every widget.

Aesthetic: terminal / cyberpunk. Monospace for data and code, a tight neon
accent, hard-edged panels with thin glowing borders. The indigo accent carries
interactive meaning; every other hue is reserved for a specific semantic
(success, warning, danger, private) so colour always means something.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class ThemeMode(str, Enum):
    DARK = "dark"
    LIGHT = "light"


@dataclass(frozen=True)
class Palette:
    """Semantic colour tokens. Never reference a raw hex outside this file."""

    # Surfaces, darkest to lightest (dark mode) / lightest to darkest (light)
    bg_base: str          # window background, furthest back
    bg_surface: str       # panels sitting on the base
    bg_elevated: str      # cards, bubbles, inputs
    bg_hover: str         # hover state for interactive surfaces
    bg_active: str        # pressed / selected surface

    # Text
    fg_primary: str       # body text
    fg_secondary: str     # labels, metadata
    fg_muted: str         # placeholders, disabled

    # Structure
    border: str           # default 1px separators
    border_strong: str    # emphasised edges
    grid: str             # faint background grid lines
    link: str             # graph/mind-map connections - must stay readable

    # Accent — interactive meaning only
    accent: str
    accent_hover: str
    accent_press: str
    accent_fg: str        # text drawn on top of accent
    accent_glow: str      # translucent accent for focus rings

    # Semantics — each hue means exactly one thing
    success: str          # model ready, operation succeeded
    warning: str          # caution, degraded
    danger: str           # destructive, errors
    private: str          # Private-mode indicator, deliberately unique

    # Chat roles
    role_user: str
    role_assistant: str
    role_system: str

    # Code
    code_fg: str
    code_bg: str


DARK = Palette(
    # Deep-purple holographic HUD. Near-black violet base so emissive
    # elements read as projected light rather than painted surfaces.
    bg_base="#06040f",
    bg_surface="#0c0820",
    bg_elevated="#140e2e",
    bg_hover="#1d1440",
    bg_active="#2a1c5c",
    fg_primary="#e8dcff",
    fg_secondary="#a894d8",
    fg_muted="#6b5a99",
    border="#2a1f52",
    border_strong="#4a3585",
    grid="#160f33",
    link="#7c5fc0",
    accent="#a855f7",
    accent_hover="#c084fc",
    accent_press="#8b3fd9",
    accent_fg="#0a0618",
    accent_glow="rgba(168, 85, 247, 0.45)",
    success="#2fe6a8",
    warning="#ffc44d",
    danger="#ff4d6d",
    private="#ff9a3c",
    role_user="#22d3ee",
    role_assistant="#a855f7",
    role_system="#6b5a99",
    code_fg="#d8b4fe",
    code_bg="#0a0620",
)

# Light mode was dropped: a holographic HUD reads as projected light, and a
# pale variant of that is incoherent. LIGHT stays defined - and identical to
# DARK - so any code or test that still references it keeps working instead of
# raising AttributeError. The theme toggle is now a no-op by design.
LIGHT = DARK


@dataclass(frozen=True)
class Metrics:
    """Spacing, radius and type scale.

    Spacing is a strict 4px scale — the single biggest reason UIs look
    unprofessional is inconsistent, eyeballed padding.
    """

    space_1: int = 4
    space_2: int = 8
    space_3: int = 12
    space_4: int = 16
    space_5: int = 24
    space_6: int = 32

    radius_sm: int = 2
    radius_md: int = 4
    radius_lg: int = 6

    # Cyberpunk reads as hard-edged: radii stay small on purpose.
    font_ui: str = "'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif"
    font_mono: str = "'Cascadia Code', 'JetBrains Mono', Consolas, monospace"

    text_xs: int = 11
    text_sm: int = 12
    text_md: int = 13
    text_lg: int = 15
    text_xl: int = 19

    rail_width: int = 78
    sidebar_width: int = 260


METRICS = Metrics()


class Theme:
    """Active theme. Holds the palette and builds the global stylesheet."""

    PALETTES: ClassVar[dict[ThemeMode, Palette]] = {
        ThemeMode.DARK: DARK,
        ThemeMode.LIGHT: LIGHT,
    }

    def __init__(self, mode: ThemeMode = ThemeMode.DARK) -> None:
        self._mode = mode

    @property
    def mode(self) -> ThemeMode:
        return self._mode

    @property
    def p(self) -> Palette:
        return self.PALETTES[self._mode]

    @property
    def m(self) -> Metrics:
        return METRICS

    def set_mode(self, mode: ThemeMode) -> None:
        self._mode = mode

    def toggle(self) -> ThemeMode:
        self._mode = (
            ThemeMode.LIGHT if self._mode is ThemeMode.DARK else ThemeMode.DARK
        )
        return self._mode

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------

    def stylesheet(self) -> str:
        p, m = self.p, self.m
        return f"""
/* ---------- base ---------- */
QWidget {{
    background-color: {p.bg_surface};
    color: {p.fg_primary};
    font-family: {m.font_ui};
    font-size: {m.text_md}px;
}}
QMainWindow, QDialog {{
    background-color: {p.bg_base};
}}
QToolTip {{
    background-color: {p.bg_elevated};
    color: {p.fg_primary};
    border: 1px solid {p.accent};
    padding: {m.space_2}px;
    font-size: {m.text_sm}px;
}}

/* ---------- nav rail ---------- */
QFrame#NavRail {{
    background-color: {p.bg_base};
    border-right: 1px solid {p.border};
    min-width: {m.rail_width}px;
    max-width: {m.rail_width}px;
}}
QToolButton#NavButton {{
    background-color: transparent;
    border: none;
    border-left: 2px solid transparent;
    color: {p.fg_muted};
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    padding: {m.space_3}px {m.space_1}px;
}}
QToolButton#NavButton:hover {{
    background-color: {p.bg_hover};
    color: {p.fg_primary};
}}
QToolButton#NavButton:checked {{
    background-color: {p.bg_elevated};
    border-left: 2px solid {p.accent};
    color: {p.accent};
}}

/* ---------- panels ---------- */
QFrame#Panel {{
    background-color: {p.bg_surface};
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
}}
QFrame#Sidebar {{
    background-color: {p.bg_base};
    border-right: 1px solid {p.border};
}}
QFrame#HeaderBar {{
    background-color: {p.bg_base};
    border-bottom: 1px solid {p.border};
}}
QFrame#Divider {{
    background-color: {p.border};
    max-height: 1px;
    border: none;
}}

/* ---------- typography ---------- */
QLabel#TitleText {{
    font-size: {m.text_xl}px;
    font-weight: 600;
    color: {p.fg_primary};
}}
QLabel#SectionLabel {{
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 600;
    color: {p.fg_muted};
    letter-spacing: 1px;
    padding: {m.space_2}px {m.space_3}px {m.space_1}px {m.space_3}px;
}}
QLabel#MetaText {{
    color: {p.fg_secondary};
    font-size: {m.text_sm}px;
}}
QLabel#MonoText {{
    font-family: {m.font_mono};
    font-size: {m.text_sm}px;
    color: {p.fg_secondary};
}}
QLabel#StatusOk    {{ color: {p.success}; font-family: {m.font_mono};
                      font-size: {m.text_xs}px; }}
QLabel#StatusWarn  {{ color: {p.warning}; font-family: {m.font_mono};
                      font-size: {m.text_xs}px; }}
QLabel#StatusError {{ color: {p.danger};  font-family: {m.font_mono};
                      font-size: {m.text_xs}px; }}
QLabel#PrivateBadge, QPushButton#PrivateBadge {{
    color: {p.private};
    background-color: {p.bg_elevated};
    border: 1px solid {p.private};
    border-radius: {m.radius_sm}px;
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 700;
    padding: {m.space_1}px {m.space_2}px;
}}
QPushButton#PrivateBadge:hover {{
    background-color: {p.private};
    color: {p.bg_base};
}}

/* Saving (normal mode): quiet green, must not compete with the accent. */
QPushButton#SavingBadge {{
    color: {p.success};
    background-color: {p.bg_elevated};
    border: 1px solid {p.success};
    border-radius: {m.radius_sm}px;
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 700;
    padding: {m.space_1}px {m.space_2}px;
}}
QPushButton#SavingBadge:hover {{
    background-color: {p.success};
    color: {p.bg_base};
}}

/* ---------- buttons ---------- */
QPushButton {{
    background-color: {p.bg_elevated};
    color: {p.fg_secondary};
    border: 1px solid {p.border_strong};
    border-radius: 2px;
    padding: {m.space_2}px {m.space_4}px;
    font-family: {m.font_mono};
    font-size: {m.text_sm}px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QPushButton:hover {{
    background-color: {p.bg_hover};
    border-color: {p.accent};
    color: {p.fg_primary};
}}
QPushButton:pressed {{ background-color: {p.bg_active}; }}
QPushButton:disabled {{
    color: {p.fg_muted};
    border-color: {p.border};
    background-color: {p.bg_surface};
}}
QPushButton#PrimaryButton {{
    background-color: {p.accent};
    color: {p.accent_fg};
    border: 1px solid {p.accent};
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover  {{ background-color: {p.accent_hover};
                                    border-color: {p.accent_hover}; }}
QPushButton#PrimaryButton:pressed{{ background-color: {p.accent_press}; }}
QPushButton#PrimaryButton:disabled {{
    background-color: {p.bg_elevated};
    color: {p.fg_muted};
    border-color: {p.border};
}}
QPushButton#DangerButton {{
    background-color: transparent;
    color: {p.danger};
    border: 1px solid {p.danger};
}}
QPushButton#DangerButton:hover {{ background-color: {p.danger};
                                  color: {p.accent_fg}; }}
QPushButton#GhostButton {{
    background-color: transparent;
    border: 1px solid transparent;
    color: {p.fg_secondary};
    padding: {m.space_2}px {m.space_3}px;
}}
QPushButton#GhostButton:hover {{
    background-color: {p.bg_hover};
    color: {p.fg_primary};
}}

/* ---------- inputs ---------- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {p.bg_elevated};
    color: {p.fg_primary};
    border: 1px solid {p.border_strong};
    border-radius: {m.radius_md}px;
    padding: {m.space_2}px {m.space_3}px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_fg};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {p.accent};
}}
QLineEdit::placeholder {{ color: {p.fg_muted}; }}
QLineEdit#ChatInput, QTextEdit#ChatInput {{
    font-size: {m.text_lg}px;
    padding: {m.space_2}px {m.space_4}px;
    background-color: {p.bg_elevated};
    border: 1px solid {p.border_strong};
    border-radius: {m.radius_md}px;
    color: {p.fg_primary};
}}
QTextEdit#ChatInput:focus {{
    border: 1px solid {p.accent};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {p.bg_elevated};
    border: 1px solid {p.border_strong};
    selection-background-color: {p.accent};
    selection-color: {p.accent_fg};
    outline: none;
}}

/* ---------- chat transcript ---------- */
QTextEdit#ChatDisplay {{
    background-color: {p.bg_surface};
    border: none;
    padding: {m.space_4}px;
    font-size: {m.text_md}px;
}}

/* ---------- code surfaces ---------- */
QTextEdit#CodeSurface {{
    background-color: {p.code_bg};
    color: {p.fg_primary};
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
    font-family: {m.font_mono};
    font-size: {m.text_sm}px;
    padding: {m.space_3}px;
}}
QTextEdit#OutputSurface {{
    background-color: {p.bg_base};
    color: {p.success};
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
    font-family: {m.font_mono};
    font-size: {m.text_sm}px;
    padding: {m.space_3}px;
}}


/* ---------- voice badge ---------- */
QPushButton#VoiceBadge {{
    color: {p.fg_muted};
    background-color: {p.bg_elevated};
    border: 1px solid {p.border_strong};
    border-radius: 2px;
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 700;
    padding: {m.space_1}px {m.space_2}px;
}}
QPushButton#VoiceBadge:hover {{ border-color: {p.accent}; color: {p.fg_secondary}; }}
QPushButton#VoiceBadgeOn {{
    color: {p.success};
    background-color: {p.bg_elevated};
    border: 1px solid {p.success};
    border-radius: 2px;
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 700;
    padding: {m.space_1}px {m.space_2}px;
}}
QPushButton#VoiceBadgeActive {{
    color: {p.accent_fg};
    background-color: {p.accent};
    border: 1px solid {p.accent};
    border-radius: 2px;
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 700;
    padding: {m.space_1}px {m.space_2}px;
}}

/* ---------- HUD chrome ---------- */
QStackedWidget {{ background: transparent; }}
QWidget#HudHeader {{
    background-color: {p.bg_surface};
    border-bottom: 1px solid {p.border};
}}
QWidget#HudHeader QWidget {{ background-color: transparent; }}

QLabel#PrivateNotice {{
    color: {p.private};
    background-color: {p.bg_elevated};
    border: 1px solid {p.private};
    border-radius: 2px;
    padding: {m.space_2}px {m.space_3}px;
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 700;
}}

/* ---------- HUD telemetry ---------- */
QLabel#TelemetryKey {{
    color: {p.fg_muted};
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    letter-spacing: 1.5px;
}}
QLabel#TelemetryValue {{
    color: {p.accent};
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#HudTitle {{
    color: {p.fg_primary};
    font-family: {m.font_mono};
    font-size: {m.text_lg}px;
    font-weight: 700;
    letter-spacing: 3px;
}}
QLabel#SectionLabel {{
    color: {p.fg_muted};
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    font-weight: 700;
    letter-spacing: 2px;
}}

/* ---------- lists & trees ---------- */
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {p.bg_base};
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
    outline: none;
    padding: {m.space_1}px;
}}
QListWidget::item, QTreeWidget::item {{
    padding: {m.space_2}px {m.space_3}px;
    border-radius: {m.radius_sm}px;
    color: {p.fg_secondary};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {p.bg_hover};
    color: {p.fg_primary};
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {p.bg_active};
    color: {p.accent};
    border-left: 2px solid {p.accent};
}}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {p.border_strong};
    border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.accent}; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {p.border_strong};
    border-radius: 5px; min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {p.accent}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ---------- menu / status ---------- */
QMenuBar {{
    background-color: {p.bg_base};
    border-bottom: 1px solid {p.border};
    padding: {m.space_1}px;
}}
QMenuBar::item {{
    padding: {m.space_2}px {m.space_3}px;
    border-radius: {m.radius_sm}px;
    color: {p.fg_secondary};
}}
QMenuBar::item:selected {{ background-color: {p.bg_hover};
                           color: {p.fg_primary}; }}
QMenu {{
    background-color: {p.bg_elevated};
    border: 1px solid {p.border_strong};
    padding: {m.space_1}px;
}}
QMenu::item {{ padding: {m.space_2}px {m.space_5}px;
               border-radius: {m.radius_sm}px; }}
QMenu::item:selected {{ background-color: {p.accent};
                        color: {p.accent_fg}; }}
QMenu::separator {{ height: 1px; background: {p.border};
                    margin: {m.space_1}px {m.space_2}px; }}
QStatusBar {{
    background-color: {p.bg_base};
    border-top: 1px solid {p.border};
    color: {p.fg_muted};
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
}}
QStatusBar::item {{ border: none; }}

/* ---------- tabs (settings dialog) ---------- */
QTabWidget::pane {{
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
    background-color: {p.bg_surface};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {p.fg_muted};
    border: none;
    border-bottom: 2px solid transparent;
    padding: {m.space_2}px {m.space_4}px;
    font-family: {m.font_mono};
    font-size: {m.text_sm}px;
}}
QTabBar::tab:hover {{ color: {p.fg_primary}; }}
QTabBar::tab:selected {{
    color: {p.accent};
    border-bottom: 2px solid {p.accent};
}}

/* ---------- misc ---------- */
QSplitter {{ background: transparent; }}
QSplitter::handle {{ background-color: {p.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background-color: {p.accent}; }}
QGroupBox {{
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
    margin-top: {m.space_3}px;
    padding-top: {m.space_3}px;
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
    color: {p.fg_muted};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {m.space_3}px;
    padding: 0 {m.space_2}px;
}}
QCheckBox {{ spacing: {m.space_2}px; color: {p.fg_secondary}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {p.border_strong};
    border-radius: {m.radius_sm}px;
    background-color: {p.bg_elevated};
}}
QCheckBox::indicator:checked {{
    background-color: {p.accent};
    border-color: {p.accent};
}}
QProgressBar {{
    background-color: {p.bg_elevated};
    border: 1px solid {p.border};
    border-radius: {m.radius_sm}px;
    text-align: center;
    font-family: {m.font_mono};
    font-size: {m.text_xs}px;
}}
QProgressBar::chunk {{ background-color: {p.accent}; }}
QGraphicsView {{
    background-color: {p.bg_base};
    border: 1px solid {p.border};
    border-radius: {m.radius_md}px;
}}
"""


# Module-level singleton — panels import this rather than constructing their own.
theme = Theme(ThemeMode.DARK)

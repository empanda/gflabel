"""
Labels for "Modern Gridfinity Case"

https://www.printables.com/model/894202-modern-gridfinity-case
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import pint
from build123d import (
    Align,
    Axis,
    Box,
    BuildLine,
    BuildPart,
    BuildSketch,
    Locations,
    Mode,
    Plane,
    Polyline,
    Select,
    Vector,
    add,
    chamfer,
    extrude,
    make_face,
    mirror,
)

from ..util import unit_registry
from . import LabelBase

logger = logging.getLogger(__name__)


def round_step(value, step):
    """Round to a step, the same way onshape does"""
    return math.floor(value / step + 0.5) * step


class ModernBase(LabelBase):
    """
    Generate a Modern-Gridfinity-Case label body

    Origin is positioned at the center of the label, with the label
    surface at z=0.
    """

    def __init__(self, args: argparse.Namespace):
        # Temporary: Pull these out of the args without checking
        # width = args.width
        # height_mm = args.height

        # Label depth (as of Rev2.3) is dependent on acetate window depth:
        #       depth = #window_t + 1.2
        # Just treat this as a configurable depth, and we can change this
        # to be something easier to remember later
        LABEL_DEPTH = (
            args.label_depth.to("mm").magnitude
            if args.label_depth
            else round_step(args.window_depth, 0.2) + 1.2 + 0.2
        )
        INTERNAL_WIDTHS = {3: 16, 4: 35, 5: 60, 6: 100, 7: 125, 8: 125}
        MARGIN_X = 8
        MARGIN_Y = 4.7
        WINDOW_HEIGHT = 13
        INDENT_DEPTH = 0.6  # How deep the indent on the back is
        X_OFF_FACE = 0.1415  # How far the angled faces width moves with offset face
        X_OFF_BASE = 0.1  # How far the boxed base moves with offset face

        def _convert_u_to_mm(u: pint.Quantity):
            if u.magnitude not in INTERNAL_WIDTHS:
                logger.error(
                    "'Modern' label u-dimensions only known for 3u-8u boxes. Specify mm for custom sizes."
                )
                sys.exit(1)
            return pint.Quantity(
                INTERNAL_WIDTHS[u.magnitude] + MARGIN_X * 2 - 2 * X_OFF_BASE,
                "mm",
            )

        with unit_registry.context("u", fn=_convert_u_to_mm):
            W_mm = args.width.to("mm").magnitude

        # Work out the window width. We know for fixed U, but must
        # reverse calculate for custom widths.
        W_window = W_mm - MARGIN_X * 2 + 2 * X_OFF_BASE

        # Height basis. Used for centering, but the actual height is affected by offset
        H_mm = MARGIN_Y * 2 + WINDOW_HEIGHT
        # Top offset due to multiple face inset in onshape. This is non-
        # trivial to calculate as depends on offsetting doubly-angled
        # faces. Custom heights are assumed to have the same offset.
        Y_offset = 0.28284

        if args.height is not None:
            H_mm = args.height.to("mm").magnitude

        if LABEL_DEPTH >= H_mm / 2 or LABEL_DEPTH >= W_mm / 2:
            raise ValueError(
                f"Error: Cannot have label depth ({LABEL_DEPTH:.1f} mm) being greater than half the width ({W_mm / 2:.1f} mm) or height ({H_mm / 2:.1f} mm)"
            )

        # Rather than sweeping a triangle (original), work out the inner
        # edge and extrude in both directions from there
        with BuildPart() as part:
            with BuildSketch() as _sketch:
                with BuildLine() as _line:
                    corner_offset = 1.97574  # Measured after face offset
                    # dx = difference between actual max width (bottom)
                    # and the width of the main body (top, chamfered edge)
                    dx = abs(X_OFF_BASE - X_OFF_FACE)
                    Polyline(
                        [
                            (0, -H_mm / 2),
                            (-W_mm / 2 + dx, -H_mm / 2),
                            (-W_mm / 2 + dx, H_mm / 2 - corner_offset - Y_offset),
                            (-W_mm / 2 + dx + corner_offset, H_mm / 2 - Y_offset),
                            (0, H_mm / 2 - Y_offset),
                        ]
                    )
                    mirror(_line.line, Plane.YZ)
                make_face()
            extrude(amount=LABEL_DEPTH / 2, taper=45)
            mid_face = part.faces().sort_by(Axis.Z)[0]
            extrude(mid_face, amount=LABEL_DEPTH / 2, taper=-45)

            # Add the flattened base
            with BuildPart(mode=Mode.PRIVATE) as _bottom_part:
                with Locations([(0, -H_mm / 2, -LABEL_DEPTH / 2)]):
                    Box(
                        W_mm,
                        0.95858 + LABEL_DEPTH,
                        LABEL_DEPTH,
                        align=(Align.CENTER, Align.MIN, Align.CENTER),
                    )

                    edges = (
                        _bottom_part.edges(Select.LAST)
                        .filter_by(Axis.Z)
                        .group_by(Axis.Y)[-1]
                    )
                    chamfer(edges, length=LABEL_DEPTH)
            add(_bottom_part.part)
            del _bottom_part

            # Add the indent
            # 60mm x 13mm, 4.7m from bottom
            with Locations([(0, -H_mm / 2 + MARGIN_Y, -LABEL_DEPTH)]):
                Box(
                    W_window,
                    WINDOW_HEIGHT,
                    INDENT_DEPTH,
                    mode=Mode.SUBTRACT,
                    align=(Align.CENTER, Align.MIN, Align.MIN),
                )

        self.part = part.part
        self.area = Vector(W_mm, H_mm)

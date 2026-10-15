"""One definition of the escape rate, because there were three.

A landmark's distance to the fitted ear is NaN when the predicted pixel was
background: the ray hit nothing, so there is no 3D point and no distance.
That is the worst thing the 2D model can do, not a missing observation.

Three conventions were in use in this repository and all three are wrong in
different directions:

    d > gate                scores NaN as a SUCCESS, because NaN > x is
                            False. handedness_vs_illumination did this until
                            2026-10-14 and it manufactured a result -- the
                            arms do not miss equally, so the forgiving
                            convention favoured whichever arm missed most.
    d[isfinite(d)] > gate   DROPS them. Defensible for one arm, wrong for
                            comparing two, and it conditions on the outcome.
    ~(d <= gate)            counts them, which is what "not within `gate` mm
                            of the ear" means for a point that is nowhere
                            near any geometry.

Use `escape_rate`, and print `off_head` beside it so the two failure modes
stay visible as two.
"""

import numpy as np


def escape_rate(d, gate):
    """Fraction past `gate`, counting a landmark that missed the head."""
    return float(np.mean(~(np.asarray(d, float) <= gate)))


def off_head(d):
    """Fraction whose pixel was background, so the lift had nothing to hit."""
    return float(np.mean(~np.isfinite(np.asarray(d, float))))

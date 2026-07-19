"""Face-flashing active-light liveness verifier.

The screen shows random background/lighting frame pairs; a live face reflects
each flash, so its brightness tracks the challenge sequence. Replayed video or
a virtual camera cannot follow randomly chosen flashes.

Consumers import the submodules directly (active_light, challenges, face).
"""

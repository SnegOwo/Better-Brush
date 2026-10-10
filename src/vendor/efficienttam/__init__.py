# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Better Brush modification: the upstream package registers a Hydra search
# path on import.  That global singleton is process-wide and would clash with
# the ``sam2`` package Better Brush also imports (whichever loads first wins,
# and the other one's configs would go missing).  Configs are loaded directly
# from disk in ``build_efficienttam.py`` instead, so no global Hydra state is
# touched here.

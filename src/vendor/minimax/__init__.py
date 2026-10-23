# Copyright (c) 2025 MiniMax-Remover authors (Bojia Zi et al.).
# All rights reserved.
#
# Upstream: https://github.com/zibojia/MiniMax-Remover
# (transformer_minimax_remover.py). The upstream repository publishes no license
# file, so its source remains all rights reserved by its authors; this adapted
# module is included for interoperability, with attribution retained. See
# NOTICE.md in this directory for provenance and the list of modifications.
#
# Better Brush modifications:
#   * class renamed Transformer3DModel -> MinimaxRemoverTransformer3DModel
#     (the upstream repo's file used diffusers' historical class name; the
#     rename avoids clashing with the Wan transformer shipped by diffusers)
#   * vendored under src/vendor/minimax so the package resolves without the
#     upstream repo on sys.path

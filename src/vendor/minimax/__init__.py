# Copyright (c) 2025 MiniMax-Remover authors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Upstream: https://github.com/zibojia/MiniMax-Remover
# (transformer_minimax_remover.py)
#
# Better Brush modifications:
#   * class renamed Transformer3DModel -> MinimaxRemoverTransformer3DModel
#     (the upstream repo's file used diffusers' historical class name; the
#     rename avoids clashing with the Wan transformer shipped by diffusers)
#   * vendored under src/vendor/minimax so the package resolves without the
#     upstream repo on sys.path

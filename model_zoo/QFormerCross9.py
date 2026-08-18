# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

from .QFormerCross8 import QFormerCross8


class QFormerCross9(QFormerCross8):
    """Three-layer NS recurrence variant of QFormerCross8.

    Both the non-sequential and sequence CrossValue stacks contain three
    residual recurrence layers by default.  All other V8 data flow and
    operators remain unchanged for a controlled depth comparison.
    """

    def __init__(self,
                 feature_map,
                 model_id="QFormerCross9",
                 num_ns_layers=3,
                 **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            num_ns_layers=num_ns_layers,
            **kwargs,
        )


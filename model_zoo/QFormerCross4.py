# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

from .QFormerCross3 import QFormerCross3


class QFormerCross4(QFormerCross3):
    """Sixteen-query variant of the SDPA-based QFormerCross3."""

    def __init__(self,
                 feature_map,
                 model_id="QFormerCross4",
                 num_queries=16,
                 **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            num_queries=num_queries,
            **kwargs,
        )
